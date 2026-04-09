"""Pipeline orchestration — tracker -> smoother -> cropper."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from clpga_demo.cleaner import FrameCleaner
from clpga_demo.cropper import VideoWriter, calculate_crop
from clpga_demo.momentum import create_tracker
from clpga_demo.scene_analyzer import SceneAnalyzer
from clpga_demo.smoother import GaussianSmoother
from clpga_demo.tracker import select_ball, track_video
from clpga_demo.void_model import VoidModelWrapper

logger = logging.getLogger(__name__)


def _save_debug_video(path: str, frames: np.ndarray, fps: float) -> None:
    """Write a numpy frame array to a video file for debugging."""
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(len(frames)):
        writer.write(frames[i])
    writer.release()


def _retrack_cleaned(
    cleaned_frames: np.ndarray,
    model: str,
    confidence: float,
    text: list[str] | None,
    tracker,
    src_w: int,
    src_h: int,
    fps: float,
) -> list[tuple[float, float] | None]:
    """Re-run detection and tracking on cleaned frames.

    Writes cleaned frames to a temp video file, then uses the standard
    track_video pipeline (SAM3VideoSemanticPredictor) which requires a video source.
    """
    import tempfile

    # Write cleaned frames to a temporary video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    h, w = cleaned_frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for i in range(len(cleaned_frames)):
        writer.write(cleaned_frames[i])
    writer.release()

    # Re-run standard tracking pipeline on cleaned video
    positions: list[tuple[float, float] | None] = []
    selected_obj_id: int | None = None
    frames_since_lost = 0

    try:
        for frame_idx, orig_frame, boxes in track_video(tmp_path, model=model, confidence=confidence, text=text):
            result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)

            accepted = False
            if result is not None:
                if frames_since_lost > 0 and tracker.has_position:
                    if not tracker.accept((result.center_x, result.center_y), result.bbox):
                        result = None
                    else:
                        accepted = True
                else:
                    accepted = True

            if accepted and result is not None:
                if selected_obj_id is None:
                    selected_obj_id = result.obj_id
                pos = tracker.update((result.center_x, result.center_y), result.bbox)
                positions.append(pos)
                frames_since_lost = 0
            else:
                if tracker.has_position:
                    pos = tracker.predict()
                    positions.append(pos)
                else:
                    positions.append(None)
                frames_since_lost += 1
                if fps > 0 and frames_since_lost > fps * 3:
                    selected_obj_id = None
                    tracker.reset()
                    frames_since_lost = 0
    finally:
        import os
        os.unlink(tmp_path)

    return positions


def process_video(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_sigma_seconds: float = 0.5,
    text: list[str] | None = None,
    tracker_type: str = "momentum",
    momentum_history_size: int = 5,
    momentum_radius_scale: float = 4.0,
    momentum_confirm_frames: int = 3,
    momentum_max_size_ratio: float = 2.0,
    momentum_max_aspect_ratio: float = 2.0,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
    clean: bool = False,
    corridor_multiplier: float = 4.0,
    corridor_speed_scale: float = 1.5,
    mask_dilation_px: int = 5,
    segment_max_frames: int = 180,
    segment_overlap_frames: int = 16,
    void_model_dir: str | None = None,
    clean_prompt: str = "golf course background",
    gemini_model: str = "gemini-2.5-flash-preview-05-20",
) -> None:
    """Process a pre-recorded video: track ball, smooth trajectory, crop portrait.

    Two-pass pipeline:
      Pass 1 — collect all ball positions from SAM3 tracking with momentum filtering.
      Pass 2 — smooth positions, re-read video, write cropped output.
    """
    if not Path(source).exists():
        raise ValueError(f"Input video does not exist: {source}")

    # --- Get video properties ---
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    clip_duration = frame_count / fps if fps > 0 else 1.0
    tracker = create_tracker(
        tracker_type,
        clip_duration_seconds=clip_duration,
        fps=fps,
        momentum_history_size=momentum_history_size,
        momentum_radius_scale=momentum_radius_scale,
        momentum_confirm_frames=momentum_confirm_frames,
        momentum_max_size_ratio=momentum_max_size_ratio,
        momentum_max_aspect_ratio=momentum_max_aspect_ratio,
        kalman_process_noise=kalman_process_noise,
        kalman_measurement_noise=kalman_measurement_noise,
        kalman_gate_threshold=kalman_gate_threshold,
    )

    # --- Pass 1: Collect positions with momentum filtering ---
    positions: list[tuple[float, float] | None] = []
    ball_sizes: list[float] = []
    speeds_list: list[float] = []
    selected_obj_id: int | None = None
    frames_since_lost = 0

    for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence, text=text):
        result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)

        accepted = False
        if result is not None:
            if frames_since_lost > 0 and tracker.has_position:
                # Re-acquisition: check proximity to tracker prediction
                if not tracker.accept((result.center_x, result.center_y), result.bbox):
                    logger.debug(
                        "Frame %d: rejected detection obj_id=%d — too far from tracker prediction",
                        frame_idx, result.obj_id,
                    )
                    result = None  # Treat as no detection
                else:
                    accepted = True
            else:
                accepted = True

        if accepted and result is not None:
            if selected_obj_id is None:
                selected_obj_id = result.obj_id
                logger.info(f"Selected ball obj_id={result.obj_id} at frame {frame_idx}")
            pos = tracker.update((result.center_x, result.center_y), result.bbox)
            positions.append(pos)
            w_box = result.bbox[2] - result.bbox[0]
            h_box = result.bbox[3] - result.bbox[1]
            ball_sizes.append((w_box + h_box) / 2)
            speeds_list.append(tracker.speed)
            frames_since_lost = 0
        else:
            if tracker.has_position:
                pos = tracker.predict()
                positions.append(pos)
            else:
                positions.append(None)
            ball_sizes.append(ball_sizes[-1] if ball_sizes else 0.0)
            speeds_list.append(tracker.speed)
            frames_since_lost += 1
            if fps > 0 and frames_since_lost > fps * 3:
                selected_obj_id = None
                tracker.reset()
                frames_since_lost = 0

    if all(p is None for p in positions):
        raise RuntimeError("No golf ball detected in video")

    # --- Pass 2 (optional): Clean distractors and re-track ---
    if clean:
        void_wrapper = VoidModelWrapper(model_dir=void_model_dir)
        void_wrapper.download_if_needed()
        void_wrapper.load()

        corridor_config = {
            "corridor_multiplier": corridor_multiplier,
            "corridor_speed_scale": corridor_speed_scale,
            "radius_scale": momentum_radius_scale,
            "mask_dilation_px": mask_dilation_px,
            "max_aspect_ratio": momentum_max_aspect_ratio,
            "max_size_ratio": momentum_max_size_ratio,
        }
        cleaner = FrameCleaner(sam3_model=model, void_model=void_wrapper, corridor_config=corridor_config)
        corridors = cleaner.compute_corridors(positions, ball_sizes, speeds_list, frame_w=src_w, frame_h=src_h)

        # Read all frames into memory for cleaning
        cap = cv2.VideoCapture(source)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        video_frames = np.stack(all_frames)

        # Compute median ball size for mask identification
        valid_sizes = [s for s in ball_sizes if s > 0]
        median_ball_size = float(sorted(valid_sizes)[len(valid_sizes) // 2]) if valid_sizes else 20.0

        # Generate quadmasks per segment using VLM scene analysis
        segments = FrameCleaner.split_into_segments(frame_count, segment_max_frames, segment_overlap_frames)
        analyzer = SceneAnalyzer(model=gemini_model)

        all_quadmasks = np.full((frame_count, src_h, src_w), 255, dtype=np.uint8)
        segment_prompts: list[str] = []

        for seg_start, seg_end in segments:
            mid_idx = seg_start + (seg_end - seg_start) // 2
            keyframe = video_frames[mid_idx]
            logger.info("Analyzing keyframe %d for segment %d-%d ...", mid_idx, seg_start, seg_end)
            analysis = analyzer.analyze_frame(keyframe)
            segment_prompts.append(analysis.scene_description)

            logger.info(
                "Scene: %s | Ball: %s | Distractors: %d",
                analysis.scene_description,
                analysis.ball_bbox,
                len(analysis.distractors),
            )

            seg_frames = video_frames[seg_start:seg_end]
            seg_corridors = corridors[seg_start:seg_end]
            seg_quadmasks = cleaner.generate_quadmasks_targeted(
                seg_frames, seg_corridors, analysis, median_ball_size,
            )
            all_quadmasks[seg_start:seg_end] = seg_quadmasks

        # Run void-model per segment with scene-aware prompts
        cleaned_segments = []
        for (seg_start, seg_end), prompt in zip(segments, segment_prompts):
            seg_video = video_frames[seg_start:seg_end]
            seg_mask = all_quadmasks[seg_start:seg_end]

            if np.all(seg_mask == 255):
                logger.debug("Segment %d-%d: no distractors, skipping", seg_start, seg_end)
                cleaned_segments.append(seg_video.copy())
                continue

            logger.info("Cleaning segment %d-%d: %s", seg_start, seg_end, prompt)
            result = void_wrapper.inpaint(seg_video, seg_mask, prompt)
            cleaned_segments.append(result)

        cleaned_video = FrameCleaner.blend_segments(cleaned_segments, segments, frame_count)

        # Save intermediate outputs for debugging
        output_dir = Path(output).parent
        stem = Path(output).stem
        _save_debug_video(
            str(output_dir / f"{stem}_cleaned.mp4"),
            cleaned_video, fps,
        )
        _save_debug_video(
            str(output_dir / f"{stem}_quadmask.mp4"),
            np.stack([cv2.cvtColor(q, cv2.COLOR_GRAY2BGR) for q in all_quadmasks]),
            fps,
        )
        logger.info("Saved debug videos: %s_cleaned.mp4, %s_quadmask.mp4", stem, stem)

        # Re-track on cleaned frames
        tracker2 = create_tracker(
            tracker_type,
            clip_duration_seconds=clip_duration,
            fps=fps,
            momentum_history_size=momentum_history_size,
            momentum_radius_scale=momentum_radius_scale,
            momentum_confirm_frames=momentum_confirm_frames,
            momentum_max_size_ratio=momentum_max_size_ratio,
            momentum_max_aspect_ratio=momentum_max_aspect_ratio,
            kalman_process_noise=kalman_process_noise,
            kalman_measurement_noise=kalman_measurement_noise,
            kalman_gate_threshold=kalman_gate_threshold,
        )

        refined_positions = _retrack_cleaned(
            cleaned_video, model, confidence, text, tracker2, src_w, src_h, fps,
        )

        if not all(p is None for p in refined_positions):
            positions = refined_positions
        else:
            logger.warning("Cleaning pass produced no detections; falling back to rough trajectory")

    # --- Pass 3: Smooth and crop ---
    pos_array = np.array([p if p is not None else (np.nan, np.nan) for p in positions])
    smoother = GaussianSmoother.from_fps(fps=fps, sigma_seconds=smoothing_sigma_seconds)
    smoothed = smoother.smooth(pos_array)

    crop_sample = calculate_crop(smoothed[0][0], smoothed[0][1], src_w, src_h)

    with VideoWriter(output, fps, crop_sample.width, crop_sample.height) as writer:
        cap = cv2.VideoCapture(source)
        for i in range(len(smoothed)):
            ret, frame = cap.read()
            if not ret:
                break
            crop = calculate_crop(smoothed[i][0], smoothed[i][1], src_w, src_h)
            cropped = crop.apply(frame)
            writer.write(cropped)
        cap.release()

    logger.info(f"Output written to {output}")
