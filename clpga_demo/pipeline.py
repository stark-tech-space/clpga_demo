"""Pipeline orchestration — tracker -> smoother -> cropper."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from clpga_demo.cropper import VideoWriter, calculate_crop
from clpga_demo.momentum import create_tracker
from clpga_demo.smoother import GaussianSmoother
from clpga_demo.tracker import select_ball, track_video

logger = logging.getLogger(__name__)


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
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
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
        kalman_process_noise=kalman_process_noise,
        kalman_measurement_noise=kalman_measurement_noise,
        kalman_gate_threshold=kalman_gate_threshold,
    )

    # --- Pass 1: Collect positions with momentum filtering ---
    positions: list[tuple[float, float] | None] = []
    selected_obj_id: int | None = None
    frames_since_lost = 0

    for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence, text=text):
        result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)

        accepted = False
        if result is not None:
            if frames_since_lost > 0 and tracker.has_position:
                # Re-acquisition: check proximity to tracker prediction
                ball_w = result.bbox[2] - result.bbox[0]
                ball_h = result.bbox[3] - result.bbox[1]
                ball_size = (ball_w + ball_h) / 2
                if not tracker.accept((result.center_x, result.center_y), ball_size):
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
            pos = tracker.update((result.center_x, result.center_y))
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

    if all(p is None for p in positions):
        raise RuntimeError("No golf ball detected in video")

    # --- Pass 2: Smooth and crop ---
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
