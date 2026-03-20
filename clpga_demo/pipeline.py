"""Pipeline orchestration — tracker -> smoother -> cropper."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from clpga_demo.cropper import VideoWriter, calculate_crop
from clpga_demo.smoother import EMASmoother, GaussianSmoother
from clpga_demo.tracker import select_ball, track_video

logger = logging.getLogger(__name__)


def process_video(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_sigma_seconds: float = 0.5,
    text: list[str] | None = None,
) -> None:
    """Process a pre-recorded video: track ball, smooth trajectory, crop portrait.

    Two-pass pipeline:
      Pass 1 — collect all ball positions from SAM3 tracking.
      Pass 2 — smooth positions, re-read video, write cropped output.
    """
    if not Path(source).exists():
        raise ValueError(f"Input video does not exist: {source}")

    # --- Pass 1: Collect positions ---
    positions: list[tuple[float, float] | None] = []
    selected_obj_id: int | None = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence, text=text):
        result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)
        if result is not None:
            if selected_obj_id is None:
                selected_obj_id = result.obj_id
                logger.info(f"Selected ball obj_id={result.obj_id} at frame {frame_idx}")
            positions.append((result.center_x, result.center_y))
        else:
            positions.append(None)

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


def process_stream(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_alpha: float = 0.15,
    text: list[str] | None = None,
) -> None:
    """Process a live stream: track ball, smooth with EMA, crop portrait in real-time.

    Single-pass pipeline — each frame is tracked, smoothed, and cropped immediately.
    """
    import time

    smoother = EMASmoother(alpha=smoothing_alpha)
    selected_obj_id: int | None = None
    writer: VideoWriter | None = None
    frames_since_lost = 0
    fps_for_loss = 30.0  # will be updated from source
    max_retries = 3

    try:
        retries = 0
        while retries <= max_retries:
            try:
                for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence, text=text):
                    src_h, src_w = orig_frame.shape[:2]

                    result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)

                    if result is not None:
                        selected_obj_id = result.obj_id
                        sx, sy = smoother.update(result.center_x, result.center_y)
                        frames_since_lost = 0
                    else:
                        sx, sy = smoother.update(None, None)
                        frames_since_lost += 1
                        # Re-evaluate heuristic after 3 seconds of loss
                        if frames_since_lost > fps_for_loss * 3:
                            selected_obj_id = None
                            frames_since_lost = 0

                    crop = calculate_crop(sx, sy, src_w, src_h)

                    if writer is None:
                        cap_temp = cv2.VideoCapture(source)
                        fps_for_loss = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
                        cap_temp.release()
                        writer = VideoWriter(output, fps_for_loss, crop.width, crop.height)

                    cropped = crop.apply(orig_frame)
                    writer.write(cropped)

                break  # Normal completion

            except (cv2.error, ConnectionError, OSError) as e:
                retries += 1
                if retries > max_retries:
                    logger.warning("Stream disconnected after %d retries. Finalizing output.", max_retries)
                    break
                logger.warning("Stream error (attempt %d/%d): %s. Retrying in 1s...", retries, max_retries, e)
                time.sleep(1)

    finally:
        if writer is not None:
            writer.release()

    if writer is None:
        raise RuntimeError("No frames received from stream")

    logger.info(f"Stream output written to {output}")
