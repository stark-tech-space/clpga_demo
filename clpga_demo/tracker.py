"""SAM3 ball detection, tracking, and ball selection heuristic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackResult:
    """Tracking result for a single frame."""

    frame_idx: int
    center_x: float
    center_y: float
    bbox: tuple[float, float, float, float]
    confidence: float
    obj_id: int


def select_ball(
    boxes: np.ndarray,
    frame_width: int,
    frame_height: int,
    preferred_obj_id: int | None = None,
    frame_idx: int = 0,
) -> TrackResult | None:
    """Select the best golf ball from detected boxes.

    Args:
        boxes: (N, 7) array with columns [x1, y1, x2, y2, obj_id, score, cls].
        frame_width: Source frame width.
        frame_height: Source frame height.
        preferred_obj_id: If set, prefer this obj_id (sticky selection).
        frame_idx: Current frame index for the TrackResult.

    Returns:
        TrackResult for the selected ball, or None if no detections.
    """
    if len(boxes) == 0:
        return None

    # Sticky selection: if preferred obj_id is present, use it
    if preferred_obj_id is not None:
        for row in boxes:
            if int(row[4]) == preferred_obj_id:
                return _box_to_result(row, frame_idx=frame_idx)

    # Heuristic: score = bbox_area / distance_from_center
    frame_cx = frame_width / 2
    frame_cy = frame_height / 2
    best_score = -1.0
    best_row = None

    for row in boxes:
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        dist = max(np.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2), 1.0)
        score = area / dist
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return None
    return _box_to_result(best_row, frame_idx=frame_idx)


def _box_to_result(row: np.ndarray, frame_idx: int) -> TrackResult:
    """Convert a single box row to a TrackResult."""
    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    return TrackResult(
        frame_idx=frame_idx,
        center_x=(x1 + x2) / 2,
        center_y=(y1 + y2) / 2,
        bbox=(x1, y1, x2, y2),
        confidence=float(row[5]),
        obj_id=int(row[4]),
    )


_FALLBACK_FRAME_THRESHOLD = 30


def track_video(
    source: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    text: list[str] | None = None,
) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
    """Run SAM3 tracking on a video source.

    Yields (frame_idx, original_frame, boxes) for each frame.
    boxes is an (N, 7) array: [x1, y1, x2, y2, obj_id, score, cls].

    If no detections are found in the first 30 frames via text prompt,
    falls back to bbox prompt at center of frame.
    """
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    if text is None:
        text = ["golf ball"]

    overrides = dict(
        conf=confidence,
        task="segment",
        mode="predict",
        model=model,
        half=True,
        save=False,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    results = predictor(source=source, text=text, stream=True)

    any_detection = False
    buffered_frames: list[tuple[int, np.ndarray, np.ndarray]] = []

    for frame_idx, r in enumerate(results):
        boxes = r.boxes.data.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.empty((0, 7))
        orig_frame = r.orig_img

        if len(boxes) > 0:
            any_detection = True

        # Buffer first N frames to check for fallback
        if not any_detection and frame_idx < _FALLBACK_FRAME_THRESHOLD:
            buffered_frames.append((frame_idx, orig_frame, boxes))
            continue

        # Fallback: no detections in first 30 frames — re-run with bbox prompt
        if not any_detection and frame_idx == _FALLBACK_FRAME_THRESHOLD:
            logger.warning(
                "No golf ball detected in first %d frames via text prompt. "
                "Falling back to center-of-frame bounding box.",
                _FALLBACK_FRAME_THRESHOLD,
            )
            h, w = orig_frame.shape[:2]
            cx, cy = w / 2, h / 2
            size = min(w, h) / 4
            center_bbox = [[cx - size, cy - size, cx + size, cy + size]]
            predictor.add_prompt(frame_idx=0, bboxes=np.array(center_bbox))
            # Yield buffered frames (they had no detections)
            for buffered in buffered_frames:
                yield buffered
            buffered_frames.clear()

        yield frame_idx, orig_frame, boxes

    # If we never hit the threshold, yield any remaining buffered frames
    for buffered in buffered_frames:
        yield buffered
