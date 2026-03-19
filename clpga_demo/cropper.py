"""Portrait crop calculation and video writing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropRegion:
    """A rectangular crop region within a video frame."""

    x: int
    y: int
    width: int
    height: int

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Extract this crop region from a frame (H, W, C) array."""
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


def calculate_crop(
    ball_x: float,
    ball_y: float,
    source_width: int,
    source_height: int,
) -> CropRegion:
    """Calculate a 9:16 portrait crop region centered on the ball position.

    For landscape/wide sources: uses full height, crops width to 9:16.
    For narrow sources (already narrower than 9:16): uses full width, crops height to 9:16.
    """
    target_w = int(source_height * 9 / 16)

    if target_w <= source_width:
        # Landscape or wide enough — crop horizontally, use full height
        crop_w = target_w
        crop_h = source_height
        crop_x = int(max(0, min(ball_x - crop_w / 2, source_width - crop_w)))
        crop_y = 0
    else:
        # Source is narrower than 9:16 — crop vertically, use full width
        crop_w = source_width
        crop_h = int(source_width * 16 / 9)
        crop_x = 0
        crop_y = int(max(0, min(ball_y - crop_h / 2, source_height - crop_h)))

    return CropRegion(x=crop_x, y=crop_y, width=crop_w, height=crop_h)


class VideoWriter:
    """Writes cropped frames to an output video file."""

    def __init__(self, output_path: str, fps: float, width: int, height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise ValueError(f"Cannot open video writer for {output_path}")

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame (H, W, C) BGR array."""
        self._writer.write(frame)

    def release(self) -> None:
        """Finalize and close the output video."""
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
