"""Distractor removal via SAM3 segment-everything + void-model inpainting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Corridor:
    """A rectangular region around a trajectory point where distractors are searched."""

    center_x: float
    center_y: float
    radius: float
    x1: int
    y1: int
    x2: int
    y2: int


class FrameCleaner:
    """Orchestrates distractor removal: corridor computation, segmentation, quadmask generation."""

    def __init__(
        self,
        sam3_model,
        void_model,
        corridor_config: dict,
    ) -> None:
        self._sam3_model = sam3_model
        self._void_model = void_model
        self._corridor_multiplier = corridor_config.get("corridor_multiplier", 4.0)
        self._corridor_speed_scale = corridor_config.get("corridor_speed_scale", 1.5)
        self._radius_scale = corridor_config.get("radius_scale", 4.0)
        self._mask_dilation_px = corridor_config.get("mask_dilation_px", 5)
        self._max_aspect_ratio = corridor_config.get("max_aspect_ratio", 2.0)
        self._max_size_ratio = corridor_config.get("max_size_ratio", 2.0)

    def compute_corridors(
        self,
        rough_trajectory: list[tuple[float, float] | None],
        ball_sizes: list[float],
        speeds: list[float],
        frame_w: int,
        frame_h: int,
    ) -> list[Corridor | None]:
        """Compute per-frame corridors from a rough trajectory."""
        corridors: list[Corridor | None] = []
        for pos, size, speed in zip(rough_trajectory, ball_sizes, speeds):
            if pos is None:
                corridors.append(None)
                continue

            cx, cy = pos
            size_radius = size * self._corridor_multiplier
            speed_radius = speed * self._corridor_speed_scale * self._radius_scale
            radius = max(size_radius, speed_radius)

            x1 = max(0, int(cx - radius))
            y1 = max(0, int(cy - radius))
            x2 = min(frame_w, int(cx + radius))
            y2 = min(frame_h, int(cy + radius))

            corridors.append(Corridor(
                center_x=cx,
                center_y=cy,
                radius=radius,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

        return corridors

    def identify_ball_mask(
        self,
        masks: list[np.ndarray],
        trajectory_point: tuple[float, float],
        median_ball_size: float,
    ) -> int | None:
        """Identify which mask is the ball by proximity to trajectory point.

        Returns index of ball mask, or None if no mask passes validation.
        """
        if not masks:
            return None

        tx, ty = trajectory_point
        candidates: list[tuple[float, int]] = []

        for i, mask in enumerate(masks):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            cx = float(xs.mean())
            cy = float(ys.mean())
            min_x, max_x = float(xs.min()), float(xs.max())
            min_y, max_y = float(ys.min()), float(ys.max())
            w = max_x - min_x + 1
            h = max_y - min_y + 1

            # Aspect ratio gate
            short = min(w, h)
            if short <= 0 or max(w, h) / short > self._max_aspect_ratio:
                continue

            # Size consistency gate
            mask_size = (w + h) / 2
            if median_ball_size > 0:
                ratio = mask_size / median_ball_size
                if ratio > self._max_size_ratio or ratio < 1.0 / self._max_size_ratio:
                    continue

            dist = float(np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2))
            candidates.append((dist, i))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][1]

    def generate_quadmask_frame(
        self,
        ball_mask: np.ndarray | None,
        distractor_masks: list[np.ndarray],
        corridor: Corridor,
        frame_h: int,
        frame_w: int,
    ) -> np.ndarray:
        """Generate a single-frame quadmask for void-model.

        Args:
            ball_mask: (H, W) boolean mask of the ball, or None.
            distractor_masks: List of (H, W) boolean masks to remove.
            corridor: The corridor for this frame.
            frame_h: Frame height.
            frame_w: Frame width.

        Returns:
            (H, W) uint8 quadmask: 0=remove, 127=affected, 255=keep.
        """
        quadmask = np.full((frame_h, frame_w), 255, dtype=np.uint8)

        if not distractor_masks:
            return quadmask

        # Merge distractor masks
        merged = np.zeros((frame_h, frame_w), dtype=bool)
        for m in distractor_masks:
            merged |= m

        # Clip to corridor bounds
        corridor_mask = np.zeros((frame_h, frame_w), dtype=bool)
        corridor_mask[corridor.y1:corridor.y2, corridor.x1:corridor.x2] = True
        merged &= corridor_mask

        # Dilate to avoid artifact halos
        if self._mask_dilation_px > 0:
            merged = binary_dilation(merged, iterations=self._mask_dilation_px)
            merged &= corridor_mask  # Re-clip after dilation

        # Set distractor region to 0 (remove)
        quadmask[merged] = 0

        # Set corridor boundary to 127 (affected)
        boundary = corridor_mask & ~merged
        if ball_mask is not None:
            boundary &= ~ball_mask
        quadmask[boundary & (quadmask == 255)] = 127

        # Ensure ball region stays 255
        if ball_mask is not None:
            quadmask[ball_mask] = 255

        # Ensure outside corridor is 255
        quadmask[~corridor_mask] = 255

        return quadmask
