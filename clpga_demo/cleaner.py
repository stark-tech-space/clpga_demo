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
        # sam3_model can be a path string or an already-loaded model
        if isinstance(sam3_model, str):
            self._sam3_model_path = sam3_model
            self._sam3_model = None  # Lazy-load on first use
        else:
            self._sam3_model_path = None
            self._sam3_model = sam3_model
        self._void_model = void_model
        self._corridor_multiplier = corridor_config.get("corridor_multiplier", 4.0)
        self._corridor_speed_scale = corridor_config.get("corridor_speed_scale", 1.5)
        self._radius_scale = corridor_config.get("radius_scale", 4.0)
        self._mask_dilation_px = corridor_config.get("mask_dilation_px", 5)
        self._max_aspect_ratio = corridor_config.get("max_aspect_ratio", 2.0)
        self._max_size_ratio = corridor_config.get("max_size_ratio", 2.0)

    def _get_sam3(self):
        """Lazy-load SAM3 model if needed."""
        if self._sam3_model is None:
            from ultralytics import SAM
            logger.info("Loading SAM3 model from %s ...", self._sam3_model_path)
            self._sam3_model = SAM(self._sam3_model_path)
        return self._sam3_model

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
            # Cap corridor to avoid excessive SAM3 processing
            max_radius = min(frame_w, frame_h) / 2
            radius = min(radius, max_radius)

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

    def generate_quadmasks(
        self,
        video_frames: np.ndarray,
        corridors: list[Corridor | None],
        trajectory: list[tuple[float, float] | None],
        median_ball_size: float,
    ) -> np.ndarray:
        """Generate quadmask array for all frames using SAM3 segment-everything.

        Args:
            video_frames: (T, H, W, 3) uint8 BGR frames.
            corridors: Per-frame Corridor or None.
            trajectory: Per-frame (x, y) or None.
            median_ball_size: Median ball size for mask validation.

        Returns:
            (T, H, W) uint8 quadmask array.
        """
        num_frames, frame_h, frame_w = video_frames.shape[:3]
        quadmasks = np.full((num_frames, frame_h, frame_w), 255, dtype=np.uint8)

        for i in range(num_frames):
            corridor = corridors[i]
            traj_point = trajectory[i]

            if corridor is None or traj_point is None:
                continue

            # Run SAM3 segment-everything on the corridor region
            frame = video_frames[i]
            crop = frame[corridor.y1:corridor.y2, corridor.x1:corridor.x2]

            sam = self._get_sam3()
            # Use a point grid for segment-everything mode
            ch = corridor.y2 - corridor.y1
            cw = corridor.x2 - corridor.x1
            grid_step = max(32, min(ch, cw) // 8)  # Adaptive grid, max ~64 points
            ys_grid = list(range(grid_step // 2, ch, grid_step))
            xs_grid = list(range(grid_step // 2, cw, grid_step))
            points = [[x, y] for y in ys_grid for x in xs_grid]
            labels = [1] * len(points)
            if not points:
                continue
            results = sam(crop, points=points, labels=labels)
            if not results or results[0].masks is None or len(results[0].masks) == 0:
                continue

            # Work in crop-space to avoid full-frame mask allocation
            import cv2 as _cv2
            crop_masks_raw = results[0].masks.data.cpu().numpy().astype(bool)
            crop_masks: list[np.ndarray] = []
            for m in crop_masks_raw:
                if m.shape != (ch, cw):
                    m = _cv2.resize(m.astype(np.uint8), (cw, ch), interpolation=_cv2.INTER_NEAREST).astype(bool)
                crop_masks.append(m)
            del crop_masks_raw  # Free GPU tensor memory

            # Identify ball mask in crop-space (translate trajectory to crop coords)
            crop_traj = (traj_point[0] - corridor.x1, traj_point[1] - corridor.y1)
            ball_idx = self._identify_ball_mask_crop(crop_masks, crop_traj, median_ball_size)
            ball_mask_crop = crop_masks[ball_idx] if ball_idx is not None else None

            # Build distractor mask in crop-space (merge all except ball)
            distractor_merged = np.zeros((ch, cw), dtype=bool)
            for j, m in enumerate(crop_masks):
                if j != ball_idx:
                    distractor_merged |= m
            del crop_masks  # Free mask memory immediately

            # Build quadmask for this frame directly from crop-space data
            quadmasks[i] = self._generate_quadmask_from_crop(
                ball_mask_crop=ball_mask_crop,
                distractor_merged_crop=distractor_merged,
                corridor=corridor,
                frame_h=frame_h,
                frame_w=frame_w,
            )
            del distractor_merged, ball_mask_crop  # Free immediately

            if i % 50 == 0:
                logger.info("Quadmask generation: frame %d / %d", i, num_frames)

        return quadmasks

    def _identify_ball_mask_crop(
        self,
        crop_masks: list[np.ndarray],
        crop_trajectory_point: tuple[float, float],
        median_ball_size: float,
    ) -> int | None:
        """Identify ball mask working in crop-space coordinates."""
        if not crop_masks:
            return None

        tx, ty = crop_trajectory_point
        candidates: list[tuple[float, int]] = []

        for i, mask in enumerate(crop_masks):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            cx = float(xs.mean())
            cy = float(ys.mean())
            min_x, max_x = float(xs.min()), float(xs.max())
            min_y, max_y = float(ys.min()), float(ys.max())
            w = max_x - min_x + 1
            h = max_y - min_y + 1

            short = min(w, h)
            if short <= 0 or max(w, h) / short > self._max_aspect_ratio:
                continue

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

    def _generate_quadmask_from_crop(
        self,
        ball_mask_crop: np.ndarray | None,
        distractor_merged_crop: np.ndarray,
        corridor: Corridor,
        frame_h: int,
        frame_w: int,
    ) -> np.ndarray:
        """Generate a full-frame quadmask from crop-space masks efficiently."""
        quadmask = np.full((frame_h, frame_w), 255, dtype=np.uint8)

        if not distractor_merged_crop.any():
            return quadmask

        ch = corridor.y2 - corridor.y1
        cw = corridor.x2 - corridor.x1

        # Dilate in crop-space
        dilated = distractor_merged_crop.copy()
        if self._mask_dilation_px > 0:
            dilated = binary_dilation(dilated, iterations=self._mask_dilation_px)

        # Set distractor pixels to 0 within the corridor
        crop_quadmask = np.full((ch, cw), 127, dtype=np.uint8)  # Corridor interior = affected
        crop_quadmask[dilated] = 0  # Distractor = remove

        # Ball mask stays 255
        if ball_mask_crop is not None:
            crop_quadmask[ball_mask_crop] = 255

        # Non-distractor, non-ball corridor pixels stay 127

        # Place crop quadmask into full frame
        quadmask[corridor.y1:corridor.y2, corridor.x1:corridor.x2] = crop_quadmask

        return quadmask

    def clean_segments(
        self,
        video_frames: np.ndarray,
        quadmasks: np.ndarray,
        segments: list[tuple[int, int]],
        prompt: str,
    ) -> list[np.ndarray]:
        """Run void-model inpainting on each video segment.

        Args:
            video_frames: (T, H, W, 3) uint8 full video frames.
            quadmasks: (T, H, W) uint8 quadmask for each frame.
            segments: List of (start, end) frame ranges.
            prompt: Background description for void-model.

        Returns:
            List of (seg_len, H, W, 3) uint8 cleaned segment arrays.
        """
        cleaned: list[np.ndarray] = []

        for start, end in segments:
            seg_video = video_frames[start:end]
            seg_mask = quadmasks[start:end]

            if np.all(seg_mask == 255):
                logger.debug("Segment %d-%d: no distractors, skipping inpainting", start, end)
                cleaned.append(seg_video.copy())
                continue

            logger.info("Cleaning segment frames %d-%d with void-model", start, end)
            result = self._void_model.inpaint(seg_video, seg_mask, prompt)
            cleaned.append(result)

        return cleaned

    @staticmethod
    def split_into_segments(
        total_frames: int,
        max_frames: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """Split a frame range into overlapping segments for void-model."""
        if total_frames <= max_frames:
            return [(0, total_frames)]

        segments: list[tuple[int, int]] = []
        start = 0
        stride = max_frames - overlap

        while start < total_frames:
            end = min(start + max_frames, total_frames)
            segments.append((start, end))
            if end == total_frames:
                break
            start += stride

        return segments

    @staticmethod
    def blend_segments(
        seg_frames: list[np.ndarray],
        segments: list[tuple[int, int]],
        total_frames: int,
    ) -> np.ndarray:
        """Blend overlapping video segments using linear crossfade.

        In overlap zones, weights transition linearly from 1.0/0.0 to 0.0/1.0
        between preceding and following segments.
        """
        if len(seg_frames) == 1:
            return seg_frames[0]

        h, w, c = seg_frames[0].shape[1:]
        result = np.zeros((total_frames, h, w, c), dtype=np.float32)
        weights = np.zeros((total_frames, 1, 1, 1), dtype=np.float32)

        for seg_arr, (start, end) in zip(seg_frames, segments):
            seg_len = end - start
            w_arr = np.ones(seg_len, dtype=np.float32)

            # Ramp up at start if overlapping with previous segment
            if start > 0:
                prev_end = None
                for ps, pe in segments:
                    if pe > start and ps < start:
                        prev_end = pe
                        break
                if prev_end is not None:
                    overlap_len = prev_end - start
                    ramp = np.linspace(0, 1, overlap_len + 2)[1:-1]
                    w_arr[:overlap_len] = ramp

            # Ramp down at end if overlapping with next segment
            next_start = None
            for ns, ne in segments:
                if ns > start and ns < end:
                    next_start = ns
                    break
            if next_start is not None:
                overlap_len = end - next_start
                ramp = np.linspace(1, 0, overlap_len + 2)[1:-1]
                w_arr[seg_len - overlap_len:] = ramp

            w_shaped = w_arr.reshape(-1, 1, 1, 1)
            result[start:end] += seg_arr.astype(np.float32) * w_shaped
            weights[start:end] += w_shaped

        weights = np.maximum(weights, 1e-8)
        result = (result / weights).clip(0, 255).astype(np.uint8)
        return result
