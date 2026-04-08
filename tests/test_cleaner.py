import numpy as np
import pytest

from clpga_demo.cleaner import FrameCleaner, Corridor


class TestCorridorComputation:
    def test_corridor_size_scales_with_ball_size(self):
        """Corridor radius should be at least ball_size * corridor_multiplier."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(100.0, 100.0), (100.0, 100.0), (100.0, 100.0), (100.0, 100.0), (100.0, 100.0)]
        ball_sizes = [20.0, 20.0, 20.0, 20.0, 20.0]
        speeds = [0.0, 0.0, 0.0, 0.0, 0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert len(corridors) == 5
        for c in corridors:
            assert c.radius == 80.0
            assert c.center_x == 100.0
            assert c.center_y == 100.0

    def test_corridor_size_scales_with_speed(self):
        """Corridor radius should grow with ball speed."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(100.0, 100.0)]
        ball_sizes = [20.0]
        speeds = [50.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        # speed * corridor_speed_scale * radius_scale = 50 * 1.5 * 4.0 = 300
        assert corridors[0].radius == 300.0

    def test_corridor_clips_to_frame_bounds(self):
        """Corridor bbox should be clipped to frame edges."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(10.0, 10.0)]
        ball_sizes = [20.0]
        speeds = [0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert corridors[0].x1 >= 0
        assert corridors[0].y1 >= 0

    def test_corridor_none_trajectory_point_skipped(self):
        """Frames where trajectory is None should produce a None corridor."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(100.0, 100.0), None, (120.0, 100.0)]
        ball_sizes = [20.0, 0.0, 20.0]
        speeds = [0.0, 0.0, 0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert corridors[0] is not None
        assert corridors[1] is None
        assert corridors[2] is not None


class TestBallMaskIdentification:
    def _make_cleaner(self):
        return FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

    def test_selects_closest_mask_to_trajectory(self):
        """Should pick the mask whose centroid is closest to trajectory point."""
        cleaner = self._make_cleaner()
        mask_near = np.zeros((480, 640), dtype=bool)
        mask_near[90:110, 90:110] = True
        mask_far = np.zeros((480, 640), dtype=bool)
        mask_far[290:310, 290:310] = True
        masks = [mask_near, mask_far]
        result = cleaner.identify_ball_mask(masks, (100.0, 100.0), 20.0)
        assert result == 0

    def test_rejects_mask_with_bad_aspect_ratio(self):
        """Closest mask should be rejected if aspect ratio is too large."""
        cleaner = self._make_cleaner()
        mask_elongated = np.zeros((480, 640), dtype=bool)
        mask_elongated[99:100, 80:120] = True  # 1px tall, 40px wide
        mask_round = np.zeros((480, 640), dtype=bool)
        mask_round[190:210, 190:210] = True
        masks = [mask_elongated, mask_round]
        result = cleaner.identify_ball_mask(masks, (100.0, 100.0), 20.0)
        assert result == 1

    def test_rejects_mask_with_bad_size_ratio(self):
        """Closest mask should be rejected if too large compared to median ball size."""
        cleaner = self._make_cleaner()
        mask_big = np.zeros((480, 640), dtype=bool)
        mask_big[50:150, 50:150] = True  # 100x100 vs median 20
        mask_right = np.zeros((480, 640), dtype=bool)
        mask_right[195:205, 195:205] = True
        masks = [mask_big, mask_right]
        result = cleaner.identify_ball_mask(masks, (100.0, 100.0), 20.0)
        assert result == 1

    def test_returns_none_when_no_valid_masks(self):
        """Should return None if all masks fail validation."""
        cleaner = self._make_cleaner()
        mask_huge = np.zeros((480, 640), dtype=bool)
        mask_huge[0:200, 0:200] = True
        masks = [mask_huge]
        result = cleaner.identify_ball_mask(masks, (100.0, 100.0), 20.0)
        assert result is None

    def test_empty_masks_returns_none(self):
        """Should return None for empty mask list."""
        cleaner = self._make_cleaner()
        result = cleaner.identify_ball_mask([], (100.0, 100.0), 20.0)
        assert result is None


class TestQuadmaskGeneration:
    def _make_cleaner(self):
        return FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

    def test_ball_mask_region_is_255(self):
        """Ball mask pixels should be 255 (keep) in the quadmask."""
        cleaner = self._make_cleaner()
        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[200:220, 200:220] = True
        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)
        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask, distractor_masks=[distractor_mask],
            corridor=corridor, frame_h=480, frame_w=640,
        )
        assert quadmask.shape == (480, 640)
        assert quadmask.dtype == np.uint8
        assert np.all(quadmask[90:110, 90:110] == 255)

    def test_distractor_region_is_0(self):
        """Distractor mask pixels (after dilation) should be 0 (remove)."""
        cleaner = self._make_cleaner()
        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[140:160, 140:160] = True  # Inside corridor
        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)
        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask, distractor_masks=[distractor_mask],
            corridor=corridor, frame_h=480, frame_w=640,
        )
        assert np.all(quadmask[142:158, 142:158] == 0)

    def test_outside_corridor_is_255(self):
        """Everything outside the corridor should be 255 (keep)."""
        cleaner = self._make_cleaner()
        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)
        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=None, distractor_masks=[], corridor=corridor, frame_h=480, frame_w=640,
        )
        assert np.all(quadmask[0:20, :] == 255)
        assert np.all(quadmask[180:, :] == 255)
        assert np.all(quadmask[:, 0:20] == 255)
        assert np.all(quadmask[:, 180:] == 255)

    def test_no_distractors_returns_all_255(self):
        """If no distractor masks, entire quadmask should be 255."""
        cleaner = self._make_cleaner()
        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True
        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)
        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask, distractor_masks=[], corridor=corridor, frame_h=480, frame_w=640,
        )
        assert np.all(quadmask == 255)
