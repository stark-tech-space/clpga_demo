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
