import math
import pytest
from clpga_demo.momentum import MomentumTracker


class TestVelocityEstimation:
    def test_velocity_zero_with_single_position(self):
        """Single position gives zero velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0))
        assert mt.velocity == (0.0, 0.0)
        assert mt.speed == 0.0

    def test_velocity_from_two_positions(self):
        """Two positions give exact velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0))
        mt.update((110.0, 205.0))
        assert mt.velocity == pytest.approx((10.0, 5.0))

    def test_velocity_weighted_average(self):
        """With 3 positions, most recent delta weighted higher."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, history_size=5)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))  # delta: (10, 0) weight=1
        mt.update((130.0, 100.0))  # delta: (20, 0) weight=2
        # weighted avg = (10*1 + 20*2) / (1+2) = 50/3 ≈ 16.667
        vx, vy = mt.velocity
        assert vx == pytest.approx(50.0 / 3.0)
        assert vy == pytest.approx(0.0)

    def test_speed_magnitude(self):
        """Speed is the magnitude of velocity vector."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((0.0, 0.0))
        mt.update((3.0, 4.0))
        assert mt.speed == pytest.approx(5.0)

    def test_history_rolls_over(self):
        """Old positions drop out of the rolling buffer."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, history_size=3)
        mt.update((0.0, 0.0))
        mt.update((10.0, 0.0))   # delta: 10
        mt.update((20.0, 0.0))   # delta: 10
        mt.update((50.0, 0.0))   # delta: 30 — oldest (0,0) dropped
        # history is now [(10,0), (20,0), (50,0)]
        # deltas: (10, 0) weight=1, (30, 0) weight=2
        # weighted avg = (10*1 + 30*2) / 3 = 70/3 ≈ 23.33
        vx, _ = mt.velocity
        assert vx == pytest.approx(70.0 / 3.0)
