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


class TestExponentialDecay:
    def test_predict_advances_position(self):
        """predict() should move position along velocity vector."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))  # vx=10, vy=0
        px, py = mt.predict()
        # Position should advance beyond 110 (last known)
        assert px > 110.0
        assert py == pytest.approx(100.0, abs=0.1)

    def test_predict_decays_velocity(self):
        """Each predict() call should reduce velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))
        speed_before = mt.speed
        mt.predict()
        assert mt.speed < speed_before

    def test_velocity_near_zero_at_clip_end(self):
        """After clip_duration worth of frames, velocity should be ~5% of initial."""
        fps = 30.0
        duration = 10.0
        mt = MomentumTracker(clip_duration_seconds=duration, fps=fps)
        mt.update((0.0, 0.0))
        mt.update((10.0, 0.0))  # vx=10
        initial_speed = mt.speed
        total_frames = int(duration * fps)
        for _ in range(total_frames):
            mt.predict()
        assert mt.speed == pytest.approx(initial_speed * 0.05, rel=0.01)

    def test_short_clip_duration_clamped(self):
        """Clip duration < 1.0 should be clamped to 1.0, not explode."""
        mt = MomentumTracker(clip_duration_seconds=0.1, fps=30.0)
        mt.update((0.0, 0.0))
        mt.update((10.0, 0.0))
        # Should not raise, and velocity should not be instantly zero
        mt.predict()
        assert mt.speed > 0.0

    def test_predict_returns_position(self):
        """predict() should return the predicted (x, y) tuple."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0))
        mt.update((110.0, 205.0))
        result = mt.predict()
        assert isinstance(result, tuple)
        assert len(result) == 2
