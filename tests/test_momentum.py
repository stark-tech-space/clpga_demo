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


class TestAcceptance:
    def test_accept_near_prediction(self):
        """Candidate near predicted position should be accepted."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))  # vx=10
        mt.predict()  # predicted ~= (119.97, 100)
        predicted = (mt._predicted_x, mt._predicted_y)
        # Candidate right at prediction
        assert mt.accept((predicted[0], predicted[1]), ball_size=10.0) is True

    def test_reject_far_from_prediction(self):
        """Candidate far from predicted position should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))  # vx=10, speed=10
        mt.predict()
        # radius = max(1.5*10, 10*2.0) = 20; candidate at 500 px away
        assert mt.accept((500.0, 500.0), ball_size=10.0) is False

    def test_stationary_ball_uses_min_radius(self):
        """Near-zero speed should fall back to min_radius = 1.5 * ball_size."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((100.0, 100.0))  # velocity = 0
        # min_radius = 1.5 * 15 = 22.5; candidate 20px away should be accepted
        assert mt.accept((120.0, 100.0), ball_size=15.0) is True
        # candidate 25px away should be rejected
        assert mt.accept((125.0, 100.0), ball_size=15.0) is False

    def test_faster_ball_wider_radius(self):
        """Higher speed should produce a wider acceptance radius."""
        slow = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, radius_scale=4.0)
        slow.update((100.0, 100.0))
        slow.update((105.0, 100.0))  # speed=5, radius=max(20, 5*4)=20

        fast = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, radius_scale=4.0)
        fast.update((100.0, 100.0))
        fast.update((130.0, 100.0))  # speed=30, radius=max(20, 30*4)=120

        # 50px offset: slow rejects (radius=20), fast accepts (radius=120)
        assert slow.accept((155.0, 100.0), ball_size=10.0) is False
        assert fast.accept((180.0, 100.0), ball_size=10.0) is True


class TestReset:
    def test_reset_clears_velocity(self):
        """After reset, velocity should be zero."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))
        assert mt.speed > 0
        mt.reset()
        assert mt.velocity == (0.0, 0.0)
        assert mt.speed == 0.0

    def test_reset_clears_history(self):
        """After reset, next update should be like first position."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((200.0, 200.0))
        mt.reset()
        mt.update((500.0, 500.0))
        assert mt.velocity == (0.0, 0.0)


class TestMomentumTrackerProtocol:
    def test_update_returns_position(self):
        """update() should return the input position (pass-through)."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        result = mt.update((100.0, 200.0))
        assert result == (100.0, 200.0)

    def test_update_returns_each_position(self):
        """update() should return whichever position was passed in."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        r1 = mt.update((10.0, 20.0))
        r2 = mt.update((30.0, 40.0))
        assert r1 == (10.0, 20.0)
        assert r2 == (30.0, 40.0)

    def test_momentum_conforms_to_protocol(self):
        """MomentumTracker should satisfy the BallTracker protocol."""
        from clpga_demo.momentum import BallTracker
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        assert isinstance(mt, BallTracker)
