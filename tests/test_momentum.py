import math
import pytest
from clpga_demo.momentum import MomentumTracker, KalmanBallTracker, create_tracker


class TestVelocityEstimation:
    def test_velocity_zero_with_single_position(self):
        """Single position gives zero velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0), (95.0, 195.0, 105.0, 205.0))
        assert mt.velocity == (0.0, 0.0)
        assert mt.speed == 0.0

    def test_velocity_from_two_positions(self):
        """Two positions give exact velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0), (95.0, 195.0, 105.0, 205.0))
        mt.update((110.0, 205.0), (105.0, 200.0, 115.0, 210.0))
        assert mt.velocity == pytest.approx((10.0, 5.0))

    def test_velocity_weighted_average(self):
        """With 3 positions, most recent delta weighted higher."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, history_size=5)
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        mt.update((110.0, 100.0), (105.0, 95.0, 115.0, 105.0))  # delta: (10, 0) weight=1
        mt.update((130.0, 100.0), (125.0, 95.0, 135.0, 105.0))  # delta: (20, 0) weight=2
        # weighted avg = (10*1 + 20*2) / (1+2) = 50/3 ≈ 16.667
        vx, vy = mt.velocity
        assert vx == pytest.approx(50.0 / 3.0)
        assert vy == pytest.approx(0.0)

    def test_speed_magnitude(self):
        """Speed is the magnitude of velocity vector."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((0.0, 0.0), (-5.0, -5.0, 5.0, 5.0))
        mt.update((3.0, 4.0), (-2.0, -1.0, 8.0, 9.0))
        assert mt.speed == pytest.approx(5.0)

    def test_history_rolls_over(self):
        """Old positions drop out of the rolling buffer."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, history_size=3)
        mt.update((0.0, 0.0), (-5.0, -5.0, 5.0, 5.0))
        mt.update((10.0, 0.0), (5.0, -5.0, 15.0, 5.0))   # delta: 10
        mt.update((20.0, 0.0), (15.0, -5.0, 25.0, 5.0))   # delta: 10
        mt.update((50.0, 0.0), (45.0, -5.0, 55.0, 5.0))   # delta: 30 — oldest (0,0) dropped
        # history is now [(10,0), (20,0), (50,0)]
        # deltas: (10, 0) weight=1, (30, 0) weight=2
        # weighted avg = (10*1 + 30*2) / 3 = 70/3 ≈ 23.33
        vx, _ = mt.velocity
        assert vx == pytest.approx(70.0 / 3.0)


class TestExponentialDecay:
    def test_predict_advances_position(self):
        """predict() should move position along velocity vector."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        mt.update((110.0, 100.0), (105.0, 95.0, 115.0, 105.0))  # vx=10, vy=0
        px, py = mt.predict()
        # Position should advance beyond 110 (last known)
        assert px > 110.0
        assert py == pytest.approx(100.0, abs=0.1)

    def test_predict_decays_velocity(self):
        """Each predict() call should reduce velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        mt.update((110.0, 100.0), (105.0, 95.0, 115.0, 105.0))
        speed_before = mt.speed
        mt.predict()
        assert mt.speed < speed_before

    def test_velocity_near_zero_at_clip_end(self):
        """After clip_duration worth of frames, velocity should be ~5% of initial."""
        fps = 30.0
        duration = 10.0
        mt = MomentumTracker(clip_duration_seconds=duration, fps=fps)
        mt.update((0.0, 0.0), (-5.0, -5.0, 5.0, 5.0))
        mt.update((10.0, 0.0), (5.0, -5.0, 15.0, 5.0))  # vx=10
        initial_speed = mt.speed
        total_frames = int(duration * fps)
        for _ in range(total_frames):
            mt.predict()
        assert mt.speed == pytest.approx(initial_speed * 0.05, rel=0.01)

    def test_short_clip_duration_clamped(self):
        """Clip duration < 1.0 should be clamped to 1.0, not explode."""
        mt = MomentumTracker(clip_duration_seconds=0.1, fps=30.0)
        mt.update((0.0, 0.0), (-5.0, -5.0, 5.0, 5.0))
        mt.update((10.0, 0.0), (5.0, -5.0, 15.0, 5.0))
        # Should not raise, and velocity should not be instantly zero
        mt.predict()
        assert mt.speed > 0.0

    def test_predict_returns_position(self):
        """predict() should return the predicted (x, y) tuple."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0), (95.0, 195.0, 105.0, 205.0))
        mt.update((110.0, 205.0), (105.0, 200.0, 115.0, 210.0))
        result = mt.predict()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestAcceptance:
    def test_accept_near_prediction(self):
        """Candidate near predicted position should be accepted."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=1)
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        mt.update((110.0, 100.0), (105.0, 95.0, 115.0, 105.0))  # vx=10
        mt.predict()  # predicted ~= (119.97, 100)
        predicted = (mt._predicted_x, mt._predicted_y)
        # Candidate right at prediction
        assert mt.accept((predicted[0], predicted[1]), (0.0, 0.0, 10.0, 10.0)) is True

    def test_reject_far_from_prediction(self):
        """Candidate far from predicted position should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        mt.update((110.0, 100.0), (105.0, 95.0, 115.0, 105.0))  # vx=10, speed=10
        mt.predict()
        # radius = max(1.5*10, 10*2.0) = 20; candidate at 500 px away
        assert mt.accept((500.0, 500.0), (0.0, 0.0, 10.0, 10.0)) is False

    def test_stationary_ball_uses_min_radius(self):
        """Near-zero speed should fall back to min_radius = 1.5 * ball_size."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))  # velocity = 0
        # min_radius = 1.5 * 15 = 22.5; candidate 20px away should be accepted
        assert mt.accept((120.0, 100.0), (0.0, 0.0, 15.0, 15.0)) is True
        # candidate 25px away should be rejected
        assert mt.accept((125.0, 100.0), (0.0, 0.0, 15.0, 15.0)) is False

    def test_faster_ball_wider_radius(self):
        """Higher speed should produce a wider acceptance radius."""
        slow = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, radius_scale=4.0)
        slow.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        slow.update((105.0, 100.0), (100.0, 95.0, 110.0, 105.0))  # speed=5, radius=max(20, 5*4)=20

        fast = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, radius_scale=4.0)
        fast.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        fast.update((130.0, 100.0), (125.0, 95.0, 135.0, 105.0))  # speed=30, radius=max(20, 30*4)=120

        # 50px offset: slow rejects (radius=20), fast accepts (radius=120)
        assert slow.accept((155.0, 100.0), (0.0, 0.0, 10.0, 10.0)) is False
        assert fast.accept((180.0, 100.0), (0.0, 0.0, 10.0, 10.0)) is True


class TestReset:
    def test_reset_clears_velocity(self):
        """After reset, velocity should be zero."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        mt.update((110.0, 100.0), (105.0, 95.0, 115.0, 105.0))
        assert mt.speed > 0
        mt.reset()
        assert mt.velocity == (0.0, 0.0)
        assert mt.speed == 0.0

    def test_reset_clears_history(self):
        """After reset, next update should be like first position."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (95.0, 95.0, 105.0, 105.0))
        mt.update((200.0, 200.0), (195.0, 195.0, 205.0, 205.0))
        mt.reset()
        mt.update((500.0, 500.0), (495.0, 495.0, 505.0, 505.0))
        assert mt.velocity == (0.0, 0.0)


class TestMomentumTrackerProtocol:
    def test_update_returns_position(self):
        """update() should return the input position (pass-through)."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        result = mt.update((100.0, 200.0), (95.0, 195.0, 105.0, 205.0))
        assert result == (100.0, 200.0)

    def test_update_returns_each_position(self):
        """update() should return whichever position was passed in."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        r1 = mt.update((10.0, 20.0), (5.0, 15.0, 15.0, 25.0))
        r2 = mt.update((30.0, 40.0), (25.0, 35.0, 35.0, 45.0))
        assert r1 == (10.0, 20.0)
        assert r2 == (30.0, 40.0)

    def test_momentum_conforms_to_protocol(self):
        """MomentumTracker should satisfy the BallTracker protocol."""
        from clpga_demo.momentum import BallTracker
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        assert isinstance(mt, BallTracker)


class TestKalmanInit:
    def test_has_position_false_before_update(self):
        kt = KalmanBallTracker()
        assert kt.has_position is False

    def test_velocity_zero_before_update(self):
        kt = KalmanBallTracker()
        assert kt.velocity == (0.0, 0.0)
        assert kt.speed == 0.0

    def test_kalman_conforms_to_protocol(self):
        from clpga_demo.momentum import BallTracker
        kt = KalmanBallTracker()
        assert isinstance(kt, BallTracker)


class TestKalmanUpdate:
    def test_first_update_sets_position(self):
        kt = KalmanBallTracker()
        result = kt.update((100.0, 200.0), (95.0, 195.0, 105.0, 205.0))
        assert kt.has_position is True
        assert result[0] == pytest.approx(100.0, abs=1.0)
        assert result[1] == pytest.approx(200.0, abs=1.0)

    def test_velocity_learned_from_updates(self):
        kt = KalmanBallTracker()
        for i in range(10):
            x = 100.0 + i * 10.0
            kt.update((x, 200.0), (x - 5.0, 195.0, x + 5.0, 205.0))
        vx, vy = kt.velocity
        assert vx == pytest.approx(10.0, abs=2.0)
        assert vy == pytest.approx(0.0, abs=2.0)

    def test_update_returns_filtered_position(self):
        kt = KalmanBallTracker()
        for i in range(5):
            x = 100.0 + i * 10.0
            kt.update((x, 200.0), (x - 5.0, 195.0, x + 5.0, 205.0))
        result = kt.update((155.0, 205.0), (150.0, 200.0, 160.0, 210.0))
        assert 145.0 < result[0] < 160.0
        assert 195.0 < result[1] < 210.0


class TestKalmanPredict:
    def test_predict_continues_trajectory(self):
        kt = KalmanBallTracker()
        for i in range(5):
            x = 100.0 + i * 10.0
            kt.update((x, 200.0), (x - 5.0, 195.0, x + 5.0, 205.0))
        px, py = kt.predict()
        assert px == pytest.approx(150.0, abs=5.0)
        assert py == pytest.approx(200.0, abs=5.0)

    def test_predict_returns_tuple(self):
        kt = KalmanBallTracker()
        kt.update((100.0, 200.0), (95.0, 195.0, 105.0, 205.0))
        result = kt.predict()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestKalmanGating:
    def test_accept_near_prediction(self):
        """Candidate near predicted position should be accepted."""
        kt = KalmanBallTracker()
        for i in range(5):
            x = 100.0 + i * 10.0
            kt.update((x, 200.0), (x - 5.0, 195.0, x + 5.0, 205.0))
        assert kt.accept((150.0, 200.0), (0.0, 0.0, 10.0, 10.0)) is True

    def test_reject_far_from_prediction(self):
        """Candidate far from predicted position should be rejected."""
        kt = KalmanBallTracker()
        for i in range(5):
            x = 100.0 + i * 10.0
            kt.update((x, 200.0), (x - 5.0, 195.0, x + 5.0, 205.0))
        assert kt.accept((600.0, 600.0), (0.0, 0.0, 10.0, 10.0)) is False

    def test_gate_widens_with_uncertainty(self):
        """After more predict() calls (longer gap), gate should be wider."""
        kt_tight = KalmanBallTracker()
        for i in range(5):
            x = 100.0 + i * 10.0
            kt_tight.update((x, 200.0), (x - 5.0, 195.0, x + 5.0, 205.0))

        kt_wide = KalmanBallTracker()
        for i in range(5):
            x = 100.0 + i * 10.0
            kt_wide.update((x, 200.0), (x - 5.0, 195.0, x + 5.0, 205.0))

        # Grow uncertainty in kt_wide by predicting 30 frames
        for _ in range(30):
            kt_wide.predict()

        # Candidate 200px off from where the trajectory would be
        candidate = (350.0, 200.0)
        assert kt_tight.accept(candidate, (0.0, 0.0, 10.0, 10.0)) is False
        assert kt_wide.accept(candidate, (0.0, 0.0, 10.0, 10.0)) is True

    def test_accept_before_init_returns_true(self):
        """Before any update, accept should return True (no basis to reject)."""
        kt = KalmanBallTracker()
        assert kt.accept((100.0, 200.0), (0.0, 0.0, 10.0, 10.0)) is True


class TestKalmanReset:
    def test_reset_clears_state(self):
        kt = KalmanBallTracker()
        kt.update((100.0, 200.0), (95.0, 195.0, 105.0, 205.0))
        kt.update((110.0, 200.0), (105.0, 195.0, 115.0, 205.0))
        assert kt.has_position is True
        assert kt.speed > 0
        kt.reset()
        assert kt.has_position is False
        assert kt.velocity == (0.0, 0.0)
        assert kt.speed == 0.0

    def test_reset_allows_reinit(self):
        """After reset, next update should reinitialize."""
        kt = KalmanBallTracker()
        kt.update((100.0, 200.0), (95.0, 195.0, 105.0, 205.0))
        kt.update((110.0, 200.0), (105.0, 195.0, 115.0, 205.0))
        kt.reset()
        result = kt.update((500.0, 500.0), (495.0, 495.0, 505.0, 505.0))
        assert result[0] == pytest.approx(500.0, abs=1.0)
        assert result[1] == pytest.approx(500.0, abs=1.0)


class TestKalmanBlending:
    def test_smooth_reacquisition(self):
        """After a gap, re-detection should blend with prediction, not hard-snap."""
        kt = KalmanBallTracker()
        for i in range(10):
            x = 100.0 + i * 10.0
            kt.update((x, 200.0), (x - 5.0, 195.0, x + 5.0, 205.0))

        # Single predict step keeps covariance low enough that the Kalman
        # gain still blends prediction with measurement noticeably.
        kt.predict()

        result = kt.update((250.0, 210.0), (245.0, 205.0, 255.0, 215.0))

        assert result[0] != pytest.approx(250.0, abs=0.1)
        assert result[1] != pytest.approx(210.0, abs=0.1)
        assert 235.0 < result[0] < 255.0
        assert 195.0 < result[1] < 215.0


class TestShapeGate:
    def test_reject_elongated_bbox(self):
        """Candidate with aspect ratio > 2.0 should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()  # enter gap
        # Elongated bbox: 50px wide, 10px tall → ratio = 5.0
        assert mt.accept((120.0, 100.0), (95.0, 95.0, 145.0, 105.0)) is False

    def test_reject_oversized_detection(self):
        """Candidate 5x larger than remembered ball size should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        # Ball is ~20px (bbox 20x20)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        # Candidate is ~100px (bbox 100x100) — 5x larger
        assert mt.accept((120.0, 100.0), (70.0, 50.0, 170.0, 150.0)) is False

    def test_reject_undersized_detection(self):
        """Candidate 5x smaller than remembered ball size should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        # Ball is ~20px
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        # Candidate is ~4px (bbox 4x4)
        assert mt.accept((120.0, 100.0), (118.0, 98.0, 122.0, 102.0)) is False

    def test_accept_similar_size(self):
        """Candidate within 2x size tolerance should pass size gate."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=1)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        # Candidate is ~30px (1.5x) — within 2x tolerance
        assert mt.accept((120.0, 100.0), (105.0, 85.0, 135.0, 115.0)) is True

    def test_size_gate_skipped_on_first_detection(self):
        """No size history → size gate should not reject."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=1)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.predict()
        assert mt.accept((110.0, 100.0), (95.0, 85.0, 125.0, 115.0)) is True


class TestMultiFrameConfirmation:
    def test_transient_false_positive_rejected(self):
        """1-2 frames of a passing candidate should not be accepted."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()  # frame lost → now in gap
        bbox = (105.0, 85.0, 125.0, 115.0)
        # 2 frames of a valid-looking candidate — should NOT be accepted (need 3)
        assert mt.accept((120.0, 100.0), bbox) is False
        assert mt.accept((120.0, 100.0), bbox) is False

    def test_confirmed_after_n_frames(self):
        """N consecutive passing candidates should be accepted on Nth frame."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()  # enter gap
        bbox = (105.0, 85.0, 125.0, 115.0)
        assert mt.accept((120.0, 100.0), bbox) is False  # 1 of 3
        assert mt.accept((120.0, 100.0), bbox) is False  # 2 of 3
        assert mt.accept((120.0, 100.0), bbox) is True   # 3 of 3 → confirmed

    def test_confirmation_resets_on_failure(self):
        """A failing candidate mid-confirmation resets the counter."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        good_bbox = (105.0, 85.0, 125.0, 115.0)
        # 2 good, then 1 far away (spatial fail), then need full 3 again
        assert mt.accept((120.0, 100.0), good_bbox) is False  # 1
        assert mt.accept((120.0, 100.0), good_bbox) is False  # 2
        assert mt.accept((500.0, 500.0), good_bbox) is False  # fail → reset
        assert mt.accept((120.0, 100.0), good_bbox) is False  # 1 again
        assert mt.accept((120.0, 100.0), good_bbox) is False  # 2
        assert mt.accept((120.0, 100.0), good_bbox) is True   # 3 → confirmed

    def test_confirmation_requires_spatial_consistency(self):
        """Consecutive candidates must be near each other, not just near prediction."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        bbox = (105.0, 85.0, 125.0, 115.0)
        # Two candidates that both pass spatial gate but are far from each other
        assert mt.accept((115.0, 100.0), bbox) is False  # count=1, pos=(115,100)
        assert mt.accept((125.0, 100.0), bbox) is False  # near (115,100)? depends on radius
        # Put them really far apart to ensure they fail consistency
        # First at one valid spot, second at a different valid spot far away
        mt2 = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3, radius_scale=10.0)
        mt2.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt2.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt2.predict()
        # With radius_scale=10, speed~10, radius = max(30, ~100) ≈ 100
        # predicted pos ≈ (120, 100). Both (40,100) and (200,100) are within
        # 100px of prediction, but 160px from each other → consistency reset
        assert mt2.accept((40.0, 100.0), bbox) is False    # count=1
        assert mt2.accept((200.0, 100.0), bbox) is False   # 160px from (40,100) → reset to 1
        assert mt2.accept((200.0, 100.0), bbox) is False   # count=2
        assert mt2.accept((200.0, 100.0), bbox) is True    # count=3 → confirmed

    def test_no_confirmation_needed_when_tracking(self):
        """During continuous tracking (no gap), accept should return True immediately."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        # No predict() call → not in a gap
        bbox = (105.0, 85.0, 125.0, 115.0)
        assert mt.accept((120.0, 100.0), bbox) is True


class TestCreateTracker:
    def test_creates_momentum_tracker(self):
        tracker = create_tracker(
            "momentum", clip_duration_seconds=10.0, fps=30.0,
        )
        assert isinstance(tracker, MomentumTracker)

    def test_creates_kalman_tracker(self):
        tracker = create_tracker("kalman")
        assert isinstance(tracker, KalmanBallTracker)

    def test_raises_on_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown tracker"):
            create_tracker("unknown")

    def test_passes_momentum_params(self):
        tracker = create_tracker(
            "momentum", clip_duration_seconds=5.0, fps=60.0,
            momentum_history_size=3, momentum_radius_scale=6.0,
        )
        assert isinstance(tracker, MomentumTracker)
        assert tracker._history.maxlen == 3

    def test_passes_kalman_params(self):
        tracker = create_tracker(
            "kalman", kalman_process_noise=2.0,
            kalman_measurement_noise=3.0, kalman_gate_threshold=16.0,
        )
        assert isinstance(tracker, KalmanBallTracker)
        assert tracker._gate_threshold == 16.0
