import numpy as np
import pytest

from clpga_demo.smoother import EMASmoother, GaussianSmoother


class TestGaussianSmoother:
    def test_constant_positions_unchanged(self):
        """Smoothing constant positions should return the same positions."""
        smoother = GaussianSmoother(sigma=5.0)
        positions = np.array([[100.0, 200.0]] * 30)
        result = smoother.smooth(positions)
        np.testing.assert_allclose(result, positions, atol=1e-6)

    def test_reduces_jitter(self):
        """Smoothing should reduce the variance of jittery positions."""
        smoother = GaussianSmoother(sigma=5.0)
        base = np.array([[100.0, 200.0]] * 30)
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, 10, size=base.shape)
        noisy = base + jitter
        result = smoother.smooth(noisy)
        # Smoothed variance should be less than original
        assert np.var(result[:, 0]) < np.var(noisy[:, 0])
        assert np.var(result[:, 1]) < np.var(noisy[:, 1])

    def test_preserves_shape(self):
        """Output shape must match input shape."""
        smoother = GaussianSmoother(sigma=3.0)
        positions = np.array([[i * 10.0, i * 5.0] for i in range(20)])
        result = smoother.smooth(positions)
        assert result.shape == positions.shape

    def test_interpolates_missing_frames(self):
        """NaN positions should be interpolated before smoothing."""
        smoother = GaussianSmoother(sigma=3.0)
        positions = np.array([
            [100.0, 200.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [130.0, 230.0],
        ])
        result = smoother.smooth(positions)
        # No NaNs in output
        assert not np.any(np.isnan(result))
        # Interpolated middle values should be between start and end
        assert 100.0 < result[1, 0] < 130.0
        assert 100.0 < result[2, 0] < 130.0

    def test_from_fps_creates_correct_sigma(self):
        """Factory method should compute sigma from fps and seconds."""
        smoother = GaussianSmoother.from_fps(fps=30.0, sigma_seconds=0.5)
        assert smoother.sigma == 15.0

        smoother60 = GaussianSmoother.from_fps(fps=60.0, sigma_seconds=0.5)
        assert smoother60.sigma == 30.0


class TestEMASmoother:
    def test_first_position_passthrough(self):
        """First position should be returned as-is."""
        smoother = EMASmoother(alpha=0.15)
        result = smoother.update(100.0, 200.0)
        assert result == (100.0, 200.0)

    def test_smooths_toward_new_position(self):
        """After update, smoothed position should move toward new position."""
        smoother = EMASmoother(alpha=0.5)
        smoother.update(100.0, 200.0)
        x, y = smoother.update(200.0, 300.0)
        assert x == pytest.approx(150.0)
        assert y == pytest.approx(250.0)

    def test_hold_on_none(self):
        """None input (occlusion) should hold the last known position."""
        smoother = EMASmoother(alpha=0.15)
        smoother.update(100.0, 200.0)
        x, y = smoother.update(None, None)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_low_alpha_more_smooth(self):
        """Lower alpha should produce smoother (slower-moving) results."""
        slow = EMASmoother(alpha=0.1)
        fast = EMASmoother(alpha=0.9)
        slow.update(100.0, 100.0)
        fast.update(100.0, 100.0)
        sx, _ = slow.update(200.0, 100.0)
        fx, _ = fast.update(200.0, 100.0)
        # Fast alpha should move further toward 200
        assert fx > sx

    def test_reset(self):
        """Reset should clear state so next update is passthrough."""
        smoother = EMASmoother(alpha=0.5)
        smoother.update(100.0, 200.0)
        smoother.update(150.0, 250.0)
        smoother.reset()
        result = smoother.update(500.0, 600.0)
        assert result == (500.0, 600.0)
