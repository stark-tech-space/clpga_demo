import numpy as np
import pytest

from clpga_demo.smoother import GaussianSmoother


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
