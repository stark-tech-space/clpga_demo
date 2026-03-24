"""Trajectory smoothing for crop window positioning."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


class GaussianSmoother:
    """Bidirectional Gaussian smoothing for pre-recorded video trajectories.

    Interpolates missing (NaN) positions linearly before smoothing.
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    @classmethod
    def from_fps(cls, fps: float, sigma_seconds: float = 0.5) -> GaussianSmoother:
        """Create a smoother with sigma computed from FPS and desired time window."""
        return cls(sigma=fps * sigma_seconds)

    def smooth(self, positions: np.ndarray) -> np.ndarray:
        """Smooth an (N, 2) array of [x, y] positions. NaN entries are interpolated first."""
        result = positions.copy().astype(float)
        for col in range(2):
            series = result[:, col]
            mask = np.isnan(series)
            if mask.any() and not mask.all():
                valid_idx = np.where(~mask)[0]
                series[mask] = np.interp(np.where(mask)[0], valid_idx, series[valid_idx])
            result[:, col] = gaussian_filter1d(series, sigma=self.sigma)
        return result
