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


class EMASmoother:
    """Causal exponential moving average smoother for live streams."""

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float | None, y: float | None) -> tuple[float, float]:
        """Update with a new position. Pass None for both to hold during occlusion."""
        if self._x is None or self._y is None:
            # First position — passthrough
            self._x = x if x is not None else 0.0
            self._y = y if y is not None else 0.0
            return (self._x, self._y)

        if x is None or y is None:
            # Occlusion — hold last position
            return (self._x, self._y)

        self._x = self.alpha * x + (1 - self.alpha) * self._x
        self._y = self.alpha * y + (1 - self.alpha) * self._y
        return (self._x, self._y)

    def reset(self) -> None:
        """Clear state."""
        self._x = None
        self._y = None
