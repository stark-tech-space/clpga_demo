"""Momentum-based trajectory prediction for occlusion recovery."""

from __future__ import annotations

import math
from collections import deque


class MomentumTracker:
    """Predicts ball trajectory during detection gaps using exponential velocity decay."""

    def __init__(
        self,
        clip_duration_seconds: float,
        fps: float,
        history_size: int = 5,
        radius_scale: float = 2.0,
    ) -> None:
        clamped_duration = max(clip_duration_seconds, 1.0)
        k = -math.log(0.05) / clamped_duration
        self._per_frame_decay = math.exp(-k / fps)
        self._radius_scale = radius_scale
        self._history: deque[tuple[float, float]] = deque(maxlen=history_size)
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._predicted_x: float = 0.0
        self._predicted_y: float = 0.0

    def update(self, position: tuple[float, float]) -> None:
        """Feed a confirmed detection. Appends to history and recomputes velocity."""
        self._history.append(position)
        self._recompute_velocity()
        self._predicted_x = position[0]
        self._predicted_y = position[1]

    def predict(self) -> tuple[float, float]:
        """Advance one frame during occlusion. Returns predicted position with decayed velocity."""
        self._vx *= self._per_frame_decay
        self._vy *= self._per_frame_decay
        self._predicted_x += self._vx
        self._predicted_y += self._vy
        return (self._predicted_x, self._predicted_y)

    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool:
        """Check if a candidate detection is within the velocity-scaled acceptance radius."""
        min_radius = 1.5 * ball_size
        radius = max(min_radius, self.speed * self._radius_scale)
        dx = candidate[0] - self._predicted_x
        dy = candidate[1] - self._predicted_y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance <= radius

    def reset(self) -> None:
        """Clear all state — position history, velocity, and predicted position."""
        self._history.clear()
        self._vx = 0.0
        self._vy = 0.0
        self._predicted_x = 0.0
        self._predicted_y = 0.0

    def _recompute_velocity(self) -> None:
        """Recompute velocity from position history using linear-weighted deltas."""
        if len(self._history) < 2:
            self._vx = 0.0
            self._vy = 0.0
            return
        deltas = []
        positions = list(self._history)
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            deltas.append((dx, dy))
        # Linear weights: [1, 2, ..., n]
        n = len(deltas)
        total_weight = n * (n + 1) / 2
        wx = sum((i + 1) * d[0] for i, d in enumerate(deltas)) / total_weight
        wy = sum((i + 1) * d[1] for i, d in enumerate(deltas)) / total_weight
        self._vx = wx
        self._vy = wy

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (vx, vy) in px/frame."""
        return (self._vx, self._vy)

    @property
    def speed(self) -> float:
        """Current speed magnitude in px/frame."""
        return math.sqrt(self._vx ** 2 + self._vy ** 2)

    @property
    def is_tracking(self) -> bool:
        """True if enough history to estimate velocity (>= 2 positions)."""
        return len(self._history) >= 2

    @property
    def has_position(self) -> bool:
        """True if at least one position has been recorded."""
        return len(self._history) >= 1
