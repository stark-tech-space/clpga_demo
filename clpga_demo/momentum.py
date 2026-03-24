"""Momentum-based trajectory prediction for occlusion recovery."""

from __future__ import annotations

import math
from collections import deque
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BallTracker(Protocol):
    def update(self, position: tuple[float, float]) -> tuple[float, float]: ...
    def predict(self) -> tuple[float, float]: ...
    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool: ...
    def reset(self) -> None: ...
    @property
    def velocity(self) -> tuple[float, float]: ...
    @property
    def speed(self) -> float: ...
    @property
    def has_position(self) -> bool: ...


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

    def update(self, position: tuple[float, float]) -> tuple[float, float]:
        """Feed a confirmed detection. Appends to history and recomputes velocity."""
        self._history.append(position)
        self._recompute_velocity()
        self._predicted_x = position[0]
        self._predicted_y = position[1]
        return position

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


class KalmanBallTracker:
    """Constant-velocity Kalman filter for ball tracking with Mahalanobis gating."""

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
        gate_threshold: float = 9.0,
    ) -> None:
        self._gate_threshold = gate_threshold
        self._initialized = False

        # State: [x, y, vx, vy]
        self._x = np.zeros(4)
        # Covariance
        self._P = np.diag([1.0, 1.0, 100.0, 100.0])

        # Transition matrix (constant velocity, dt=1)
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix (observe position only)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise Q (independent x/y acceleration)
        G = np.array([
            [0.5, 0.0],
            [0.0, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        self._Q = (process_noise ** 2) * (G @ G.T)

        # Measurement noise R
        self._R = (measurement_noise ** 2) * np.eye(2)

    def update(self, position: tuple[float, float]) -> tuple[float, float]:
        """Predict then correct with measurement. Returns filtered position."""
        z = np.array(position)
        if not self._initialized:
            self._x[:2] = z
            self._x[2:] = 0.0
            self._initialized = True
            return (float(self._x[0]), float(self._x[1]))

        # Predict
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        # Correct
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        I = np.eye(4)
        self._P = (I - K @ self._H) @ self._P

        return (float(self._x[0]), float(self._x[1]))

    def predict(self) -> tuple[float, float]:
        """Predict-only step (no measurement). Returns predicted position."""
        if not self._initialized:
            return (0.0, 0.0)
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return (float(self._x[0]), float(self._x[1]))

    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool:
        """Mahalanobis distance gating. ball_size unused (protocol compat)."""
        if not self._initialized:
            return True
        # Hypothetical prediction (don't modify state)
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q
        z = np.array(candidate)
        innovation = z - self._H @ x_pred
        S = self._H @ P_pred @ self._H.T + self._R
        d_sq = float(innovation @ np.linalg.solve(S, innovation))
        return d_sq <= self._gate_threshold

    def reset(self) -> None:
        """Clear all state."""
        self._initialized = False
        self._x = np.zeros(4)
        self._P = np.diag([1.0, 1.0, 100.0, 100.0])

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self._x[2]), float(self._x[3]))

    @property
    def speed(self) -> float:
        return float(np.sqrt(self._x[2] ** 2 + self._x[3] ** 2))

    @property
    def has_position(self) -> bool:
        return self._initialized
