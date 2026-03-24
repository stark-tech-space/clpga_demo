# Kalman Filter Ball Tracker — Design Spec

Add a constant-velocity Kalman filter as an alternative ball tracker alongside the existing MomentumTracker. Both trackers are kept and switchable via a CLI flag and preset parameter, allowing A/B comparison to determine which performs better in practice.

## Problem

The current MomentumTracker predicts positions during occlusion but discards those predictions — gaps are stored as NaN and linearly interpolated in Pass 2. When the ball reappears, the crop window snaps abruptly to the new detection instead of transitioning smoothly. The fixed-radius acceptance heuristic also lacks awareness of prediction uncertainty, leading to occasional wrong-ball lock-on after gaps.

## Goals

- **Smooth re-acquisition:** Eliminate crop jumps when the ball reappears after a detection gap by blending predictions with measurements via Kalman gain
- **Continuous trajectory:** Fill gaps with Kalman-predicted positions instead of NaN, giving the Gaussian smoother a smooth input signal
- **Uncertainty-aware gating:** Replace the fixed-radius acceptance check with Mahalanobis distance gating that automatically widens with prediction uncertainty
- **A/B comparison:** Keep MomentumTracker intact so both approaches can be tested on the same clips

## Scope

### In scope

- New `KalmanBallTracker` class in `clpga_demo/momentum.py` alongside existing `MomentumTracker`
- Common `BallTracker` protocol so pipeline code works with either tracker
- Modified Pass 1 of `process_video` to accept either tracker and store predicted positions during gaps (for Kalman) or NaN (for Momentum, preserving current behavior)
- New `--tracker` CLI flag to select `momentum` or `kalman`
- Updated presets with `tracker` field and Kalman-specific parameters
- New unit tests for `KalmanBallTracker` (existing MomentumTracker tests unchanged)

### Out of scope

- Changes to GaussianSmoother or Pass 2 (kept as-is for final crop smoothing)
- Changes to tracker.py (select_ball, track_video)
- Changes to cropper.py
- Real-time/streaming support
- Removing MomentumTracker

## Architecture

### Common protocol

Both trackers implement the same interface so pipeline code doesn't branch on tracker type:

```python
class BallTracker(Protocol):
    def update(self, position: tuple[float, float]) -> tuple[float, float]:
        """Feed a confirmed detection. Returns the (possibly filtered) position."""

    def predict(self) -> tuple[float, float]:
        """Advance one frame during occlusion. Returns predicted position."""

    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool:
        """Check if a candidate detection should be accepted."""

    def reset(self) -> None:
        """Clear all state."""

    @property
    def velocity(self) -> tuple[float, float]: ...

    @property
    def speed(self) -> float: ...

    @property
    def has_position(self) -> bool: ...
```

### Changes to MomentumTracker

Minimal changes to conform to the protocol:

- `update()` now **returns** the position it was given (pass-through, no filtering)
- `accept()` signature unchanged — still takes `ball_size`

### New class: `KalmanBallTracker`

A 2D constant-velocity Kalman filter with state vector `[x, y, vx, vy]`.

#### State model

```
State:       [x, y, vx, vy]     (position + velocity in px and px/frame)
Transition:  x' = x + vx, y' = y + vy, vx' = vx, vy' = vy
Measurement: [x, y]             (ball center from detection)
dt = 1 frame
```

#### Matrices

**Transition matrix F:**

```
F = [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
```

**Measurement matrix H:**

```
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

**Process noise Q** — discrete white noise acceleration model:

```
G = [[0.5], [0.5], [1.0], [1.0]]
Q = process_noise^2 * G @ G^T
```

Higher `process_noise` = expects more erratic movement, trusts predictions less.

**Measurement noise R:**

```
R = measurement_noise^2 * I_2
```

Higher `measurement_noise` = trusts detections less, more smoothing.

#### Initialization

On first `update()`, state is set directly to `[x, y, 0, 0]` with large initial covariance (P = diag([1, 1, 100, 100])). Velocity is learned from subsequent measurements.

#### Mahalanobis distance gating

```
innovation = candidate - H @ x_predicted
S = H @ P_predicted @ H^T + R
d_squared = innovation^T @ S^-1 @ innovation
accept if d_squared <= gate_threshold
```

The gate automatically widens as uncertainty grows during longer gaps, and tightens when the filter is confident. Default `gate_threshold = 9.0` corresponds to chi-squared with 2 DOF at ~99% confidence.

#### Interface

```python
class KalmanBallTracker:
    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 1.0,
                 gate_threshold: float = 9.0):
        ...

    def update(self, position: tuple[float, float]) -> tuple[float, float]:
        """Feed a detection. Returns the filtered (corrected) position."""

    def predict(self) -> tuple[float, float]:
        """Advance one frame with no measurement. Returns predicted position."""

    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool:
        """Mahalanobis distance gating. ball_size is accepted but unused (protocol compat)."""

    def reset(self) -> None:
        """Clear all state."""

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (vx, vy) in px/frame."""

    @property
    def speed(self) -> float:
        """Current speed magnitude in px/frame."""

    @property
    def has_position(self) -> bool:
        """True if at least one update has been called."""
```

Note: `accept()` takes `ball_size` for protocol compatibility with MomentumTracker but the Kalman implementation ignores it — it uses Mahalanobis gating from its covariance instead.

### Comparison

| Aspect | MomentumTracker | KalmanBallTracker |
|--------|----------------|-------------------|
| Velocity estimation | Linear-weighted rolling history | Kalman gain — optimal blend of prediction + measurement |
| Gap prediction | Exponential decay (physics model) | Constant-velocity with growing uncertainty |
| Re-acquisition check | Fixed radius heuristic (needs ball_size) | Mahalanobis distance gating (uncertainty-aware) |
| Output from update() | Pass-through (returns input position) | Filtered position (blend of prediction + measurement) |
| State after re-detection | Hard switch to new position | Smooth blend via Kalman gain |
| Parameters | clip_duration, fps, history_size, radius_scale | process_noise, measurement_noise, gate_threshold |

## Pipeline changes

### Tracker factory

New function in `momentum.py`:

```python
def create_tracker(
    tracker_type: str,
    *,
    # Momentum params
    clip_duration_seconds: float = 1.0,
    fps: float = 30.0,
    momentum_history_size: int = 5,
    momentum_radius_scale: float = 2.0,
    # Kalman params
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
) -> BallTracker:
```

Returns either `MomentumTracker` or `KalmanBallTracker` based on `tracker_type`.

### Modified Pass 1 in `process_video`

```
1. Create tracker via create_tracker(tracker_type, ...)
2. For each frame:
   a. Get boxes from track_video
   b. If ball detected with preferred obj_id:
      - If ball was previously lost and tracker.has_position:
        -> Call accept(candidate_position, ball_bbox_size)
        -> If rejected: treat as no detection, go to (c)
      - Call update(position) -> store returned position
   c. If no valid detection:
      - If tracker.has_position: call predict() -> store predicted position
      - Else: store None
      - If lost > 3 seconds: reset selected_obj_id AND call tracker.reset()
```

Key changes from current pipeline:
- Stores `update()` return value (filtered for Kalman, pass-through for Momentum)
- Stores `predict()` output during gaps (instead of discarding it and storing None)
- Works identically for both tracker types via the protocol

### Pass 2 unchanged

GaussianSmoother still runs. With the Kalman tracker, it receives a mostly-continuous trajectory (fewer NaN gaps). With MomentumTracker, behavior is identical to current — predictions are now stored instead of discarded, which also improves its NaN interpolation story.

**Note:** This is a behavior change for MomentumTracker too — it will now store predicted positions during gaps instead of NaN. This is an improvement for both trackers.

### Updated function signature

```python
def process_video(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_sigma_seconds: float = 0.5,
    text: list[str] | None = None,
    tracker_type: str = "momentum",
    momentum_history_size: int = 5,
    momentum_radius_scale: float = 2.0,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
) -> None:
```

## CLI changes

### Updated `build_parser()`

Keep existing:
- `--momentum-history`
- `--momentum-radius`

Add:
- `--tracker` (str, choices=["momentum", "kalman"], default None) — overrides preset
- `--kalman-process-noise` (float, default None) — overrides preset
- `--kalman-measurement-noise` (float, default None) — overrides preset
- `--kalman-gate` (float, default None) — overrides `kalman_gate_threshold`

### Updated `resolve_args()`

Add mappings:
- `"tracker"` -> `"tracker_type"`
- `"kalman_process_noise"` -> `"kalman_process_noise"`
- `"kalman_measurement_noise"` -> `"kalman_measurement_noise"`
- `"kalman_gate"` -> `"kalman_gate_threshold"`

## Preset changes

```python
SHOT_PRESETS = {
    "default": {
        "smoothing_sigma_seconds": 0.5,
        "confidence": 0.25,
        "text": ["golf ball"],
        "tracker_type": "momentum",
    },
    "putt": {
        "smoothing_sigma_seconds": 0.1,
        "confidence": 0.15,
        "text": ["golf ball on green"],
        "tracker_type": "momentum",
        "momentum_history_size": 5,
        "momentum_radius_scale": 2.0,
        "kalman_process_noise": 0.5,
        "kalman_measurement_noise": 1.0,
        "kalman_gate_threshold": 9.0,
    },
}
```

Both presets default to `"momentum"` to preserve current behavior. Users can switch with `--tracker kalman` to try the Kalman filter. The putt preset includes both sets of parameters so switching is seamless.

## Testing

### New tests for `KalmanBallTracker` in `tests/test_momentum.py`

Added alongside existing MomentumTracker tests (not replacing them):

- **Prediction accuracy:** Feed known constant-velocity positions, lose detection, verify predicted positions continue along the trajectory
- **Smooth blending on re-acquisition:** Feed positions, gap, then re-detect slightly off from prediction — verify filtered output is between prediction and measurement (not a hard snap)
- **Mahalanobis gating — accept near:** Candidate close to prediction -> `True`
- **Mahalanobis gating — reject far:** Candidate far from prediction -> `False`
- **Gate widens with uncertainty:** Same candidate distance, but longer gap -> accepted (covariance grew)
- **Velocity estimation:** After several updates, `velocity` property reflects actual movement
- **Reset:** After `reset()`, state is cleared, `has_position` is `False`
- **First update initialization:** First `update()` sets position directly, velocity starts at zero
- **No position before first update:** `has_position` is `False`

### New tests for `create_tracker` factory

- Returns `MomentumTracker` for `"momentum"`
- Returns `KalmanBallTracker` for `"kalman"`
- Raises `ValueError` for unknown tracker type

### Existing tests

- MomentumTracker tests: unchanged
- GaussianSmoother tests: unchanged
- Pipeline tests: updated to cover `tracker_type` parameter
- CLI tests: updated to cover `--tracker` flag

### MomentumTracker protocol conformance

- `update()` returns the input position (new return value)

## Error handling

No new error cases. Existing behavior preserved:
- If no detections pass gating, same as no detections (predicted position stored, or None before first detection)
- 3-second timeout resets filter and obj_id selection
- `process_video` still raises `RuntimeError` if zero detections in entire video
- `create_tracker` raises `ValueError` for unknown tracker type

## Dependencies

numpy is already a project dependency. No new dependencies needed — the Kalman filter is implemented from scratch using numpy arrays.
