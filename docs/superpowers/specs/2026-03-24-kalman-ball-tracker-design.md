# Kalman Filter Ball Tracker — Design Spec

Add a constant-velocity Kalman filter as an alternative ball tracker alongside the existing MomentumTracker. Both trackers are kept and switchable via a CLI flag and preset parameter, allowing A/B comparison to determine which performs better in practice.

## Problem

The current MomentumTracker predicts positions during occlusion but discards those predictions — gaps are stored as NaN and linearly interpolated in Pass 2. When the ball reappears, the crop window snaps abruptly to the new detection instead of transitioning smoothly. The fixed-radius acceptance heuristic also lacks awareness of prediction uncertainty, leading to occasional wrong-ball lock-on after gaps.

## Goals

- **Smooth re-acquisition:** Eliminate crop jumps when the ball reappears after a detection gap by blending predictions with measurements via Kalman gain
- **Continuous trajectory:** Fill gaps with predicted positions instead of NaN, giving the Gaussian smoother a smooth input signal
- **Uncertainty-aware gating:** Replace the fixed-radius acceptance check with Mahalanobis distance gating that automatically widens with prediction uncertainty
- **A/B comparison:** Keep MomentumTracker intact so both approaches can be tested on the same clips

## Scope

### In scope

- New `KalmanBallTracker` class in `clpga_demo/momentum.py` alongside existing `MomentumTracker`
- Common `BallTracker` protocol in `clpga_demo/momentum.py` so pipeline code works with either tracker
- `create_tracker` factory function in `clpga_demo/momentum.py`
- Minor change to `MomentumTracker.update()`: returns the input position (was `None`) for protocol conformance
- Modified Pass 1 of `process_video` to use protocol-based tracker interface and store predicted positions during gaps — **this is a behavior change for both trackers** (previously predictions were discarded and gaps stored as NaN)
- New `--tracker` CLI flag to select `momentum` or `kalman`
- Updated presets with `tracker_type` field and Kalman-specific parameters
- New unit tests for `KalmanBallTracker` (existing MomentumTracker tests unchanged)

### Out of scope

- Changes to GaussianSmoother or Pass 2 (kept as-is for final crop smoothing)
- Changes to tracker.py (select_ball, track_video)
- Changes to cropper.py
- Real-time/streaming support
- Removing MomentumTracker

## Architecture

### Common protocol

Both trackers implement the same interface so pipeline code doesn't branch on tracker type. Lives in `clpga_demo/momentum.py`.

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

Note: The protocol uses `has_position` (True after 1 update) instead of the existing `is_tracking` (True after 2 updates). This is intentional — after a single detection, both trackers can meaningfully predict (Kalman uses zero velocity, Momentum also has zero velocity from a single position). The `is_tracking` property remains on MomentumTracker but is not part of the protocol.

### Changes to MomentumTracker

Minimal changes to conform to the protocol:

- `update()` return type changes from `None` to `tuple[float, float]` — returns the input position unchanged (pass-through, no filtering). This is a signature change; callers must be updated.
- `accept()` signature unchanged — still takes `ball_size`
- All other behavior preserved

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

**Process noise Q** — discrete white noise acceleration model with independent x and y noise:

```
G = [[0.5, 0.0],
     [0.0, 0.5],
     [1.0, 0.0],
     [0.0, 1.0]]

Q = process_noise^2 * G @ G^T
```

This produces a 4x4 block-diagonal Q where x and y acceleration noise are independent (no cross-correlation between axes). Higher `process_noise` = expects more erratic movement, trusts predictions less.

**Measurement noise R:**

```
R = measurement_noise^2 * I_2
```

Higher `measurement_noise` = trusts detections less, more smoothing.

#### Initialization

On first `update()`, state is set directly to `[x, y, 0, 0]` with initial covariance `P = diag([1, 1, 100, 100])`. Position variance of 1 px^2 reflects that the first detection is accurate. Velocity variance of 100 (px/frame)^2 reflects complete uncertainty about initial velocity — the filter learns velocity from subsequent measurements within a few frames.

#### Predict-then-correct cycle

The `update()` method internally performs **both** the predict and correct steps:

1. **Predict:** Apply state transition F to advance one frame, grow covariance by Q
2. **Correct:** Compute Kalman gain, blend prediction with measurement, shrink covariance

The `predict()` method (called during gaps) only performs step 1.

This means every frame gets exactly one application of the transition matrix F, whether the frame has a detection or not.

#### Mahalanobis distance gating

```
innovation = candidate - H @ x_predicted
S = H @ P_predicted @ H^T + R
d_squared = innovation^T @ solve(S, innovation)
accept if d_squared <= gate_threshold
```

Uses `np.linalg.solve` rather than explicit matrix inversion for numerical stability. For the 2x2 case this is unlikely to matter, but it's good practice.

The gate automatically widens as uncertainty grows during longer gaps, and tightens when the filter is confident. Default `gate_threshold = 9.0` corresponds to chi-squared with 2 DOF at ~99% confidence.

#### Interface

```python
class KalmanBallTracker:
    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 1.0,
                 gate_threshold: float = 9.0):
        ...

    def update(self, position: tuple[float, float]) -> tuple[float, float]:
        """Predict, then correct with measurement. Returns filtered position."""

    def predict(self) -> tuple[float, float]:
        """Predict-only step (no measurement). Returns predicted position."""

    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool:
        """Mahalanobis distance gating. ball_size accepted but unused (protocol compat)."""

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

#### Performance

The Kalman filter adds 4x4 matrix operations per frame (matrix multiply, solve). This is trivially cheap (~microseconds) compared to the SAM3 inference that dominates each frame.

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
    # Momentum params (needed only when tracker_type == "momentum")
    clip_duration_seconds: float = 1.0,
    fps: float = 30.0,
    momentum_history_size: int = 5,
    momentum_radius_scale: float = 4.0,
    # Kalman params (needed only when tracker_type == "kalman")
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
) -> BallTracker:
```

Returns either `MomentumTracker` or `KalmanBallTracker` based on `tracker_type`. Raises `ValueError` for unknown types. Note: `momentum_radius_scale` default is 4.0, matching the current `process_video` signature default.

### Modified Pass 1 in `process_video`

```
1. Compute clip_duration = frame_count / fps (needed for MomentumTracker)
2. Create tracker via create_tracker(tracker_type, clip_duration_seconds=clip_duration, fps=fps, ...)
3. For each frame:
   a. Get boxes from track_video
   b. If ball detected with preferred obj_id:
      - If ball was previously lost and tracker.has_position:
        -> Call accept(candidate_position, ball_bbox_size)
        -> If rejected: treat as no detection, go to (c)
      - pos = tracker.update(position) -> store pos
   c. If no valid detection:
      - If tracker.has_position: pos = tracker.predict() -> store pos
      - Else: store None
      - If lost > 3 seconds: reset selected_obj_id AND call tracker.reset()
```

Key changes from current pipeline:
- Stores `tracker.update()` return value instead of the raw detection position (filtered for Kalman, identical for Momentum)
- Stores `tracker.predict()` output during gaps instead of `None` — **behavior change for both trackers**
- Uses `has_position` instead of `is_tracking` for the re-acquisition guard (see protocol note above)
- `clip_duration` is still computed from the video and passed to `create_tracker` for MomentumTracker's use

### Pass 2 unchanged

GaussianSmoother still runs. With either tracker, it now receives a mostly-continuous trajectory (fewer NaN gaps — only before first detection or after a 3-second reset). This is an improvement for both trackers.

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
    momentum_radius_scale: float = 4.0,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
) -> None:
```

Note: `momentum_radius_scale` default stays at 4.0, matching the current signature.

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

- MomentumTracker tests: unchanged (except `update()` now returns a value — tests may need minor update to check return value)
- GaussianSmoother tests: unchanged
- Pipeline tests: updated to cover `tracker_type` parameter
- CLI tests: updated to cover `--tracker` flag

## Error handling

No new error cases. Existing behavior preserved:
- If no detections pass gating, same as no detections (predicted position stored, or None before first detection)
- 3-second timeout resets filter and obj_id selection
- `process_video` still raises `RuntimeError` if zero detections in entire video
- `create_tracker` raises `ValueError` for unknown tracker type

## Dependencies

numpy is already a project dependency. No new dependencies needed — the Kalman filter is implemented from scratch using numpy arrays.
