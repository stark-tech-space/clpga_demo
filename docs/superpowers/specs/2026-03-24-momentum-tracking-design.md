# Momentum-Based Tracking for Putt Occlusion Recovery — Design Spec

Enhance the golf ball tracker to predict ball trajectory during detection gaps using momentum extrapolation with exponential velocity decay. This enables smoother crop following and reliable re-acquisition of the same ball after occlusion, particularly for putting clips where the ball rolls predictably along the green.

## Goals

- **Smooth crop continuity:** Eliminate sharp crop jumps when the ball is temporarily lost by predicting where the ball should be
- **Correct re-acquisition:** When the ball reappears, verify the detection is near the predicted trajectory before accepting it — prevents locking onto a different ball or false positive
- **Natural deceleration:** Model ball friction with exponential velocity decay tuned to the clip duration, since putting clips end shortly after the ball enters the hole

## Scope

### In scope

- New `MomentumTracker` class in `clpga_demo/momentum.py`
- Modified Pass 1 of `process_video` to use momentum-based proximity filtering
- Removal of live stream pipeline (`process_stream`, `EMASmoother`, `--live` CLI flag)
- Updated presets (remove `smoothing_alpha`, add momentum parameters to `putt`)
- Unit tests for `MomentumTracker`

### Out of scope

- Momentum-based gap-filling for Pass 2 (Gaussian smoothing with linear interpolation is sufficient)
- Constraining SAM3's search area during occlusion (only post-detection filtering)
- Live stream support

## Architecture

### New module: `clpga_demo/momentum.py`

A stateful `MomentumTracker` class used during Pass 1 to estimate velocity, predict position during occlusion, and filter re-detections by proximity.

#### State

- **Position history:** Rolling buffer of last `history_size` (default 5) confirmed positions for velocity estimation
- **Current velocity:** `(vx, vy)` computed as weighted average of recent frame-to-frame deltas (more recent frames weighted higher)
- **Decay rate:** Exponential decay constant `k = -ln(0.05) / clip_duration_seconds` — velocity reaches 5% of initial by the end of the clip
- **Frames since lost:** Counter tracking how long the ball has been missing
- **Predicted position:** Extrapolated from last known position + decaying velocity, advanced each lost frame

#### Interface

```python
class MomentumTracker:
    def __init__(self, clip_duration_seconds: float, fps: float, history_size: int = 5):
        ...

    def update(self, position: tuple[float, float]) -> None:
        """Feed a confirmed detection. Updates velocity estimate."""

    def predict(self) -> tuple[float, float]:
        """Advance one frame during occlusion. Returns predicted position with decayed velocity."""

    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool:
        """Check if a candidate detection is within the velocity-scaled acceptance radius."""

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (vx, vy)."""

    @property
    def speed(self) -> float:
        """Current speed magnitude."""
```

#### Velocity estimation

Velocity is computed from the rolling position history as a weighted average of frame-to-frame deltas. More recent deltas are weighted higher to respond to trajectory changes. With `history_size=5`, this uses up to 4 consecutive deltas.

#### Exponential decay during occlusion

Each frame the ball is lost:

```
dt = frames_lost / fps
velocity *= exp(-k * dt)
predicted_position += velocity
```

Where `k = -ln(0.05) / clip_duration_seconds`. This models friction on the green — the ball naturally decelerates toward the hole. The clip duration is used as the tuning reference because putting clips typically end a few seconds after the ball enters the hole.

#### Velocity-scaled acceptance radius

When checking a candidate detection for re-acquisition:

```
radius = max(min_radius, speed * radius_scale_factor)
```

- `min_radius = 2 * ball_size` — ensures near-stationary balls can still be re-acquired
- `radius_scale_factor` — tunable multiplier (default 4.0), represents how many frames of travel to allow as uncertainty
- Candidate is accepted if `distance(candidate, predicted_position) <= radius`

## Pipeline changes

### Removals

- **`process_stream`** from `pipeline.py`
- **`EMASmoother`** from `smoother.py`
- **`--live` flag and `smoothing-alpha` CLI arg** from `__main__.py`
- **`smoothing_alpha`** from all presets in `presets.py`

### Modified Pass 1 in `process_video`

```
1. Compute clip_duration = frame_count / fps
2. Create MomentumTracker(clip_duration, fps)
3. For each frame:
   a. Get boxes from track_video
   b. If ball detected with preferred obj_id:
      - If ball was previously lost (momentum has prediction):
        → Call accept(candidate_position, ball_bbox_size)
        → If rejected: treat as no detection, go to (c)
      - Call update(position) on momentum tracker
      - Store position
   c. If no valid detection:
      - Call predict() to advance momentum state
      - Store None (gap remains NaN for Gaussian interpolation)
      - If lost > 3 seconds: reset selected_obj_id
```

The momentum `predict()` is called every lost frame to advance internal state, but the predicted position is NOT stored as output — gaps remain `None`/`NaN` for the existing linear interpolation + Gaussian smoothing in Pass 2.

### Pass 2 unchanged

Linear interpolation of NaN gaps followed by Gaussian smoothing — no modifications needed.

### Updated function signature

```python
def process_video(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_sigma_seconds: float = 0.5,
    text: list[str] | None = None,
    momentum_history_size: int = 5,
    momentum_radius_scale: float = 4.0,
) -> None:
```

Decay rate is derived automatically from clip duration — no user-facing parameter.

## Preset changes

```python
SHOT_PRESETS = {
    "default": {
        "smoothing_sigma_seconds": 0.5,
        "confidence": 0.25,
        "text": ["golf ball"],
    },
    "putt": {
        "smoothing_sigma_seconds": 0.1,
        "confidence": 0.15,
        "text": ["golf ball on green"],
        "momentum_history_size": 5,
        "momentum_radius_scale": 4.0,
    },
}
```

## Testing

Unit tests for `MomentumTracker` in `tests/test_momentum.py`:

- **Velocity estimation:** Feed positions with known velocity, verify `velocity` property matches
- **Exponential decay:** Known clip duration, call `predict()` repeatedly, verify positions follow decay curve and velocity approaches zero
- **Accept — near prediction:** Steady velocity, lose ball, candidate near predicted position → `True`
- **Accept — far from prediction:** Same setup, candidate far away → `False`
- **Velocity-scaled radius:** Fast ball has wider acceptance radius than slow ball
- **Stationary ball:** Near-zero velocity → radius falls back to `min_radius`, still accepts nearby detections
- **History buffer:** Velocity stable when buffer is full vs. partially filled (first few frames)

Existing `GaussianSmoother` tests unchanged. `EMASmoother` tests removed alongside the class.

## Error handling

No new error cases introduced. Existing behavior preserved:
- `MomentumTracker` gracefully handles single-frame history (velocity = 0)
- If no detections pass the proximity filter, behavior is same as no detections today (NaN gap, linear interpolation)
