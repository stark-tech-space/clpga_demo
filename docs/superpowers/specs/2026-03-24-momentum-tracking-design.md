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
- Updated CLI (`__main__.py`) to remove live-stream flags and add momentum parameters
- Unit tests for `MomentumTracker`

### Out of scope

- Momentum-based gap-filling for Pass 2 (Gaussian smoothing with linear interpolation is sufficient)
- Constraining SAM3's search area during occlusion (only post-detection filtering)
- Live stream support

## Architecture

### New module: `clpga_demo/momentum.py`

A stateful `MomentumTracker` class used during Pass 1 to estimate velocity, predict position during occlusion, and filter re-detections by proximity.

#### Units

All velocities are in **pixels/frame**. This is the natural unit since velocity is computed from frame-to-frame position deltas. The acceptance radius (`speed * radius_scale_factor`) is therefore in pixels, where `radius_scale_factor` represents how many frames of travel to allow as uncertainty.

#### State

- **Position history:** Rolling buffer of last `history_size` (default 5) confirmed real detection positions for velocity estimation. Not modified during occlusion — only `update()` appends to it.
- **Current velocity:** `(vx, vy)` in px/frame, computed as weighted average of recent frame-to-frame deltas
- **Per-frame decay factor:** `exp(-k / fps)` where `k = -ln(0.05) / clip_duration_seconds`, computed once at init
- **Predicted position:** Extrapolated from last known position + decaying velocity, advanced each lost frame

#### Interface

```python
class MomentumTracker:
    def __init__(self, clip_duration_seconds: float, fps: float, history_size: int = 5,
                 radius_scale: float = 4.0):
        ...

    def update(self, position: tuple[float, float]) -> None:
        """Feed a confirmed detection. Appends to position history and recomputes velocity."""

    def predict(self) -> tuple[float, float]:
        """Advance one frame during occlusion. Returns predicted position with decayed velocity."""

    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool:
        """Check if a candidate detection is within the velocity-scaled acceptance radius."""

    def reset(self) -> None:
        """Clear all state — position history, velocity, and predicted position."""

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (vx, vy) in px/frame."""

    @property
    def speed(self) -> float:
        """Current speed magnitude in px/frame."""
```

#### Velocity estimation

Velocity is computed from the rolling position history as a linearly weighted average of frame-to-frame deltas. Weights are `[1, 2, ..., n]` normalized to sum to 1, where the most recent delta gets weight `n`. With `history_size=5`, this uses up to 4 consecutive deltas.

After re-acquisition from a gap, `update()` appends the new real position to the history buffer. Velocity is recomputed from whatever real positions are in the buffer — predicted positions are never stored in the history. This means velocity after re-acquisition reflects actual ball movement, not extrapolation drift.

#### Exponential decay during occlusion

The per-frame decay factor is computed once at init:

```python
k = -math.log(0.05) / clip_duration_seconds
per_frame_decay = math.exp(-k / fps)
```

Each call to `predict()` applies one frame of decay:

```python
self._vx *= self._per_frame_decay
self._vy *= self._per_frame_decay
self._predicted_x += self._vx   # px/frame, one frame step
self._predicted_y += self._vy
```

This produces correct exponential decay because `per_frame_decay^n = exp(-k * n / fps)`.

Where `k = -ln(0.05) / clip_duration_seconds`. This models friction on the green — the ball naturally decelerates toward the hole. The clip duration is used as the tuning reference because putting clips typically end a few seconds after the ball enters the hole.

**Velocity decay over clip duration** (for a 10-second clip):

| Clip % | Time (s) | Velocity % remaining |
|--------|----------|---------------------|
| 25%    | 2.5s     | ~47%                |
| 50%    | 5.0s     | ~22%                |
| 75%    | 7.5s     | ~10%                |
| 100%   | 10.0s    | ~5%                 |

**Guard clause:** If `clip_duration_seconds < 1.0`, clamp to 1.0 to prevent extreme decay rates.

#### Velocity-scaled acceptance radius

When checking a candidate detection for re-acquisition:

```
radius = max(min_radius, speed * radius_scale_factor)
```

- `min_radius = 2 * ball_size` — ensures near-stationary balls can still be re-acquired
- `ball_size` is computed as the average of bbox width and height: `((x2 - x1) + (y2 - y1)) / 2`
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
2. Create MomentumTracker(clip_duration, fps, history_size, radius_scale)
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
      - If lost > 3 seconds: reset selected_obj_id AND call momentum.reset()
```

The momentum `predict()` is called every lost frame to advance internal state, but the predicted position is NOT stored as output — gaps remain `None`/`NaN` for the existing linear interpolation + Gaussian smoothing in Pass 2.

When `selected_obj_id` is reset after prolonged loss, the `MomentumTracker` is also reset — the old velocity history is meaningless for a different ball.

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

## CLI changes

### Updated `build_parser()`

Remove:
- `--live` flag
- `--smoothing-alpha` argument

Add:
- `--momentum-history` (int, default None) — overrides `momentum_history_size` from preset
- `--momentum-radius-scale` (float, default None) — overrides `momentum_radius_scale` from preset

### Updated `resolve_args()`

Add mappings:
- `"momentum_history"` → `"momentum_history_size"`
- `"momentum_radius_scale"` → `"momentum_radius_scale"`

### Updated `main()`

Remove the `if args.live` branch and `process_stream` import. Call `process_video` with the resolved momentum parameters:

```python
process_video(
    source=args.source,
    output=args.output,
    model=args.model,
    confidence=resolved["confidence"],
    smoothing_sigma_seconds=resolved["smoothing_sigma_seconds"],
    text=resolved["text"],
    momentum_history_size=resolved.get("momentum_history_size", 5),
    momentum_radius_scale=resolved.get("momentum_radius_scale", 4.0),
)
```

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
- **Reset:** After `reset()`, velocity is zero and history is empty
- **Short clip guard:** Clip duration < 1.0s is clamped, decay doesn't explode

Existing `GaussianSmoother` tests unchanged. `EMASmoother` tests removed alongside the class.

## Error handling

No new error cases introduced. Existing behavior preserved:
- `MomentumTracker` gracefully handles single-frame history (velocity = 0)
- If no detections pass the proximity filter, behavior is same as no detections today (NaN gap, linear interpolation)
- Clip duration clamped to minimum 1.0s to prevent extreme decay rates
