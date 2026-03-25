# Improved Momentum Tracker Rejection — Design Spec

Improve the `MomentumTracker`'s re-acquisition filtering to reject transient false positives (non-ball objects) that currently pass the spatial acceptance gate. Adds two layered defenses: a size/shape gate and multi-frame confirmation.

## Problem

During detection gaps, the momentum tracker maintains a smooth predicted position. When detection resumes, false positives (tee markers, shoe edges, hole rims) occasionally pass the spatial acceptance radius because they happen to be near the predicted position. These false positives are:
- **Transient** — they appear for only 1-2 frames
- **Differently shaped** — often larger or more elongated than a golf ball

The current `accept()` method only checks spatial distance, which is insufficient.

## Goals

- Reject transient false positive detections that are not the golf ball
- Preserve smooth re-acquisition when the real ball reappears
- Minimize added latency on genuine re-acquisition

## Approach

Two independent rejection layers added to `MomentumTracker`:

1. **Size/shape gate** — fast pre-filter using bbox dimensions
2. **Multi-frame confirmation** — temporal filter requiring N consecutive accepted detections

## Design

### Layer 1: Size/Shape Gate

#### New state

- `_ball_sizes: deque[float]` — rolling buffer (same `history_size` as position history) of recent ball sizes `(w + h) / 2`, updated only on confirmed detections via `update()`

#### Checks (applied in order within `accept()`)

1. **Aspect ratio:** `ratio = max(w, h) / min(w, h)`. Reject if `ratio > max_aspect_ratio` (default 2.0). Golf ball bboxes are roughly square.
2. **Size consistency:** Compare candidate size `(w + h) / 2` against **median** of `_ball_sizes`. Reject if candidate size is more than `max_size_ratio` times larger or less than `1 / max_size_ratio` times smaller (default `max_size_ratio = 2.0`). Skipped when `_ball_sizes` is empty (first detection always passes).
3. **Spatial distance:** Existing check — `distance(candidate, predicted_position) <= max(min_radius, speed * radius_scale)`. Unchanged.

All three must pass.

Using **median** rather than mean makes the size memory robust to a single outlier in the history.

### Layer 2: Multi-Frame Confirmation

#### New state

- `_confirm_count: int` — consecutive frames a candidate has passed all gates during re-acquisition
- `_confirm_pos: tuple[float, float]` — most recent candidate position during confirmation window
- `_confirm_frames: int` — required consecutive accepted detections (default 3, set at init)

#### Logic (within `accept()`, after size/shape + spatial gates pass)

1. If tracker is not in a re-acquisition state (no gap), return `True` immediately — no confirmation needed during continuous tracking.
2. Increment `_confirm_count`. On 2nd+ confirmation frame, also check that the new candidate is within the spatial radius of `_confirm_pos` (consistency between confirmation frames). If inconsistent, reset `_confirm_count` to 1 (current candidate starts a new confirmation sequence).
3. Store candidate position in `_confirm_pos`.
4. If `_confirm_count >= _confirm_frames`, reset `_confirm_count` to 0 and return `True`.
5. Otherwise return `False` — pipeline continues using `predict()`.

#### Reset behavior

`_confirm_count` resets to 0 when:
- Any gate check fails (size, shape, spatial)
- `reset()` is called
- A confirmed re-acquisition completes

### Interface Changes

#### `accept()` — new signature

```python
def accept(self, candidate: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
```

Replaces `ball_size: float` with `bbox: tuple[float, float, float, float]` (`x1, y1, x2, y2`). The method derives `ball_size`, aspect ratio, and candidate size internally.

**Breaking change** to the `BallTracker` protocol. Both `MomentumTracker` and `KalmanBallTracker` must be updated. The Kalman tracker accepts the parameter but ignores the bbox for gating (uses Mahalanobis distance only).

#### `update()` — new signature

```python
def update(self, position: tuple[float, float], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
```

Now also records ball size from bbox into `_ball_sizes`.

**Breaking change** to the `BallTracker` protocol.

#### New constructor parameters

```python
def __init__(
    self,
    clip_duration_seconds: float,
    fps: float,
    history_size: int = 5,
    radius_scale: float = 2.0,
    confirm_frames: int = 3,
    max_size_ratio: float = 2.0,
    max_aspect_ratio: float = 2.0,
) -> None:
```

`confirm_frames`, `max_size_ratio`, and `max_aspect_ratio` flow through `create_tracker()` and `process_video()` as keyword arguments. Presets can override them.

### Pipeline Changes

In `pipeline.py` Pass 1:

```python
# Before:
ball_size = (ball_w + ball_h) / 2
if not tracker.accept((result.center_x, result.center_y), ball_size):

# After:
if not tracker.accept((result.center_x, result.center_y), result.bbox):
```

```python
# Before:
pos = tracker.update((result.center_x, result.center_y))

# After:
pos = tracker.update((result.center_x, result.center_y), result.bbox)
```

Pipeline logic is otherwise unchanged. Multi-frame confirmation is internal to the tracker.

Pass 2 (smoothing + cropping) is unchanged.

### BallTracker Protocol Update

```python
@runtime_checkable
class BallTracker(Protocol):
    def update(self, position: tuple[float, float], bbox: tuple[float, float, float, float]) -> tuple[float, float]: ...
    def predict(self) -> tuple[float, float]: ...
    def accept(self, candidate: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool: ...
    def reset(self) -> None: ...
    @property
    def velocity(self) -> tuple[float, float]: ...
    @property
    def speed(self) -> float: ...
    @property
    def has_position(self) -> bool: ...
```

### KalmanBallTracker Changes

Signature-only changes to match protocol. Behavior unchanged:
- `update()` accepts bbox parameter, ignores it
- `accept()` accepts bbox parameter, derives ball_size for any internal use, continues using Mahalanobis gating

## Testing

### Size/shape gate tests

- `test_reject_oversized_detection` — remembered ball ~10px, re-detect with 5x larger bbox → rejected
- `test_reject_undersized_detection` — re-detect with 0.2x size → rejected
- `test_reject_elongated_bbox` — candidate with aspect ratio > 2.0 → rejected
- `test_accept_similar_size` — candidate within 2x tolerance → passes
- `test_size_gate_skipped_on_first_detection` — no size history → passes

### Multi-frame confirmation tests

- `test_transient_false_positive_rejected` — 1-2 frames of passing candidate then disappear → never accepted
- `test_confirmed_after_n_frames` — `confirm_frames` consecutive passing candidates → accepted on Nth frame
- `test_confirmation_resets_on_failure` — 2 passing, 1 failing, 2 passing → needs full N again
- `test_confirmation_requires_spatial_consistency` — two consecutive candidates far from each other → counter resets
- `test_no_confirmation_needed_when_tracking` — continuous tracking, no gap → immediate acceptance

### Protocol/integration tests

- `test_kalman_accepts_bbox_parameter` — Kalman tracker works with new signature
- `test_protocol_conformance` — both trackers satisfy updated `BallTracker` protocol

### Existing test updates

- `TestAcceptance` tests updated to pass bbox tuples instead of `ball_size` floats — spatial behavior unchanged

## Error Handling

No new error cases. Graceful degradation:
- Empty `_ball_sizes` → size gate skipped (first detection always passes)
- `confirm_frames=1` → effectively disables multi-frame confirmation
- `max_size_ratio` very large (e.g., 100.0) → effectively disables size gate
