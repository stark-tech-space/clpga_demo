# Momentum Tracker Rejection Improvement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add size/shape gating and multi-frame confirmation to `MomentumTracker.accept()` to reject transient false positives during re-acquisition.

**Architecture:** Two rejection layers added inside `MomentumTracker`: (1) aspect ratio + size consistency checks using a rolling bbox size history, (2) a confirmation counter requiring N consecutive accepted frames before committing to re-acquisition. The `BallTracker` protocol, `KalmanBallTracker`, pipeline, CLI, and presets are updated to pass bbox tuples instead of scalar `ball_size`.

**Tech Stack:** Python 3.13, pytest, numpy, collections.deque

---

### Task 1: Update protocol and Kalman signatures

Update `BallTracker` protocol, `KalmanBallTracker.accept()`, and `KalmanBallTracker.update()` to accept `bbox` parameter. Update all existing tests to use new signatures.

**Files:**
- Modify: `clpga_demo/momentum.py:12-24` (protocol), `clpga_demo/momentum.py:163-184` (Kalman update), `clpga_demo/momentum.py:194-205` (Kalman accept)
- Modify: `tests/test_momentum.py` (all existing tests that call `update()` or `accept()`)

- [ ] **Step 1: Update BallTracker protocol**

Change `accept` and `update` signatures in the protocol:

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

- [ ] **Step 2: Update MomentumTracker.update() signature**

Add `bbox` parameter (ignored for now — size tracking comes in Task 2):

```python
def update(self, position: tuple[float, float], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Feed a confirmed detection. Appends to history and recomputes velocity."""
    self._history.append(position)
    self._recompute_velocity()
    self._predicted_x = position[0]
    self._predicted_y = position[1]
    return position
```

- [ ] **Step 3: Update MomentumTracker.accept() signature**

Derive `ball_size` from bbox internally:

```python
def accept(self, candidate: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    """Check if a candidate detection is within the velocity-scaled acceptance radius."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    ball_size = (w + h) / 2
    min_radius = 1.5 * ball_size
    radius = max(min_radius, self.speed * self._radius_scale)
    dx = candidate[0] - self._predicted_x
    dy = candidate[1] - self._predicted_y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    return distance <= radius
```

- [ ] **Step 4: Update KalmanBallTracker.update() signature**

Add `bbox` parameter, ignore it:

```python
def update(self, position: tuple[float, float], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
```

Body unchanged.

- [ ] **Step 5: Update KalmanBallTracker.accept() signature**

Accept `bbox`, derive `ball_size` internally (Mahalanobis gating doesn't use it, but keeps the interface consistent):

```python
def accept(self, candidate: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    """Mahalanobis distance gating."""
    if not self._initialized:
        return True
    x_pred = self._F @ self._x
    P_pred = self._F @ self._P @ self._F.T + self._Q
    z = np.array(candidate)
    innovation = z - self._H @ x_pred
    S = self._H @ P_pred @ self._H.T + self._R
    d_sq = float(innovation @ np.linalg.solve(S, innovation))
    return d_sq <= self._gate_threshold
```

- [ ] **Step 6: Update all existing tests**

Every call to `mt.update((x, y))` becomes `mt.update((x, y), (x-5, y-5, x+5, y+5))` (10px square bbox).
Every call to `mt.accept(pos, ball_size=N)` becomes `mt.accept(pos, (0, 0, N, N))` (bbox that produces the same `(w+h)/2` size).
Every call to `kt.update((x, y))` becomes `kt.update((x, y), (x-5, y-5, x+5, y+5))`.
Every call to `kt.accept(pos, ball_size=N)` becomes `kt.accept(pos, (0, 0, N, N))`.

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_momentum.py -v`
Expected: All existing tests PASS with new signatures.

- [ ] **Step 8: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "refactor: update BallTracker protocol to accept bbox tuples"
```

---

### Task 2: Add size/shape gate

Add `_ball_sizes` history, aspect ratio check, and size consistency check to `MomentumTracker`.

**Files:**
- Modify: `clpga_demo/momentum.py:26-69` (MomentumTracker __init__, update, accept)
- Test: `tests/test_momentum.py`

- [ ] **Step 1: Write failing test — reject elongated bbox**

```python
class TestShapeGate:
    def test_reject_elongated_bbox(self):
        """Candidate with aspect ratio > 2.0 should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()  # enter gap
        # Elongated bbox: 50px wide, 10px tall → ratio = 5.0
        assert mt.accept((120.0, 100.0), (95.0, 95.0, 145.0, 105.0)) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum.py::TestShapeGate::test_reject_elongated_bbox -v`
Expected: FAIL — current `accept()` doesn't check aspect ratio.

- [ ] **Step 3: Write failing test — reject oversized detection**

```python
    def test_reject_oversized_detection(self):
        """Candidate 5x larger than remembered ball size should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        # Ball is ~20px (bbox 20x20)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        # Candidate is ~100px (bbox 100x100) — 5x larger
        assert mt.accept((120.0, 100.0), (70.0, 50.0, 170.0, 150.0)) is False
```

- [ ] **Step 4: Write failing test — reject undersized detection**

```python
    def test_reject_undersized_detection(self):
        """Candidate 5x smaller than remembered ball size should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        # Ball is ~20px
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        # Candidate is ~4px (bbox 4x4)
        assert mt.accept((120.0, 100.0), (118.0, 98.0, 122.0, 102.0)) is False
```

- [ ] **Step 5: Write failing test — accept similar size**

```python
    def test_accept_similar_size(self):
        """Candidate within 2x size tolerance should pass size gate."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        # Candidate is ~30px (1.5x) — within 2x tolerance
        assert mt.accept((120.0, 100.0), (105.0, 85.0, 135.0, 115.0)) is True
```

- [ ] **Step 6: Write failing test — size gate skipped on first detection**

```python
    def test_size_gate_skipped_on_first_detection(self):
        """No size history → size gate should not reject."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        # Only 1 update, but accept is called without a gap — should pass
        # (confirm_frames gate won't apply since there's no gap yet)
        # Test that a large bbox doesn't get rejected when there's no history baseline
        # We need at least an update to have a position, then a gap, then a detection
        # But size history has only 1 entry, so the median is that entry.
        # Actually size gate applies whenever _ball_sizes has entries.
        # With 1 entry, median is that entry. A 1.5x detection should pass.
        mt.predict()
        assert mt.accept((110.0, 100.0), (95.0, 85.0, 125.0, 115.0)) is True
```

- [ ] **Step 7: Run tests to verify they all fail**

Run: `pytest tests/test_momentum.py::TestShapeGate -v`
Expected: FAIL for aspect ratio, oversized, and undersized tests.

- [ ] **Step 8: Implement size/shape gate**

In `__init__`, add:

```python
self._ball_sizes: deque[float] = deque(maxlen=history_size)
self._max_size_ratio = max_size_ratio
self._max_aspect_ratio = max_aspect_ratio
```

Add `max_size_ratio: float = 2.0` and `max_aspect_ratio: float = 2.0` to `__init__` parameters.

In `update()`, record ball size:

```python
def update(self, position: tuple[float, float], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    self._history.append(position)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    self._ball_sizes.append((w + h) / 2)
    self._recompute_velocity()
    self._predicted_x = position[0]
    self._predicted_y = position[1]
    return position
```

In `accept()`, add shape/size checks before the spatial check:

```python
def accept(self, candidate: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Aspect ratio gate
    short = min(w, h)
    if short <= 0 or max(w, h) / short > self._max_aspect_ratio:
        return False

    # Size consistency gate
    if self._ball_sizes:
        candidate_size = (w + h) / 2
        median_size = float(sorted(self._ball_sizes)[len(self._ball_sizes) // 2])
        ratio = candidate_size / median_size if median_size > 0 else 1.0
        if ratio > self._max_size_ratio or ratio < 1.0 / self._max_size_ratio:
            return False

    # Spatial gate (unchanged)
    ball_size = (w + h) / 2
    min_radius = 1.5 * ball_size
    radius = max(min_radius, self.speed * self._radius_scale)
    dx = candidate[0] - self._predicted_x
    dy = candidate[1] - self._predicted_y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    return distance <= radius
```

In `reset()`, add `self._ball_sizes.clear()`.

- [ ] **Step 9: Run all tests**

Run: `pytest tests/test_momentum.py -v`
Expected: All tests PASS including new shape gate tests.

- [ ] **Step 10: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: add size/shape gate to MomentumTracker.accept()"
```

---

### Task 3: Add multi-frame confirmation

Add confirmation counter that requires N consecutive passing detections during re-acquisition before accepting.

**Files:**
- Modify: `clpga_demo/momentum.py:26-69` (MomentumTracker __init__, update, predict, accept, reset)
- Test: `tests/test_momentum.py`

- [ ] **Step 1: Write failing test — transient false positive rejected**

```python
class TestMultiFrameConfirmation:
    def test_transient_false_positive_rejected(self):
        """1-2 frames of a passing candidate should not be accepted."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()  # frame lost → now in gap
        bbox = (105.0, 85.0, 125.0, 115.0)
        # 2 frames of a valid-looking candidate — should NOT be accepted (need 3)
        assert mt.accept((120.0, 100.0), bbox) is False
        assert mt.accept((120.0, 100.0), bbox) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum.py::TestMultiFrameConfirmation::test_transient_false_positive_rejected -v`
Expected: FAIL — second call returns True (no confirmation logic yet).

- [ ] **Step 3: Write failing test — confirmed after N frames**

```python
    def test_confirmed_after_n_frames(self):
        """N consecutive passing candidates should be accepted on Nth frame."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()  # enter gap
        bbox = (105.0, 85.0, 125.0, 115.0)
        assert mt.accept((120.0, 100.0), bbox) is False  # 1 of 3
        assert mt.accept((120.0, 100.0), bbox) is False  # 2 of 3
        assert mt.accept((120.0, 100.0), bbox) is True   # 3 of 3 → confirmed
```

- [ ] **Step 4: Write failing test — confirmation resets on failure**

```python
    def test_confirmation_resets_on_failure(self):
        """A failing candidate mid-confirmation resets the counter."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        good_bbox = (105.0, 85.0, 125.0, 115.0)
        # 2 good, then 1 far away (spatial fail), then need full 3 again
        assert mt.accept((120.0, 100.0), good_bbox) is False  # 1
        assert mt.accept((120.0, 100.0), good_bbox) is False  # 2
        assert mt.accept((500.0, 500.0), good_bbox) is False  # fail → reset
        assert mt.accept((120.0, 100.0), good_bbox) is False  # 1 again
        assert mt.accept((120.0, 100.0), good_bbox) is False  # 2
        assert mt.accept((120.0, 100.0), good_bbox) is True   # 3 → confirmed
```

- [ ] **Step 5: Write failing test — confirmation requires spatial consistency**

```python
    def test_confirmation_requires_spatial_consistency(self):
        """Consecutive candidates must be near each other, not just near prediction."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt.predict()
        bbox = (105.0, 85.0, 125.0, 115.0)
        # Two candidates that both pass spatial gate but are far from each other
        assert mt.accept((115.0, 100.0), bbox) is False  # count=1, pos=(115,100)
        assert mt.accept((125.0, 100.0), bbox) is False  # near (115,100)? depends on radius
        # Put them really far apart to ensure they fail consistency
        # First at one valid spot, second at a different valid spot far away
        mt2 = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3, radius_scale=10.0)
        mt2.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt2.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        mt2.predict()
        # With radius_scale=10, spatial gate is very wide — both pass spatial
        # But they are 60px apart from each other
        assert mt2.accept((120.0, 100.0), bbox) is False   # count=1
        assert mt2.accept((180.0, 100.0), bbox) is False   # far from (120,100) → reset to 1
        assert mt2.accept((180.0, 100.0), bbox) is False   # count=2
        assert mt2.accept((180.0, 100.0), bbox) is True    # count=3 → confirmed
```

- [ ] **Step 6: Write failing test — no confirmation when tracking continuously**

```python
    def test_no_confirmation_needed_when_tracking(self):
        """During continuous tracking (no gap), accept should return True immediately."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, confirm_frames=3)
        mt.update((100.0, 100.0), (90.0, 90.0, 110.0, 110.0))
        mt.update((110.0, 100.0), (100.0, 90.0, 120.0, 110.0))
        # No predict() call → not in a gap
        bbox = (105.0, 85.0, 125.0, 115.0)
        assert mt.accept((120.0, 100.0), bbox) is True
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `pytest tests/test_momentum.py::TestMultiFrameConfirmation -v`
Expected: FAIL — no confirmation logic exists yet.

- [ ] **Step 8: Implement multi-frame confirmation**

Add to `__init__`:

```python
self._confirm_frames = confirm_frames
self._confirm_count: int = 0
self._confirm_pos: tuple[float, float] = (0.0, 0.0)
self._in_gap: bool = False
```

Add `confirm_frames: int = 3` to `__init__` parameters.

In `update()`, add at the start:

```python
self._in_gap = False
self._confirm_count = 0
```

In `predict()`, add at the start:

```python
self._in_gap = True
```

In `accept()`, after the spatial gate passes, add confirmation logic:

```python
def accept(self, candidate: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Aspect ratio gate
    short = min(w, h)
    if short <= 0 or max(w, h) / short > self._max_aspect_ratio:
        self._confirm_count = 0
        return False

    # Size consistency gate
    if self._ball_sizes:
        candidate_size = (w + h) / 2
        median_size = float(sorted(self._ball_sizes)[len(self._ball_sizes) // 2])
        ratio = candidate_size / median_size if median_size > 0 else 1.0
        if ratio > self._max_size_ratio or ratio < 1.0 / self._max_size_ratio:
            self._confirm_count = 0
            return False

    # Spatial gate
    ball_size = (w + h) / 2
    min_radius = 1.5 * ball_size
    radius = max(min_radius, self.speed * self._radius_scale)
    dx = candidate[0] - self._predicted_x
    dy = candidate[1] - self._predicted_y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    if distance > radius:
        self._confirm_count = 0
        return False

    # No confirmation needed during continuous tracking
    if not self._in_gap:
        return True

    # Multi-frame confirmation during re-acquisition
    if self._confirm_count > 0:
        # Check consistency with previous confirmation candidate
        cdx = candidate[0] - self._confirm_pos[0]
        cdy = candidate[1] - self._confirm_pos[1]
        if math.sqrt(cdx ** 2 + cdy ** 2) > radius:
            # Inconsistent — restart confirmation from this candidate
            self._confirm_count = 1
            self._confirm_pos = candidate
            return False

    self._confirm_count += 1
    self._confirm_pos = candidate

    if self._confirm_count >= self._confirm_frames:
        self._confirm_count = 0
        return True

    return False
```

In `reset()`, add:

```python
self._confirm_count = 0
self._in_gap = False
```

- [ ] **Step 9: Run all tests**

Run: `pytest tests/test_momentum.py -v`
Expected: All tests PASS.

- [ ] **Step 10: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: add multi-frame confirmation to MomentumTracker.accept()"
```

---

### Task 4: Update create_tracker factory

Plumb new parameters through `create_tracker()`.

**Files:**
- Modify: `clpga_demo/momentum.py:226-252` (create_tracker)
- Test: `tests/test_momentum.py`

- [ ] **Step 1: Write failing test — new params passed to MomentumTracker**

```python
    def test_passes_confirmation_params(self):
        tracker = create_tracker(
            "momentum", clip_duration_seconds=5.0, fps=60.0,
            momentum_confirm_frames=5, momentum_max_size_ratio=3.0,
            momentum_max_aspect_ratio=1.5,
        )
        assert isinstance(tracker, MomentumTracker)
        assert tracker._confirm_frames == 5
        assert tracker._max_size_ratio == 3.0
        assert tracker._max_aspect_ratio == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum.py::TestCreateTracker::test_passes_confirmation_params -v`
Expected: FAIL — `create_tracker` doesn't accept these params yet.

- [ ] **Step 3: Implement — add params to create_tracker**

```python
def create_tracker(
    tracker_type: str,
    *,
    clip_duration_seconds: float = 1.0,
    fps: float = 30.0,
    momentum_history_size: int = 5,
    momentum_radius_scale: float = 4.0,
    momentum_confirm_frames: int = 3,
    momentum_max_size_ratio: float = 2.0,
    momentum_max_aspect_ratio: float = 2.0,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
) -> BallTracker:
    if tracker_type == "momentum":
        return MomentumTracker(
            clip_duration_seconds=clip_duration_seconds,
            fps=fps,
            history_size=momentum_history_size,
            radius_scale=momentum_radius_scale,
            confirm_frames=momentum_confirm_frames,
            max_size_ratio=momentum_max_size_ratio,
            max_aspect_ratio=momentum_max_aspect_ratio,
        )
    elif tracker_type == "kalman":
        return KalmanBallTracker(
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
            gate_threshold=kalman_gate_threshold,
        )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type!r}. Available: 'momentum', 'kalman'")
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_momentum.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: plumb confirmation params through create_tracker factory"
```

---

### Task 5: Update pipeline

Change `pipeline.py` to pass bbox to `accept()` and `update()`, and plumb new params through `process_video()`.

**Files:**
- Modify: `clpga_demo/pipeline.py:19-32` (process_video signature), `clpga_demo/pipeline.py:53-62` (create_tracker call), `clpga_demo/pipeline.py:76-94` (accept/update calls)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update process_video signature**

Add three new parameters after `momentum_radius_scale`:

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
    momentum_confirm_frames: int = 3,
    momentum_max_size_ratio: float = 2.0,
    momentum_max_aspect_ratio: float = 2.0,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
) -> None:
```

- [ ] **Step 2: Update create_tracker call**

Pass new params to `create_tracker()`:

```python
tracker = create_tracker(
    tracker_type,
    clip_duration_seconds=clip_duration,
    fps=fps,
    momentum_history_size=momentum_history_size,
    momentum_radius_scale=momentum_radius_scale,
    momentum_confirm_frames=momentum_confirm_frames,
    momentum_max_size_ratio=momentum_max_size_ratio,
    momentum_max_aspect_ratio=momentum_max_aspect_ratio,
    kalman_process_noise=kalman_process_noise,
    kalman_measurement_noise=kalman_measurement_noise,
    kalman_gate_threshold=kalman_gate_threshold,
)
```

- [ ] **Step 3: Update accept() call site**

Replace lines 76-79 with:

```python
if not tracker.accept((result.center_x, result.center_y), result.bbox):
```

Remove the `ball_w`, `ball_h`, `ball_size` computation (lines 76-78).

- [ ] **Step 4: Update update() call site**

Replace line 94:

```python
pos = tracker.update((result.center_x, result.center_y), result.bbox)
```

- [ ] **Step 5: Run pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: All PASS. Existing mock trackers yield 20x20 bboxes which pass all gates.

Note: The `TestMomentumFiltering::test_rejects_detection_far_from_momentum` test should still pass because the false detection at `(10, 10, 30, 30)` is far from prediction, and the multi-frame confirmation (confirm_frames=3) means even 1 frame of false positive won't be accepted. The real ball returns with obj_id=1 and consistent detections after frame 12.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add clpga_demo/pipeline.py
git commit -m "feat: update pipeline to pass bbox to tracker accept/update"
```

---

### Task 6: Update presets and CLI

Add new parameters to presets and CLI arguments.

**Files:**
- Modify: `clpga_demo/presets.py:5-23` (SHOT_PRESETS)
- Modify: `clpga_demo/__main__.py:10-28` (build_parser), `clpga_demo/__main__.py:37-46` (resolve_args), `clpga_demo/__main__.py:66-78` (main)

- [ ] **Step 1: Update presets**

Add defaults to `putt` preset:

```python
SHOT_PRESETS: dict[str, dict] = {
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
        "momentum_confirm_frames": 3,
        "momentum_max_size_ratio": 2.0,
        "momentum_max_aspect_ratio": 2.0,
        "kalman_process_noise": 0.5,
        "kalman_measurement_noise": 1.0,
        "kalman_gate_threshold": 9.0,
    },
}
```

- [ ] **Step 2: Add CLI arguments**

In `build_parser()`, add after `--momentum-radius`:

```python
parser.add_argument("--confirm-frames", type=int, default=None, help="Re-acquisition confirmation frames")
parser.add_argument("--max-size-ratio", type=float, default=None, help="Max size ratio for shape gate")
parser.add_argument("--max-aspect-ratio", type=float, default=None, help="Max aspect ratio for shape gate")
```

- [ ] **Step 3: Update resolve_args mappings**

Add to `cli_to_preset` dict:

```python
"confirm_frames": "momentum_confirm_frames",
"max_size_ratio": "momentum_max_size_ratio",
"max_aspect_ratio": "momentum_max_aspect_ratio",
```

- [ ] **Step 4: Update main() process_video call**

Add to the `process_video()` call:

```python
momentum_confirm_frames=resolved.get("momentum_confirm_frames", 3),
momentum_max_size_ratio=resolved.get("momentum_max_size_ratio", 2.0),
momentum_max_aspect_ratio=resolved.get("momentum_max_aspect_ratio", 2.0),
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add clpga_demo/presets.py clpga_demo/__main__.py
git commit -m "feat: add rejection params to presets and CLI"
```
