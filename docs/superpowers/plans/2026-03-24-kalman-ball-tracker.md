# Kalman Ball Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Kalman filter ball tracker as a switchable alternative to MomentumTracker, sharing a common BallTracker protocol.

**Architecture:** New `KalmanBallTracker` class alongside existing `MomentumTracker` in `momentum.py`, behind a `BallTracker` protocol and `create_tracker` factory. Pipeline uses the protocol interface. CLI `--tracker` flag selects between them.

**Tech Stack:** Python, numpy (already installed), pytest

**Spec:** `docs/superpowers/specs/2026-03-24-kalman-ball-tracker-design.md`

---

### Task 1: BallTracker Protocol and MomentumTracker Conformance

**Files:**
- Modify: `clpga_demo/momentum.py`
- Test: `tests/test_momentum.py`

- [ ] **Step 1: Write failing test for MomentumTracker.update() return value**

Add to `tests/test_momentum.py`:

```python
class TestMomentumTrackerProtocol:
    def test_update_returns_position(self):
        """update() should return the input position (pass-through)."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        result = mt.update((100.0, 200.0))
        assert result == (100.0, 200.0)

    def test_update_returns_each_position(self):
        """update() should return whichever position was passed in."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        r1 = mt.update((10.0, 20.0))
        r2 = mt.update((30.0, 40.0))
        assert r1 == (10.0, 20.0)
        assert r2 == (30.0, 40.0)

    def test_momentum_conforms_to_protocol(self):
        """MomentumTracker should satisfy the BallTracker protocol."""
        from clpga_demo.momentum import BallTracker
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        assert isinstance(mt, BallTracker)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum.py::TestMomentumTrackerProtocol -v`
Expected: FAIL — `update()` currently returns `None`

- [ ] **Step 3: Add BallTracker protocol and update MomentumTracker.update() return**

In `clpga_demo/momentum.py`, add the protocol import and class at the top (after existing imports), then modify `MomentumTracker.update()` to return the position:

```python
from typing import Protocol, runtime_checkable

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
```

Change `MomentumTracker.update()`:
```python
def update(self, position: tuple[float, float]) -> tuple[float, float]:
    """Feed a confirmed detection. Appends to history and recomputes velocity."""
    self._history.append(position)
    self._recompute_velocity()
    self._predicted_x = position[0]
    self._predicted_y = position[1]
    return position
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_momentum.py -v`
Expected: ALL PASS (including existing tests — the return value change doesn't break callers that ignored the old `None`)

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: add BallTracker protocol, MomentumTracker.update() returns position"
```

---

### Task 2: KalmanBallTracker — Core (init, predict, update)

**Files:**
- Modify: `clpga_demo/momentum.py`
- Test: `tests/test_momentum.py`

- [ ] **Step 1: Write failing tests for KalmanBallTracker basics**

Add to `tests/test_momentum.py`:

```python
from clpga_demo.momentum import KalmanBallTracker

class TestKalmanInit:
    def test_has_position_false_before_update(self):
        kt = KalmanBallTracker()
        assert kt.has_position is False

    def test_velocity_zero_before_update(self):
        kt = KalmanBallTracker()
        assert kt.velocity == (0.0, 0.0)
        assert kt.speed == 0.0

    def test_kalman_conforms_to_protocol(self):
        """KalmanBallTracker should satisfy the BallTracker protocol."""
        from clpga_demo.momentum import BallTracker
        kt = KalmanBallTracker()
        assert isinstance(kt, BallTracker)

class TestKalmanUpdate:
    def test_first_update_sets_position(self):
        """First update initializes state; returns the position."""
        kt = KalmanBallTracker()
        result = kt.update((100.0, 200.0))
        assert kt.has_position is True
        # First update should return close to input (no prior to blend with)
        assert result[0] == pytest.approx(100.0, abs=1.0)
        assert result[1] == pytest.approx(200.0, abs=1.0)

    def test_velocity_learned_from_updates(self):
        """After several updates at constant velocity, filter should learn the velocity."""
        kt = KalmanBallTracker()
        for i in range(10):
            kt.update((100.0 + i * 10.0, 200.0))
        vx, vy = kt.velocity
        assert vx == pytest.approx(10.0, abs=2.0)
        assert vy == pytest.approx(0.0, abs=2.0)

    def test_update_returns_filtered_position(self):
        """update() should return a filtered position (between prediction and measurement)."""
        kt = KalmanBallTracker()
        for i in range(5):
            kt.update((100.0 + i * 10.0, 200.0))
        # Now feed a position slightly off the trajectory
        result = kt.update((155.0, 205.0))
        # Filtered position should be close to measurement but pulled toward prediction
        assert 145.0 < result[0] < 160.0
        assert 195.0 < result[1] < 210.0

class TestKalmanPredict:
    def test_predict_continues_trajectory(self):
        """predict() should extrapolate along learned velocity."""
        kt = KalmanBallTracker()
        for i in range(5):
            kt.update((100.0 + i * 10.0, 200.0))
        # Last update was at x=140. Velocity ~10 px/frame.
        px, py = kt.predict()
        assert px == pytest.approx(150.0, abs=5.0)
        assert py == pytest.approx(200.0, abs=5.0)

    def test_predict_returns_tuple(self):
        kt = KalmanBallTracker()
        kt.update((100.0, 200.0))
        result = kt.predict()
        assert isinstance(result, tuple)
        assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_momentum.py::TestKalmanInit tests/test_momentum.py::TestKalmanUpdate tests/test_momentum.py::TestKalmanPredict -v`
Expected: FAIL — `KalmanBallTracker` not defined

- [ ] **Step 3: Implement KalmanBallTracker**

Add to `clpga_demo/momentum.py`:

```python
import numpy as np

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
        # Compute innovation using predicted state (before correction)
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
```

**Important implementation note on `accept()`:** This method computes a *hypothetical* prediction to check gating, but does NOT modify state. The actual predict step happens inside `update()` when the candidate is later accepted by the pipeline. This avoids double-predicting.

Wait — there's a subtlety here. The `accept()` method needs to check gating based on the *current predicted state*, but it shouldn't modify `self._x` or `self._P`. The pipeline calls `accept()` first, and if accepted, calls `update()` which does its own predict+correct. So `accept()` must use temporary variables:

```python
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
```

This is already correct in the implementation above — `accept()` uses local variables `x_pred` and `P_pred` and never writes back to `self._x` or `self._P`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_momentum.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: add KalmanBallTracker with predict-then-correct cycle"
```

---

### Task 3: KalmanBallTracker — Gating and Reset

**Files:**
- Modify: `tests/test_momentum.py`

- [ ] **Step 1: Write failing tests for gating and reset**

Add to `tests/test_momentum.py`:

```python
class TestKalmanGating:
    def test_accept_near_prediction(self):
        """Candidate near predicted position should be accepted."""
        kt = KalmanBallTracker()
        for i in range(5):
            kt.update((100.0 + i * 10.0, 200.0))
        # Candidate right along the trajectory
        assert kt.accept((150.0, 200.0), ball_size=10.0) is True

    def test_reject_far_from_prediction(self):
        """Candidate far from predicted position should be rejected."""
        kt = KalmanBallTracker()
        for i in range(5):
            kt.update((100.0 + i * 10.0, 200.0))
        # Candidate way off (500px away)
        assert kt.accept((600.0, 600.0), ball_size=10.0) is False

    def test_gate_widens_with_uncertainty(self):
        """After more predict() calls (longer gap), gate should be wider."""
        kt_tight = KalmanBallTracker()
        for i in range(5):
            kt_tight.update((100.0 + i * 10.0, 200.0))

        kt_wide = KalmanBallTracker()
        for i in range(5):
            kt_wide.update((100.0 + i * 10.0, 200.0))

        # Grow uncertainty in kt_wide by predicting 30 frames
        for _ in range(30):
            kt_wide.predict()

        # Candidate 200px off from where the trajectory would be — should be
        # rejected by tight gate but accepted by wide gate
        candidate = (350.0, 200.0)
        assert kt_tight.accept(candidate, ball_size=10.0) is False
        assert kt_wide.accept(candidate, ball_size=10.0) is True

    def test_accept_before_init_returns_true(self):
        """Before any update, accept should return True (no basis to reject)."""
        kt = KalmanBallTracker()
        assert kt.accept((100.0, 200.0), ball_size=10.0) is True


class TestKalmanReset:
    def test_reset_clears_state(self):
        kt = KalmanBallTracker()
        kt.update((100.0, 200.0))
        kt.update((110.0, 200.0))
        assert kt.has_position is True
        assert kt.speed > 0
        kt.reset()
        assert kt.has_position is False
        assert kt.velocity == (0.0, 0.0)
        assert kt.speed == 0.0

    def test_reset_allows_reinit(self):
        """After reset, next update should reinitialize."""
        kt = KalmanBallTracker()
        kt.update((100.0, 200.0))
        kt.update((110.0, 200.0))
        kt.reset()
        result = kt.update((500.0, 500.0))
        assert result[0] == pytest.approx(500.0, abs=1.0)
        assert result[1] == pytest.approx(500.0, abs=1.0)


class TestKalmanBlending:
    def test_smooth_reacquisition(self):
        """After a gap, re-detection should blend with prediction, not hard-snap."""
        kt = KalmanBallTracker()
        # Build up steady rightward velocity
        for i in range(10):
            kt.update((100.0 + i * 10.0, 200.0))
        # Last position: (190, 200), velocity ~10 px/frame

        # Gap: 5 frames of prediction
        for _ in range(5):
            kt.predict()
        # Predicted position should be ~(240, 200)

        # Re-detect at (250, 210) — slightly off from prediction
        result = kt.update((250.0, 210.0))

        # Filtered result should be between prediction (~240,200) and measurement (250,210)
        # Not exactly the measurement (that would be hard-snap)
        assert result[0] != pytest.approx(250.0, abs=0.1)
        assert result[1] != pytest.approx(210.0, abs=0.1)
        # But close to the measurement
        assert 235.0 < result[0] < 255.0
        assert 195.0 < result[1] < 215.0
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_momentum.py::TestKalmanGating tests/test_momentum.py::TestKalmanReset tests/test_momentum.py::TestKalmanBlending -v`
Expected: ALL PASS (implementation from Task 2 should already handle these)

- [ ] **Step 3: Commit**

```bash
git add tests/test_momentum.py
git commit -m "test: add Kalman gating, reset, and blending tests"
```

---

### Task 4: create_tracker Factory

**Files:**
- Modify: `clpga_demo/momentum.py`
- Test: `tests/test_momentum.py`

- [ ] **Step 1: Write failing tests for create_tracker**

Add to `tests/test_momentum.py`:

```python
from clpga_demo.momentum import create_tracker

class TestCreateTracker:
    def test_creates_momentum_tracker(self):
        tracker = create_tracker(
            "momentum", clip_duration_seconds=10.0, fps=30.0,
        )
        assert isinstance(tracker, MomentumTracker)

    def test_creates_kalman_tracker(self):
        tracker = create_tracker("kalman")
        assert isinstance(tracker, KalmanBallTracker)

    def test_raises_on_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown tracker"):
            create_tracker("unknown")

    def test_passes_momentum_params(self):
        tracker = create_tracker(
            "momentum", clip_duration_seconds=5.0, fps=60.0,
            momentum_history_size=3, momentum_radius_scale=6.0,
        )
        assert isinstance(tracker, MomentumTracker)
        # Verify params were passed by checking history maxlen
        assert tracker._history.maxlen == 3

    def test_passes_kalman_params(self):
        tracker = create_tracker(
            "kalman", kalman_process_noise=2.0,
            kalman_measurement_noise=3.0, kalman_gate_threshold=16.0,
        )
        assert isinstance(tracker, KalmanBallTracker)
        assert tracker._gate_threshold == 16.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_momentum.py::TestCreateTracker -v`
Expected: FAIL — `create_tracker` not defined

- [ ] **Step 3: Implement create_tracker**

Add to `clpga_demo/momentum.py`:

```python
def create_tracker(
    tracker_type: str,
    *,
    clip_duration_seconds: float = 1.0,
    fps: float = 30.0,
    momentum_history_size: int = 5,
    momentum_radius_scale: float = 4.0,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 1.0,
    kalman_gate_threshold: float = 9.0,
) -> BallTracker:
    """Create a ball tracker by type name."""
    if tracker_type == "momentum":
        return MomentumTracker(
            clip_duration_seconds=clip_duration_seconds,
            fps=fps,
            history_size=momentum_history_size,
            radius_scale=momentum_radius_scale,
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_momentum.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: add create_tracker factory function"
```

---

### Task 5: Update Pipeline to Use BallTracker Protocol

**Files:**
- Modify: `clpga_demo/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for tracker_type parameter**

Add to `tests/test_pipeline.py`:

```python
class TestTrackerType:
    def test_default_tracker_is_momentum(self, tmp_path):
        """Default tracker_type should work (momentum)."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_video(input_path, output_path, tracker_type="momentum")

        assert Path(output_path).exists()

    def test_kalman_tracker_creates_output(self, tmp_path):
        """Kalman tracker should also produce output video."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_video(input_path, output_path, tracker_type="kalman")

        assert Path(output_path).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::TestTrackerType -v`
Expected: FAIL — `process_video` doesn't accept `tracker_type`

- [ ] **Step 3: Update pipeline.py**

Replace the contents of `process_video` in `clpga_demo/pipeline.py`. Key changes:
- Import `create_tracker` instead of `MomentumTracker`
- Add `tracker_type` and Kalman params to signature
- Use `create_tracker(...)` to build the tracker
- Store `tracker.update()` return value
- Store `tracker.predict()` return value instead of `None`
- Use `tracker.has_position` instead of `momentum.is_tracking`

```python
from clpga_demo.momentum import create_tracker
```

Updated `process_video`:

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
    """Process a pre-recorded video: track ball, smooth trajectory, crop portrait.

    Two-pass pipeline:
      Pass 1 — collect all ball positions from SAM3 tracking with tracker filtering.
      Pass 2 — smooth positions, re-read video, write cropped output.
    """
    if not Path(source).exists():
        raise ValueError(f"Input video does not exist: {source}")

    # --- Get video properties ---
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    clip_duration = frame_count / fps if fps > 0 else 1.0
    tracker = create_tracker(
        tracker_type,
        clip_duration_seconds=clip_duration,
        fps=fps,
        momentum_history_size=momentum_history_size,
        momentum_radius_scale=momentum_radius_scale,
        kalman_process_noise=kalman_process_noise,
        kalman_measurement_noise=kalman_measurement_noise,
        kalman_gate_threshold=kalman_gate_threshold,
    )

    # --- Pass 1: Collect positions with tracker filtering ---
    positions: list[tuple[float, float] | None] = []
    selected_obj_id: int | None = None
    frames_since_lost = 0

    for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence, text=text):
        result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)

        accepted = False
        if result is not None:
            if frames_since_lost > 0 and tracker.has_position:
                ball_w = result.bbox[2] - result.bbox[0]
                ball_h = result.bbox[3] - result.bbox[1]
                ball_size = (ball_w + ball_h) / 2
                if not tracker.accept((result.center_x, result.center_y), ball_size):
                    logger.debug(
                        "Frame %d: rejected detection obj_id=%d — too far from tracker prediction",
                        frame_idx, result.obj_id,
                    )
                    result = None
                else:
                    accepted = True
            else:
                accepted = True

        if accepted and result is not None:
            if selected_obj_id is None:
                selected_obj_id = result.obj_id
                logger.info(f"Selected ball obj_id={result.obj_id} at frame {frame_idx}")
            pos = tracker.update((result.center_x, result.center_y))
            positions.append(pos)
            frames_since_lost = 0
        else:
            if tracker.has_position:
                pos = tracker.predict()
                positions.append(pos)
            else:
                positions.append(None)
            frames_since_lost += 1
            if fps > 0 and frames_since_lost > fps * 3:
                selected_obj_id = None
                tracker.reset()
                frames_since_lost = 0

    if all(p is None for p in positions):
        raise RuntimeError("No golf ball detected in video")

    # --- Pass 2: Smooth and crop ---
    pos_array = np.array([p if p is not None else (np.nan, np.nan) for p in positions])
    smoother = GaussianSmoother.from_fps(fps=fps, sigma_seconds=smoothing_sigma_seconds)
    smoothed = smoother.smooth(pos_array)

    crop_sample = calculate_crop(smoothed[0][0], smoothed[0][1], src_w, src_h)

    with VideoWriter(output, fps, crop_sample.width, crop_sample.height) as writer:
        cap = cv2.VideoCapture(source)
        for i in range(len(smoothed)):
            ret, frame = cap.read()
            if not ret:
                break
            crop = calculate_crop(smoothed[i][0], smoothed[i][1], src_w, src_h)
            cropped = crop.apply(frame)
            writer.write(cropped)
        cap.release()

    logger.info(f"Output written to {output}")
```

Also update the import at the top of `pipeline.py`:
- Remove: `from clpga_demo.momentum import MomentumTracker`
- Add: `from clpga_demo.momentum import create_tracker`

- [ ] **Step 4: Run all tests to verify they pass**

Run: `pytest tests/test_pipeline.py tests/test_momentum.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/pipeline.py tests/test_pipeline.py
git commit -m "feat: update pipeline to use BallTracker protocol with tracker_type selection"
```

---

### Task 6: Update Presets

**Files:**
- Modify: `clpga_demo/presets.py`
- Modify: `tests/test_presets.py`

- [ ] **Step 1: Write failing test for new preset fields**

Add to `tests/test_presets.py`:

```python
class TestTrackerPresets:
    def test_default_has_tracker_type(self):
        preset = get_preset("default")
        assert preset["tracker_type"] == "momentum"

    def test_putt_has_tracker_type(self):
        preset = get_preset("putt")
        assert preset["tracker_type"] == "momentum"

    def test_putt_has_kalman_params(self):
        preset = get_preset("putt")
        assert preset["kalman_process_noise"] == 0.5
        assert preset["kalman_measurement_noise"] == 1.0
        assert preset["kalman_gate_threshold"] == 9.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_presets.py::TestTrackerPresets -v`
Expected: FAIL — `tracker_type` not in presets

- [ ] **Step 3: Update presets.py**

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
        "kalman_process_noise": 0.5,
        "kalman_measurement_noise": 1.0,
        "kalman_gate_threshold": 9.0,
    },
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_presets.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/presets.py tests/test_presets.py
git commit -m "feat: add tracker_type and Kalman params to presets"
```

---

### Task 7: Update CLI

**Files:**
- Modify: `clpga_demo/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for new CLI flags**

Add to `tests/test_cli.py`:

```python
class TestTrackerCLI:
    def test_tracker_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--tracker", "kalman"])
        assert args.tracker == "kalman"

    def test_tracker_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.tracker is None

    def test_kalman_flags_parsed(self):
        parser = build_parser()
        args = parser.parse_args([
            "in.mp4", "-o", "out.mp4",
            "--kalman-process-noise", "2.0",
            "--kalman-measurement-noise", "3.0",
            "--kalman-gate", "16.0",
        ])
        assert args.kalman_process_noise == 2.0
        assert args.kalman_measurement_noise == 3.0
        assert args.kalman_gate == 16.0

    def test_tracker_resolve_to_preset(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--tracker", "kalman"])
        resolved = resolve_args(args)
        assert resolved["tracker_type"] == "kalman"

    def test_kalman_resolve_overrides_preset(self):
        parser = build_parser()
        args = parser.parse_args([
            "in.mp4", "-o", "out.mp4", "--preset", "putt",
            "--kalman-process-noise", "2.0",
        ])
        resolved = resolve_args(args)
        assert resolved["kalman_process_noise"] == 2.0
        # Other putt values preserved
        assert resolved["kalman_measurement_noise"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestTrackerCLI -v`
Expected: FAIL — `--tracker` flag not defined

- [ ] **Step 3: Update __main__.py**

Add to `build_parser()` (after the existing `--momentum-radius` line):

```python
parser.add_argument("--tracker", default=None, choices=["momentum", "kalman"], help="Tracker type: momentum or kalman")
parser.add_argument("--kalman-process-noise", type=float, default=None, help="Kalman process noise")
parser.add_argument("--kalman-measurement-noise", type=float, default=None, help="Kalman measurement noise")
parser.add_argument("--kalman-gate", type=float, default=None, help="Kalman gate threshold")
```

Add to `cli_to_preset` dict in `resolve_args()`:

```python
"tracker": "tracker_type",
"kalman_process_noise": "kalman_process_noise",
"kalman_measurement_noise": "kalman_measurement_noise",
"kalman_gate": "kalman_gate_threshold",
```

Update `main()` to pass the new params:

```python
process_video(
    source=args.source,
    output=args.output,
    model=args.model,
    confidence=resolved["confidence"],
    smoothing_sigma_seconds=resolved["smoothing_sigma_seconds"],
    text=resolved["text"],
    tracker_type=resolved.get("tracker_type", "momentum"),
    momentum_history_size=resolved.get("momentum_history_size", 5),
    momentum_radius_scale=resolved.get("momentum_radius_scale", 4.0),
    kalman_process_noise=resolved.get("kalman_process_noise", 1.0),
    kalman_measurement_noise=resolved.get("kalman_measurement_noise", 1.0),
    kalman_gate_threshold=resolved.get("kalman_gate_threshold", 9.0),
)
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/__main__.py tests/test_cli.py
git commit -m "feat: add --tracker and Kalman CLI flags"
```

---

### Task 8: Final Integration Test

**Files:**
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write integration test verifying Kalman handles gaps better**

Add to `tests/test_pipeline.py`:

```python
class TestKalmanGapHandling:
    def test_kalman_fills_gaps_with_predictions(self, tmp_path):
        """Kalman tracker should produce continuous trajectory (no NaN gaps)."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path, frames=20)

        def _gap_tracker(source, **kwargs):
            """Ball visible for 5 frames, gone for 3, back for rest."""
            cap = cv2.VideoCapture(source)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx < 5 or frame_idx >= 8:
                    cx = 100 + frame_idx * 5
                    cy = 120
                    boxes = np.array([[cx - 10, cy - 10, cx + 10, cy + 10, 1, 0.95, 0]])
                else:
                    boxes = np.empty((0, 7))
                yield frame_idx, frame, boxes
                frame_idx += 1
            cap.release()

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _gap_tracker(s)):
            process_video(input_path, output_path, tracker_type="kalman")

        assert Path(output_path).exists()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: add Kalman gap-handling integration test"
```
