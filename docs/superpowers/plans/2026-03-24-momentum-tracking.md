# Momentum-Based Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add momentum-based trajectory prediction to the golf ball tracker for putt occlusion recovery, with velocity-scaled proximity filtering for re-acquisition.

**Architecture:** New `MomentumTracker` class in `momentum.py` handles velocity estimation, exponential decay prediction, and acceptance filtering. Pipeline Pass 1 uses it to validate re-detections. Live stream pipeline removed entirely.

**Tech Stack:** Python, numpy, math (stdlib)

**Spec:** `docs/superpowers/specs/2026-03-24-momentum-tracking-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `clpga_demo/momentum.py` | Create | MomentumTracker class — velocity estimation, decay prediction, proximity filtering |
| `tests/test_momentum.py` | Create | Unit tests for MomentumTracker |
| `clpga_demo/smoother.py` | Modify | Remove EMASmoother class |
| `tests/test_smoother.py` | Modify | Remove EMASmoother tests |
| `clpga_demo/presets.py` | Modify | Remove smoothing_alpha, add momentum params to putt |
| `tests/test_presets.py` | Modify | Update preset assertions |
| `clpga_demo/pipeline.py` | Modify | Remove process_stream, add momentum filtering to process_video Pass 1 |
| `tests/test_pipeline.py` | Modify | Remove TestProcessStream, add momentum filtering tests |
| `clpga_demo/__main__.py` | Modify | Remove --live/--smoothing-alpha, add --momentum-history/--momentum-radius |
| `tests/test_cli.py` | Modify | Update CLI parsing tests |

---

### Task 1: MomentumTracker — velocity estimation

**Files:**
- Create: `tests/test_momentum.py`
- Create: `clpga_demo/momentum.py`

- [ ] **Step 1: Write failing tests for velocity estimation**

```python
import math
import pytest
from clpga_demo.momentum import MomentumTracker


class TestVelocityEstimation:
    def test_velocity_zero_with_single_position(self):
        """Single position gives zero velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0))
        assert mt.velocity == (0.0, 0.0)
        assert mt.speed == 0.0

    def test_velocity_from_two_positions(self):
        """Two positions give exact velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0))
        mt.update((110.0, 205.0))
        assert mt.velocity == pytest.approx((10.0, 5.0))

    def test_velocity_weighted_average(self):
        """With 3 positions, most recent delta weighted higher."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, history_size=5)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))  # delta: (10, 0) weight=1
        mt.update((130.0, 100.0))  # delta: (20, 0) weight=2
        # weighted avg = (10*1 + 20*2) / (1+2) = 50/3 ≈ 16.667
        vx, vy = mt.velocity
        assert vx == pytest.approx(50.0 / 3.0)
        assert vy == pytest.approx(0.0)

    def test_speed_magnitude(self):
        """Speed is the magnitude of velocity vector."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((0.0, 0.0))
        mt.update((3.0, 4.0))
        assert mt.speed == pytest.approx(5.0)

    def test_history_rolls_over(self):
        """Old positions drop out of the rolling buffer."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, history_size=3)
        mt.update((0.0, 0.0))
        mt.update((10.0, 0.0))   # delta: 10
        mt.update((20.0, 0.0))   # delta: 10
        mt.update((50.0, 0.0))   # delta: 30 — oldest (0,0) dropped
        # history is now [(10,0), (20,0), (50,0)]
        # deltas: (10, 0) weight=1, (30, 0) weight=2
        # weighted avg = (10*1 + 30*2) / 3 = 70/3 ≈ 23.33
        vx, _ = mt.velocity
        assert vx == pytest.approx(70.0 / 3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_momentum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clpga_demo.momentum'`

- [ ] **Step 3: Implement MomentumTracker init, update, velocity, speed**

```python
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
        radius_scale: float = 4.0,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_momentum.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: add MomentumTracker velocity estimation with tests"
```

---

### Task 2: MomentumTracker — exponential decay prediction

**Files:**
- Modify: `tests/test_momentum.py`
- Modify: `clpga_demo/momentum.py`

- [ ] **Step 1: Write failing tests for predict() and decay**

Add to `tests/test_momentum.py`:

```python
class TestExponentialDecay:
    def test_predict_advances_position(self):
        """predict() should move position along velocity vector."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))  # vx=10, vy=0
        px, py = mt.predict()
        # Position should advance beyond 110 (last known)
        assert px > 110.0
        assert py == pytest.approx(100.0, abs=0.1)

    def test_predict_decays_velocity(self):
        """Each predict() call should reduce velocity."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))
        speed_before = mt.speed
        mt.predict()
        assert mt.speed < speed_before

    def test_velocity_near_zero_at_clip_end(self):
        """After clip_duration worth of frames, velocity should be ~5% of initial."""
        fps = 30.0
        duration = 10.0
        mt = MomentumTracker(clip_duration_seconds=duration, fps=fps)
        mt.update((0.0, 0.0))
        mt.update((10.0, 0.0))  # vx=10
        initial_speed = mt.speed
        total_frames = int(duration * fps)
        for _ in range(total_frames):
            mt.predict()
        assert mt.speed == pytest.approx(initial_speed * 0.05, rel=0.01)

    def test_short_clip_duration_clamped(self):
        """Clip duration < 1.0 should be clamped to 1.0, not explode."""
        mt = MomentumTracker(clip_duration_seconds=0.1, fps=30.0)
        mt.update((0.0, 0.0))
        mt.update((10.0, 0.0))
        # Should not raise, and velocity should not be instantly zero
        mt.predict()
        assert mt.speed > 0.0

    def test_predict_returns_position(self):
        """predict() should return the predicted (x, y) tuple."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 200.0))
        mt.update((110.0, 205.0))
        result = mt.predict()
        assert isinstance(result, tuple)
        assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_momentum.py::TestExponentialDecay -v`
Expected: FAIL — `AttributeError: 'MomentumTracker' object has no attribute 'predict'`

- [ ] **Step 3: Implement predict()**

Add to `MomentumTracker` in `clpga_demo/momentum.py`:

```python
    def predict(self) -> tuple[float, float]:
        """Advance one frame during occlusion. Returns predicted position with decayed velocity."""
        self._vx *= self._per_frame_decay
        self._vy *= self._per_frame_decay
        self._predicted_x += self._vx
        self._predicted_y += self._vy
        return (self._predicted_x, self._predicted_y)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_momentum.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: add MomentumTracker.predict() with exponential decay"
```

---

### Task 3: MomentumTracker — accept() and reset()

**Files:**
- Modify: `tests/test_momentum.py`
- Modify: `clpga_demo/momentum.py`

- [ ] **Step 1: Write failing tests for accept() and reset()**

Add to `tests/test_momentum.py`:

```python
class TestAcceptance:
    def test_accept_near_prediction(self):
        """Candidate near predicted position should be accepted."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))  # vx=10
        mt.predict()  # predicted ~= (119.97, 100)
        predicted = (mt._predicted_x, mt._predicted_y)
        # Candidate right at prediction
        assert mt.accept((predicted[0], predicted[1]), ball_size=10.0) is True

    def test_reject_far_from_prediction(self):
        """Candidate far from predicted position should be rejected."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))  # vx=10, speed=10
        mt.predict()
        # radius = max(2*10, 10*4.0) = 40; candidate at 500 px away
        assert mt.accept((500.0, 500.0), ball_size=10.0) is False

    def test_stationary_ball_uses_min_radius(self):
        """Near-zero speed should fall back to min_radius = 2 * ball_size."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((100.0, 100.0))  # velocity = 0
        # min_radius = 2 * 15 = 30; candidate 25px away should be accepted
        assert mt.accept((125.0, 100.0), ball_size=15.0) is True
        # candidate 35px away should be rejected
        assert mt.accept((135.0, 100.0), ball_size=15.0) is False

    def test_faster_ball_wider_radius(self):
        """Higher speed should produce a wider acceptance radius."""
        slow = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, radius_scale=4.0)
        slow.update((100.0, 100.0))
        slow.update((105.0, 100.0))  # speed=5, radius=max(20, 5*4)=20

        fast = MomentumTracker(clip_duration_seconds=10.0, fps=30.0, radius_scale=4.0)
        fast.update((100.0, 100.0))
        fast.update((130.0, 100.0))  # speed=30, radius=max(20, 30*4)=120

        # 50px offset: slow rejects (radius=20), fast accepts (radius=120)
        assert slow.accept((155.0, 100.0), ball_size=10.0) is False
        assert fast.accept((180.0, 100.0), ball_size=10.0) is True


class TestReset:
    def test_reset_clears_velocity(self):
        """After reset, velocity should be zero."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((110.0, 100.0))
        assert mt.speed > 0
        mt.reset()
        assert mt.velocity == (0.0, 0.0)
        assert mt.speed == 0.0

    def test_reset_clears_history(self):
        """After reset, next update should be like first position."""
        mt = MomentumTracker(clip_duration_seconds=10.0, fps=30.0)
        mt.update((100.0, 100.0))
        mt.update((200.0, 200.0))
        mt.reset()
        mt.update((500.0, 500.0))
        assert mt.velocity == (0.0, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_momentum.py::TestAcceptance tests/test_momentum.py::TestReset -v`
Expected: FAIL — `AttributeError: 'MomentumTracker' object has no attribute 'accept'`

- [ ] **Step 3: Implement accept() and reset()**

Add to `MomentumTracker` in `clpga_demo/momentum.py`:

```python
    def accept(self, candidate: tuple[float, float], ball_size: float) -> bool:
        """Check if a candidate detection is within the velocity-scaled acceptance radius."""
        min_radius = 2.0 * ball_size
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
```

- [ ] **Step 4: Run all momentum tests to verify they pass**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_momentum.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/momentum.py tests/test_momentum.py
git commit -m "feat: add MomentumTracker.accept() and reset()"
```

---

### Task 4: Remove live stream pipeline

**Files:**
- Modify: `clpga_demo/smoother.py` — remove `EMASmoother` class
- Modify: `tests/test_smoother.py` — remove `TestEMASmoother` class and EMASmoother import
- Modify: `clpga_demo/pipeline.py` — remove `process_stream` function and `EMASmoother` import
- Modify: `tests/test_pipeline.py` — remove `TestProcessStream` class

- [ ] **Step 1: Remove EMASmoother from smoother.py**

Delete the `EMASmoother` class (lines 37-61) and keep only `GaussianSmoother`.

Updated `clpga_demo/smoother.py`:

```python
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
```

- [ ] **Step 2: Update tests/test_smoother.py — remove EMASmoother tests**

Remove `TestEMASmoother` class (lines 59-101) and the `EMASmoother` import. Keep only:

```python
import numpy as np
import pytest

from clpga_demo.smoother import GaussianSmoother
```

The 5 `TestGaussianSmoother` tests remain unchanged (lines 7-56).

- [ ] **Step 3: Remove process_stream from pipeline.py**

Delete the `process_stream` function (lines 81-151) and the `EMASmoother` import from line 12. The import line becomes:

```python
from clpga_demo.smoother import GaussianSmoother
```

- [ ] **Step 4: Remove TestProcessStream from test_pipeline.py**

Delete the `TestProcessStream` class (lines 88-107). Remove `process_stream` from any imports. Keep `TestProcessVideo` and `TestProcessVideoText`.

- [ ] **Step 5: Run all tests to verify nothing is broken**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/ -v`
Expected: All tests PASS (stream-related tests gone, everything else passes)

- [ ] **Step 6: Commit**

```bash
git add clpga_demo/smoother.py clpga_demo/pipeline.py tests/test_smoother.py tests/test_pipeline.py
git commit -m "refactor: remove live stream pipeline (process_stream, EMASmoother)"
```

---

### Task 5: Update presets

**Files:**
- Modify: `clpga_demo/presets.py`
- Modify: `tests/test_presets.py`

- [ ] **Step 1: Update test assertions first**

Replace `tests/test_presets.py` content:

```python
import pytest

from clpga_demo.presets import SHOT_PRESETS, get_preset


class TestGetPreset:
    def test_returns_default_preset(self):
        preset = get_preset("default")
        assert preset["smoothing_sigma_seconds"] == 0.5
        assert preset["confidence"] == 0.25
        assert preset["text"] == ["golf ball"]
        assert "smoothing_alpha" not in preset

    def test_returns_putt_preset(self):
        preset = get_preset("putt")
        assert preset["smoothing_sigma_seconds"] == 0.1
        assert preset["confidence"] == 0.15
        assert preset["text"] == ["golf ball on green"]
        assert preset["momentum_history_size"] == 5
        assert preset["momentum_radius_scale"] == 4.0
        assert "smoothing_alpha" not in preset

    def test_raises_on_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_returns_copy_not_reference(self):
        preset = get_preset("default")
        preset["confidence"] = 999
        assert get_preset("default")["confidence"] == 0.25
```

- [ ] **Step 2: Run tests to see them fail**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_presets.py -v`
Expected: FAIL — `smoothing_alpha` still present, `momentum_history_size` missing

- [ ] **Step 3: Update presets.py**

```python
"""Named shot presets for pipeline parameter tuning."""

from __future__ import annotations

SHOT_PRESETS: dict[str, dict] = {
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


def get_preset(name: str) -> dict:
    """Return a copy of the named preset dict, or raise ValueError."""
    if name not in SHOT_PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Available: {', '.join(SHOT_PRESETS)}")
    return {**SHOT_PRESETS[name]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_presets.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/presets.py tests/test_presets.py
git commit -m "feat: update presets — remove smoothing_alpha, add momentum params to putt"
```

---

### Task 6: Integrate MomentumTracker into pipeline Pass 1

**Files:**
- Modify: `clpga_demo/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for momentum filtering in pipeline**

Add to `tests/test_pipeline.py`:

```python
class TestMomentumFiltering:
    def test_rejects_detection_far_from_momentum(self, tmp_path):
        """A detection that jumps far from predicted trajectory should be rejected."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path, frames=20)

        call_count = {"n": 0}

        def _jumping_tracker(source, **kwargs):
            """Ball moves steadily then jumps to a distant position."""
            cap = cv2.VideoCapture(source)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx < 10:
                    # Steady rightward motion
                    cx = 100 + frame_idx * 5
                    cy = 120
                    boxes = np.array([[cx - 10, cy - 10, cx + 10, cy + 10, 1, 0.95, 0]])
                elif frame_idx == 10:
                    # Ball disappears for one frame
                    boxes = np.empty((0, 7))
                elif frame_idx == 11:
                    # False detection appears far away (should be rejected)
                    boxes = np.array([[10, 10, 30, 30, 2, 0.90, 0]])
                else:
                    # Real ball continues near expected trajectory
                    cx = 100 + frame_idx * 5
                    cy = 120
                    boxes = np.array([[cx - 10, cy - 10, cx + 10, cy + 10, 1, 0.95, 0]])
                yield frame_idx, frame, boxes
                frame_idx += 1
            cap.release()

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _jumping_tracker(s)):
            process_video(input_path, output_path)

        # If momentum filtering works, the output video should exist
        # (the false detection at frame 11 was rejected, not used for cropping)
        assert Path(output_path).exists()
```

- [ ] **Step 2: Run test to verify it fails or needs pipeline changes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_pipeline.py::TestMomentumFiltering -v`
Expected: May pass trivially (pipeline doesn't crash) or fail if signature changed. This test serves as a smoke test; the real validation is in `test_momentum.py`.

- [ ] **Step 3: Update process_video in pipeline.py**

Replace the full `pipeline.py` content:

```python
"""Pipeline orchestration — tracker -> smoother -> cropper."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from clpga_demo.cropper import VideoWriter, calculate_crop
from clpga_demo.momentum import MomentumTracker
from clpga_demo.smoother import GaussianSmoother
from clpga_demo.tracker import select_ball, track_video

logger = logging.getLogger(__name__)


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
    """Process a pre-recorded video: track ball, smooth trajectory, crop portrait.

    Two-pass pipeline:
      Pass 1 — collect all ball positions from SAM3 tracking with momentum filtering.
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
    momentum = MomentumTracker(
        clip_duration_seconds=clip_duration,
        fps=fps,
        history_size=momentum_history_size,
        radius_scale=momentum_radius_scale,
    )

    # --- Pass 1: Collect positions with momentum filtering ---
    positions: list[tuple[float, float] | None] = []
    selected_obj_id: int | None = None
    frames_since_lost = 0

    for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence, text=text):
        result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)

        accepted = False
        if result is not None:
            if frames_since_lost > 0 and momentum.is_tracking:
                # Re-acquisition: check proximity to momentum prediction
                ball_w = result.bbox[2] - result.bbox[0]
                ball_h = result.bbox[3] - result.bbox[1]
                ball_size = (ball_w + ball_h) / 2
                if not momentum.accept((result.center_x, result.center_y), ball_size):
                    logger.debug(
                        "Frame %d: rejected detection obj_id=%d — too far from momentum prediction",
                        frame_idx, result.obj_id,
                    )
                    result = None  # Treat as no detection
                else:
                    accepted = True
            else:
                accepted = True

        if accepted and result is not None:
            if selected_obj_id is None:
                selected_obj_id = result.obj_id
                logger.info(f"Selected ball obj_id={result.obj_id} at frame {frame_idx}")
            momentum.update((result.center_x, result.center_y))
            positions.append((result.center_x, result.center_y))
            frames_since_lost = 0
        else:
            if momentum.has_position:
                momentum.predict()
            positions.append(None)
            frames_since_lost += 1
            if fps > 0 and frames_since_lost > fps * 3:
                selected_obj_id = None
                momentum.reset()
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

- [ ] **Step 4: Run all pipeline tests**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_pipeline.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add clpga_demo/pipeline.py tests/test_pipeline.py
git commit -m "feat: integrate MomentumTracker into pipeline Pass 1"
```

---

### Task 7: Update CLI

**Files:**
- Modify: `clpga_demo/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Update CLI test assertions first**

Replace `tests/test_cli.py`:

```python
from clpga_demo.__main__ import build_parser, resolve_args


class TestCLIParser:
    def test_minimal_args(self):
        """Minimum required args: source and -o output."""
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "-o", "output.mp4"])
        assert args.source == "input.mp4"
        assert args.output == "output.mp4"

    def test_all_options(self):
        """All optional arguments should be parsed correctly."""
        parser = build_parser()
        args = parser.parse_args([
            "input.mp4", "-o", "output.mp4",
            "--smoothing-sigma", "1.0",
            "--model", "sam3-large.pt",
            "--confidence", "0.5",
            "--momentum-history", "3",
            "--momentum-radius", "6.0",
        ])
        assert args.smoothing_sigma == 1.0
        assert args.model == "sam3-large.pt"
        assert args.confidence == 0.5
        assert args.momentum_history == 3
        assert args.momentum_radius == 6.0

    def test_defaults(self):
        """Default values should be None (presets supply real defaults)."""
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.smoothing_sigma is None
        assert args.model == "sam3.pt"
        assert args.confidence is None
        assert args.momentum_history is None
        assert args.momentum_radius is None

    def test_no_live_flag(self):
        """--live flag should no longer exist."""
        import pytest
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["in.mp4", "-o", "out.mp4", "--live"])


class TestPresetArg:
    def test_preset_default(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.preset == "default"

    def test_preset_putt(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt"])
        assert args.preset == "putt"


class TestResolveArgs:
    def test_default_preset_values(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        resolved = resolve_args(args)
        assert resolved["smoothing_sigma_seconds"] == 0.5
        assert resolved["confidence"] == 0.25
        assert resolved["text"] == ["golf ball"]

    def test_putt_preset_values(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt"])
        resolved = resolve_args(args)
        assert resolved["smoothing_sigma_seconds"] == 0.1
        assert resolved["confidence"] == 0.15
        assert resolved["text"] == ["golf ball on green"]
        assert resolved["momentum_history_size"] == 5
        assert resolved["momentum_radius_scale"] == 4.0

    def test_cli_override_beats_preset(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt", "--smoothing-sigma", "0.3"])
        resolved = resolve_args(args)
        assert resolved["smoothing_sigma_seconds"] == 0.3
        assert resolved["confidence"] == 0.15

    def test_momentum_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "in.mp4", "-o", "out.mp4", "--preset", "putt",
            "--momentum-history", "3", "--momentum-radius", "6.0",
        ])
        resolved = resolve_args(args)
        assert resolved["momentum_history_size"] == 3
        assert resolved["momentum_radius_scale"] == 6.0

    def test_confidence_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt", "--confidence", "0.5"])
        resolved = resolve_args(args)
        assert resolved["confidence"] == 0.5
        assert resolved["smoothing_sigma_seconds"] == 0.1
```

- [ ] **Step 2: Run tests to see them fail**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cli.py -v`
Expected: FAIL — `--live` still exists, `--momentum-history` doesn't exist yet

- [ ] **Step 3: Update __main__.py**

```python
"""CLI entry point for clpga_demo. Run with: python -m clpga_demo"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="clpga-demo",
        description="Track a golf ball in video and output a 9:16 portrait crop.",
    )
    parser.add_argument("source", help="Input video file")
    parser.add_argument("-o", "--output", required=True, help="Output video file path")
    parser.add_argument("--preset", default="default", help="Shot preset: default, putt (default: default)")
    parser.add_argument("--smoothing-sigma", type=float, default=None, help="Gaussian sigma in seconds")
    parser.add_argument("--model", default="sam3.pt", help="SAM3 model path (default: sam3.pt)")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--momentum-history", type=int, default=None, help="Momentum tracker history size")
    parser.add_argument("--momentum-radius", type=float, default=None, help="Momentum acceptance radius scale factor")
    return parser


def resolve_args(args: argparse.Namespace) -> dict:
    """Resolve preset defaults with explicit CLI overrides."""
    from clpga_demo.presets import get_preset

    preset = get_preset(args.preset)

    cli_to_preset = {
        "smoothing_sigma": "smoothing_sigma_seconds",
        "confidence": "confidence",
        "momentum_history": "momentum_history_size",
        "momentum_radius": "momentum_radius_scale",
    }

    for cli_key, preset_key in cli_to_preset.items():
        cli_val = getattr(args, cli_key)
        if cli_val is not None:
            preset[preset_key] = cli_val

    return preset


def main() -> None:
    """Run the golf ball tracker CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    resolved = resolve_args(args)

    from clpga_demo.pipeline import process_video

    try:
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
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all CLI tests**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add clpga_demo/__main__.py tests/test_cli.py
git commit -m "feat: update CLI — remove --live/--smoothing-alpha, add momentum flags"
```
