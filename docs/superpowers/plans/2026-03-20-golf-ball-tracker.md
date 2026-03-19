# Golf Ball Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool and Python API that tracks a golf ball in video using SAM3 and outputs a 9:16 portrait-cropped video centered on the ball with smooth camera following.

**Architecture:** Hybrid pipeline — two-pass for pre-recorded files (Gaussian smoothing over full trajectory), single-pass for live streams (EMA smoothing). Four modules: tracker (SAM3 wrapper), smoother (trajectory smoothing), cropper (portrait crop extraction), pipeline (orchestration). CLI via `__main__.py`.

**Tech Stack:** Python 3.13+, ultralytics SAM3 (`SAM3VideoSemanticPredictor`), OpenCV (`cv2`), SciPy (`gaussian_filter1d`), NumPy

**Spec:** `docs/superpowers/specs/2026-03-19-golf-ball-tracker-design.md`

---

## File Structure

```
clpga_demo/
├── __init__.py          # Package init, re-exports process_video and process_stream
├── __main__.py          # CLI entry point (argparse)
├── tracker.py           # SAM3 ball detection, tracking, and ball selection heuristic
├── smoother.py          # GaussianSmoother and EMASmoother classes
├── cropper.py           # Portrait crop calculation and video writing
├── pipeline.py          # Orchestrates tracker -> smoother -> cropper
tests/
├── __init__.py
├── test_smoother.py     # Unit tests for smoothing logic
├── test_cropper.py      # Unit tests for crop calculation and edge clamping
├── test_tracker.py      # Unit tests for ball selection heuristic (mocked SAM3)
├── test_pipeline.py     # Integration tests with mocked tracker
└── test_cli.py          # CLI argument parsing tests
pyproject.toml           # Add opencv-python, scipy, pytest
```

---

### Task 1: Project Setup — Convert to Package and Add Dependencies

**Files:**
- Create: `clpga_demo/__init__.py`
- Create: `clpga_demo/__main__.py` (placeholder)
- Create: `tests/__init__.py`
- Modify: `pyproject.toml`
- Delete: `main.py` (replaced by package)

- [ ] **Step 1: Update pyproject.toml with dependencies and test config**

```toml
[project]
name = "clpga-demo"
version = "0.1.0"
description = "Track golf balls in video using SAM3 and output portrait-cropped video"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "ultralytics>=8.4.23",
    "opencv-python>=4.8",
    "scipy>=1.11",
]

[project.scripts]
clpga-demo = "clpga_demo.__main__:main"

[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Create package directory and init file**

```bash
mkdir -p clpga_demo tests
```

`clpga_demo/__init__.py`:
```python
"""Golf ball tracker — track and crop golf balls from video using SAM3."""


def __getattr__(name: str):
    """Lazy imports — avoids ImportError before pipeline.py exists."""
    if name == "process_video":
        from clpga_demo.pipeline import process_video
        return process_video
    if name == "process_stream":
        from clpga_demo.pipeline import process_stream
        return process_stream
    raise AttributeError(f"module 'clpga_demo' has no attribute {name!r}")


__all__ = ["process_video", "process_stream"]
```

- [ ] **Step 3: Create placeholder __main__.py**

`clpga_demo/__main__.py`:
```python
"""CLI entry point for clpga_demo. Run with: python -m clpga_demo"""


def main():
    print("clpga-demo: golf ball tracker (not yet implemented)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create tests/__init__.py**

`tests/__init__.py`: empty file

- [ ] **Step 5: Delete old main.py**

```bash
rm main.py
```

- [ ] **Step 6: Install dependencies**

```bash
uv sync
```

- [ ] **Step 7: Verify package runs**

Run: `uv run python -m clpga_demo`
Expected: `clpga-demo: golf ball tracker (not yet implemented)`

- [ ] **Step 8: Commit**

```bash
git add clpga_demo/ tests/ pyproject.toml uv.lock
git rm main.py
git commit -m "feat: convert to package structure, add opencv and scipy deps"
```

---

### Task 2: Smoother Module — Gaussian and EMA Smoothing

**Files:**
- Create: `clpga_demo/smoother.py`
- Create: `tests/test_smoother.py`

- [ ] **Step 1: Write failing tests for GaussianSmoother**

`tests/test_smoother.py`:
```python
import numpy as np
import pytest

from clpga_demo.smoother import EMASmoother, GaussianSmoother


class TestGaussianSmoother:
    def test_constant_positions_unchanged(self):
        """Smoothing constant positions should return the same positions."""
        smoother = GaussianSmoother(sigma=5.0)
        positions = np.array([[100.0, 200.0]] * 30)
        result = smoother.smooth(positions)
        np.testing.assert_allclose(result, positions, atol=1e-6)

    def test_reduces_jitter(self):
        """Smoothing should reduce the variance of jittery positions."""
        smoother = GaussianSmoother(sigma=5.0)
        base = np.array([[100.0, 200.0]] * 30)
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, 10, size=base.shape)
        noisy = base + jitter
        result = smoother.smooth(noisy)
        # Smoothed variance should be less than original
        assert np.var(result[:, 0]) < np.var(noisy[:, 0])
        assert np.var(result[:, 1]) < np.var(noisy[:, 1])

    def test_preserves_shape(self):
        """Output shape must match input shape."""
        smoother = GaussianSmoother(sigma=3.0)
        positions = np.array([[i * 10.0, i * 5.0] for i in range(20)])
        result = smoother.smooth(positions)
        assert result.shape == positions.shape

    def test_interpolates_missing_frames(self):
        """NaN positions should be interpolated before smoothing."""
        smoother = GaussianSmoother(sigma=3.0)
        positions = np.array([
            [100.0, 200.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [130.0, 230.0],
        ])
        result = smoother.smooth(positions)
        # No NaNs in output
        assert not np.any(np.isnan(result))
        # Interpolated middle values should be between start and end
        assert 100.0 < result[1, 0] < 130.0
        assert 100.0 < result[2, 0] < 130.0

    def test_from_fps_creates_correct_sigma(self):
        """Factory method should compute sigma from fps and seconds."""
        smoother = GaussianSmoother.from_fps(fps=30.0, sigma_seconds=0.5)
        assert smoother.sigma == 15.0

        smoother60 = GaussianSmoother.from_fps(fps=60.0, sigma_seconds=0.5)
        assert smoother60.sigma == 30.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_smoother.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write failing tests for EMASmoother and run to verify they fail**

Add to `tests/test_smoother.py`:
```python
class TestEMASmoother:
    def test_first_position_passthrough(self):
        """First position should be returned as-is."""
        smoother = EMASmoother(alpha=0.15)
        result = smoother.update(100.0, 200.0)
        assert result == (100.0, 200.0)

    def test_smooths_toward_new_position(self):
        """After update, smoothed position should move toward new position."""
        smoother = EMASmoother(alpha=0.5)
        smoother.update(100.0, 200.0)
        x, y = smoother.update(200.0, 300.0)
        assert x == pytest.approx(150.0)
        assert y == pytest.approx(250.0)

    def test_hold_on_none(self):
        """None input (occlusion) should hold the last known position."""
        smoother = EMASmoother(alpha=0.15)
        smoother.update(100.0, 200.0)
        x, y = smoother.update(None, None)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_low_alpha_more_smooth(self):
        """Lower alpha should produce smoother (slower-moving) results."""
        slow = EMASmoother(alpha=0.1)
        fast = EMASmoother(alpha=0.9)
        slow.update(100.0, 100.0)
        fast.update(100.0, 100.0)
        sx, _ = slow.update(200.0, 100.0)
        fx, _ = fast.update(200.0, 100.0)
        # Fast alpha should move further toward 200
        assert fx > sx

    def test_reset(self):
        """Reset should clear state so next update is passthrough."""
        smoother = EMASmoother(alpha=0.5)
        smoother.update(100.0, 200.0)
        smoother.update(150.0, 250.0)
        smoother.reset()
        result = smoother.update(500.0, 600.0)
        assert result == (500.0, 600.0)
```

- [ ] **Step 4: Implement smoother.py**

`clpga_demo/smoother.py`:
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


class EMASmoother:
    """Causal exponential moving average smoother for live streams."""

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float | None, y: float | None) -> tuple[float, float]:
        """Update with a new position. Pass None for both to hold during occlusion."""
        if self._x is None or self._y is None:
            # First position — passthrough
            self._x = x if x is not None else 0.0
            self._y = y if y is not None else 0.0
            return (self._x, self._y)

        if x is None or y is None:
            # Occlusion — hold last position
            return (self._x, self._y)

        self._x = self.alpha * x + (1 - self.alpha) * self._x
        self._y = self.alpha * y + (1 - self.alpha) * self._y
        return (self._x, self._y)

    def reset(self) -> None:
        """Clear state."""
        self._x = None
        self._y = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_smoother.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add clpga_demo/smoother.py tests/test_smoother.py
git commit -m "feat: add Gaussian and EMA trajectory smoothers with tests"
```

---

### Task 3: Cropper Module — Portrait Crop Calculation

**Files:**
- Create: `clpga_demo/cropper.py`
- Create: `tests/test_cropper.py`

- [ ] **Step 1: Write failing tests for crop region calculation**

`tests/test_cropper.py`:
```python
import numpy as np
import pytest

from clpga_demo.cropper import CropRegion, calculate_crop


class TestCalculateCrop:
    def test_landscape_1080p_centered(self):
        """1920x1080 source with ball centered should produce 607x1080 crop."""
        crop = calculate_crop(
            ball_x=960.0, ball_y=540.0,
            source_width=1920, source_height=1080,
        )
        expected_w = int(1080 * 9 / 16)  # 607
        assert crop.width == expected_w
        assert crop.height == 1080
        # Centered: x should be (960 - 607/2) ≈ 656
        assert crop.x == pytest.approx(960 - expected_w / 2, abs=1)
        assert crop.y == 0

    def test_edge_clamp_left(self):
        """Ball near left edge should clamp crop x to 0."""
        crop = calculate_crop(
            ball_x=50.0, ball_y=540.0,
            source_width=1920, source_height=1080,
        )
        assert crop.x == 0
        assert crop.width == int(1080 * 9 / 16)

    def test_edge_clamp_right(self):
        """Ball near right edge should clamp crop to not exceed frame."""
        crop = calculate_crop(
            ball_x=1900.0, ball_y=540.0,
            source_width=1920, source_height=1080,
        )
        assert crop.x + crop.width <= 1920

    def test_narrow_source_crops_vertically(self):
        """Source narrower than 9:16 should crop vertically instead."""
        # 360x640 source (9:16 already) — should use full width
        crop = calculate_crop(
            ball_x=180.0, ball_y=320.0,
            source_width=360, source_height=640,
        )
        assert crop.width == 360
        expected_h = int(360 * 16 / 9)  # 640
        assert crop.height == expected_h

    def test_very_narrow_source_crops_vertically(self):
        """Source much narrower than 9:16 should crop height to fit ratio."""
        # 200x800 source — too narrow, must crop vertically
        crop = calculate_crop(
            ball_x=100.0, ball_y=400.0,
            source_width=200, source_height=800,
        )
        assert crop.width == 200
        expected_h = int(200 * 16 / 9)  # 355
        assert crop.height == expected_h
        # Centered vertically on ball
        assert crop.y + crop.height <= 800

    def test_vertical_edge_clamp(self):
        """Ball near bottom of narrow source should clamp y."""
        crop = calculate_crop(
            ball_x=100.0, ball_y=790.0,
            source_width=200, source_height=800,
        )
        assert crop.y + crop.height <= 800
        assert crop.y >= 0


class TestCropRegion:
    def test_apply_extracts_region(self):
        """CropRegion.apply should extract the correct sub-array from a frame."""
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        crop = CropRegion(x=10, y=20, width=50, height=30)
        result = crop.apply(frame)
        assert result.shape == (30, 50, 3)
        np.testing.assert_array_equal(result, frame[20:50, 10:60, :])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cropper.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement cropper.py**

`clpga_demo/cropper.py`:
```python
"""Portrait crop calculation and video writing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropRegion:
    """A rectangular crop region within a video frame."""

    x: int
    y: int
    width: int
    height: int

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Extract this crop region from a frame (H, W, C) array."""
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


def calculate_crop(
    ball_x: float,
    ball_y: float,
    source_width: int,
    source_height: int,
) -> CropRegion:
    """Calculate a 9:16 portrait crop region centered on the ball position.

    For landscape/wide sources: uses full height, crops width to 9:16.
    For narrow sources (already narrower than 9:16): uses full width, crops height to 9:16.
    """
    target_w = int(source_height * 9 / 16)

    if target_w <= source_width:
        # Landscape or wide enough — crop horizontally, use full height
        crop_w = target_w
        crop_h = source_height
        crop_x = int(max(0, min(ball_x - crop_w / 2, source_width - crop_w)))
        crop_y = 0
    else:
        # Source is narrower than 9:16 — crop vertically, use full width
        crop_w = source_width
        crop_h = int(source_width * 16 / 9)
        crop_x = 0
        crop_y = int(max(0, min(ball_y - crop_h / 2, source_height - crop_h)))

    return CropRegion(x=crop_x, y=crop_y, width=crop_w, height=crop_h)


class VideoWriter:
    """Writes cropped frames to an output video file."""

    def __init__(self, output_path: str, fps: float, width: int, height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise ValueError(f"Cannot open video writer for {output_path}")

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame (H, W, C) BGR array."""
        self._writer.write(frame)

    def release(self) -> None:
        """Finalize and close the output video."""
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cropper.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/cropper.py tests/test_cropper.py
git commit -m "feat: add portrait crop calculation with edge clamping and video writer"
```

---

### Task 4: Tracker Module — SAM3 Wrapper and Ball Selection

**Files:**
- Create: `clpga_demo/tracker.py`
- Create: `tests/test_tracker.py`

- [ ] **Step 1: Write failing tests for ball selection heuristic**

`tests/test_tracker.py`:
```python
import numpy as np
import pytest

from clpga_demo.tracker import TrackResult, select_ball


class TestSelectBall:
    def test_single_detection_selected(self):
        """With one detection, it should be selected."""
        boxes = np.array([[100, 100, 200, 200, 1, 0.9, 0]])  # x1,y1,x2,y2,obj_id,score,cls
        result = select_ball(boxes, frame_width=1920, frame_height=1080)
        assert result.obj_id == 1

    def test_centered_and_large_preferred(self):
        """A centered, large ball should score higher than a small corner one."""
        boxes = np.array([
            [0, 0, 20, 20, 1, 0.9, 0],          # small, top-left corner
            [900, 500, 1020, 580, 2, 0.9, 0],    # large, centered
        ])
        result = select_ball(boxes, frame_width=1920, frame_height=1080)
        assert result.obj_id == 2

    def test_no_detections_returns_none(self):
        """Empty boxes should return None."""
        boxes = np.array([]).reshape(0, 7)
        result = select_ball(boxes, frame_width=1920, frame_height=1080)
        assert result is None

    def test_track_result_center(self):
        """TrackResult should compute center from bbox."""
        result = TrackResult(
            frame_idx=0,
            center_x=150.0,
            center_y=150.0,
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            obj_id=1,
        )
        assert result.center_x == 150.0
        assert result.center_y == 150.0


class TestStickySelection:
    def test_sticky_follows_same_obj_id(self):
        """select_ball with preferred_obj_id should return that object if present."""
        boxes = np.array([
            [900, 500, 1020, 580, 2, 0.9, 0],  # centered, large
            [100, 100, 120, 120, 5, 0.9, 0],    # small, corner
        ])
        result = select_ball(boxes, frame_width=1920, frame_height=1080, preferred_obj_id=5)
        assert result.obj_id == 5

    def test_sticky_falls_back_when_obj_lost(self):
        """If preferred_obj_id is not in boxes, re-evaluate heuristic."""
        boxes = np.array([
            [900, 500, 1020, 580, 2, 0.9, 0],
            [100, 100, 120, 120, 3, 0.9, 0],
        ])
        result = select_ball(boxes, frame_width=1920, frame_height=1080, preferred_obj_id=99)
        # preferred_obj_id 99 not found — should fall back to heuristic (obj_id=2)
        assert result.obj_id == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tracker.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement tracker.py**

`clpga_demo/tracker.py`:
```python
"""SAM3 ball detection, tracking, and ball selection heuristic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackResult:
    """Tracking result for a single frame."""

    frame_idx: int
    center_x: float
    center_y: float
    bbox: tuple[float, float, float, float]
    confidence: float
    obj_id: int


def select_ball(
    boxes: np.ndarray,
    frame_width: int,
    frame_height: int,
    preferred_obj_id: int | None = None,
    frame_idx: int = 0,
) -> TrackResult | None:
    """Select the best golf ball from detected boxes.

    Args:
        boxes: (N, 7) array with columns [x1, y1, x2, y2, obj_id, score, cls].
        frame_width: Source frame width.
        frame_height: Source frame height.
        preferred_obj_id: If set, prefer this obj_id (sticky selection).
        frame_idx: Current frame index for the TrackResult.

    Returns:
        TrackResult for the selected ball, or None if no detections.
    """
    if len(boxes) == 0:
        return None

    # Sticky selection: if preferred obj_id is present, use it
    if preferred_obj_id is not None:
        for row in boxes:
            if int(row[4]) == preferred_obj_id:
                return _box_to_result(row, frame_idx=frame_idx)

    # Heuristic: score = bbox_area / distance_from_center
    frame_cx = frame_width / 2
    frame_cy = frame_height / 2
    best_score = -1.0
    best_row = None

    for row in boxes:
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        dist = max(np.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2), 1.0)
        score = area / dist
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return None
    return _box_to_result(best_row, frame_idx=frame_idx)


def _box_to_result(row: np.ndarray, frame_idx: int) -> TrackResult:
    """Convert a single box row to a TrackResult."""
    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    return TrackResult(
        frame_idx=frame_idx,
        center_x=(x1 + x2) / 2,
        center_y=(y1 + y2) / 2,
        bbox=(x1, y1, x2, y2),
        confidence=float(row[5]),
        obj_id=int(row[4]),
    )


_FALLBACK_FRAME_THRESHOLD = 30


def track_video(
    source: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    text: list[str] | None = None,
) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
    """Run SAM3 tracking on a video source.

    Yields (frame_idx, original_frame, boxes) for each frame.
    boxes is an (N, 7) array: [x1, y1, x2, y2, obj_id, score, cls].

    If no detections are found in the first 30 frames via text prompt,
    falls back to bbox prompt at center of frame.
    """
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    if text is None:
        text = ["golf ball"]

    overrides = dict(
        conf=confidence,
        task="segment",
        mode="predict",
        model=model,
        half=True,
        save=False,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    results = predictor(source=source, text=text, stream=True)

    any_detection = False
    buffered_frames: list[tuple[int, np.ndarray, np.ndarray]] = []

    for frame_idx, r in enumerate(results):
        boxes = r.boxes.data.cpu().numpy() if r.boxes is not None and len(r.boxes) > 0 else np.empty((0, 7))
        orig_frame = r.orig_img

        if len(boxes) > 0:
            any_detection = True

        # Buffer first N frames to check for fallback
        if not any_detection and frame_idx < _FALLBACK_FRAME_THRESHOLD:
            buffered_frames.append((frame_idx, orig_frame, boxes))
            continue

        # Fallback: no detections in first 30 frames — re-run with bbox prompt
        if not any_detection and frame_idx == _FALLBACK_FRAME_THRESHOLD:
            logger.warning(
                "No golf ball detected in first %d frames via text prompt. "
                "Falling back to center-of-frame bounding box.",
                _FALLBACK_FRAME_THRESHOLD,
            )
            h, w = orig_frame.shape[:2]
            cx, cy = w / 2, h / 2
            size = min(w, h) / 4
            center_bbox = [[cx - size, cy - size, cx + size, cy + size]]
            predictor.add_prompt(frame_idx=0, bboxes=np.array(center_bbox))
            # Yield buffered frames (they had no detections)
            for buffered in buffered_frames:
                yield buffered
            buffered_frames.clear()

        yield frame_idx, orig_frame, boxes

    # If we never hit the threshold, yield any remaining buffered frames
    for buffered in buffered_frames:
        yield buffered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracker.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/tracker.py tests/test_tracker.py
git commit -m "feat: add SAM3 tracker with ball selection heuristic"
```

---

### Task 5: Pipeline Module — Orchestration

**Files:**
- Create: `clpga_demo/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for process_video (mocked tracker)**

`tests/test_pipeline.py`:
```python
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from clpga_demo.pipeline import process_video


def _create_test_video(path: str, width: int = 320, height: int = 240, frames: int = 10) -> None:
    """Create a small test video with a white circle (fake ball)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a moving white circle
        cx = 100 + i * 10
        cy = 120
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _mock_track_video(source, **kwargs):
    """Mock tracker that yields fake detections with a moving ball."""
    cap = cv2.VideoCapture(source)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cx = 100 + frame_idx * 10
        cy = 120
        boxes = np.array([[cx - 10, cy - 10, cx + 10, cy + 10, 1, 0.95, 0]])
        yield frame_idx, frame, boxes
        frame_idx += 1
    cap.release()


class TestProcessVideo:
    def test_creates_output_file(self, tmp_path):
        """process_video should create an output video file."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_video(input_path, output_path)

        assert Path(output_path).exists()
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        # Output should be portrait 9:16 ratio
        expected_w = int(h * 9 / 16)
        assert w == expected_w

    def test_raises_on_missing_input(self):
        """process_video should raise ValueError for nonexistent input."""
        with pytest.raises(ValueError, match="does not exist"):
            process_video("/nonexistent/video.mp4", "/tmp/out.mp4")

    def test_raises_on_no_detections(self, tmp_path):
        """process_video should raise RuntimeError when no ball is detected."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        def _empty_tracker(source, **kwargs):
            cap = cv2.VideoCapture(source)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_idx, frame, np.empty((0, 7))
                frame_idx += 1
            cap.release()

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _empty_tracker(s)):
            with pytest.raises(RuntimeError, match="No golf ball detected"):
                process_video(input_path, output_path)


class TestProcessStream:
    def test_creates_output_file(self, tmp_path):
        """process_stream should create an output video file."""
        from clpga_demo.pipeline import process_stream

        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_stream(input_path, output_path)

        assert Path(output_path).exists()
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        expected_w = int(h * 9 / 16)
        assert w == expected_w
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement pipeline.py**

`clpga_demo/pipeline.py`:
```python
"""Pipeline orchestration — tracker -> smoother -> cropper."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from clpga_demo.cropper import VideoWriter, calculate_crop
from clpga_demo.smoother import EMASmoother, GaussianSmoother
from clpga_demo.tracker import TrackResult, select_ball, track_video

logger = logging.getLogger(__name__)


def process_video(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_sigma_seconds: float = 0.5,
) -> None:
    """Process a pre-recorded video: track ball, smooth trajectory, crop portrait.

    Two-pass pipeline:
      Pass 1 — collect all ball positions from SAM3 tracking.
      Pass 2 — smooth positions, re-read video, write cropped output.
    """
    if not Path(source).exists():
        raise ValueError(f"Input video does not exist: {source}")

    # --- Pass 1: Collect positions ---
    positions: list[tuple[float, float] | None] = []
    frames_meta: dict = {}
    selected_obj_id: int | None = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames_meta = {"fps": fps, "width": src_w, "height": src_h}

    for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence):
        result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)
        if result is not None:
            if selected_obj_id is None:
                selected_obj_id = result.obj_id
                logger.info(f"Selected ball obj_id={result.obj_id} at frame {frame_idx}")
            positions.append((result.center_x, result.center_y))
        else:
            positions.append(None)

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


def process_stream(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_alpha: float = 0.15,
) -> None:
    """Process a live stream: track ball, smooth with EMA, crop portrait in real-time.

    Single-pass pipeline — each frame is tracked, smoothed, and cropped immediately.
    """
    import time

    smoother = EMASmoother(alpha=smoothing_alpha)
    selected_obj_id: int | None = None
    writer: VideoWriter | None = None
    frames_since_lost = 0
    fps_for_loss = 30.0  # will be updated from source
    max_retries = 3

    try:
        retries = 0
        while retries <= max_retries:
            try:
                for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence):
                    src_h, src_w = orig_frame.shape[:2]

                    result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)

                    if result is not None:
                        selected_obj_id = result.obj_id
                        sx, sy = smoother.update(result.center_x, result.center_y)
                        frames_since_lost = 0
                    else:
                        sx, sy = smoother.update(None, None)
                        frames_since_lost += 1
                        # Re-evaluate heuristic after 3 seconds of loss
                        if frames_since_lost > fps_for_loss * 3:
                            selected_obj_id = None
                            frames_since_lost = 0

                    crop = calculate_crop(sx, sy, src_w, src_h)

                    if writer is None:
                        cap_temp = cv2.VideoCapture(source)
                        fps_for_loss = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
                        cap_temp.release()
                        writer = VideoWriter(output, fps_for_loss, crop.width, crop.height)

                    cropped = crop.apply(orig_frame)
                    writer.write(cropped)

                break  # Normal completion

            except (cv2.error, ConnectionError, OSError) as e:
                retries += 1
                if retries > max_retries:
                    logger.warning("Stream disconnected after %d retries. Finalizing output.", max_retries)
                    break
                logger.warning("Stream error (attempt %d/%d): %s. Retrying in 1s...", retries, max_retries, e)
                time.sleep(1)

    finally:
        if writer is not None:
            writer.release()

    if writer is None:
        raise RuntimeError("No frames received from stream")

    logger.info(f"Stream output written to {output}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestration for video and stream processing"
```

---

### Task 6: CLI Entry Point

**Files:**
- Modify: `clpga_demo/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI argument parsing**

`tests/test_cli.py`:
```python
from unittest.mock import patch

from clpga_demo.__main__ import build_parser


class TestCLIParser:
    def test_minimal_args(self):
        """Minimum required args: source and -o output."""
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "-o", "output.mp4"])
        assert args.source == "input.mp4"
        assert args.output == "output.mp4"
        assert args.live is False

    def test_live_flag(self):
        """--live flag should be parsed."""
        parser = build_parser()
        args = parser.parse_args(["rtsp://cam/stream", "-o", "out.mp4", "--live"])
        assert args.live is True

    def test_all_options(self):
        """All optional arguments should be parsed correctly."""
        parser = build_parser()
        args = parser.parse_args([
            "input.mp4", "-o", "output.mp4",
            "--smoothing-sigma", "1.0",
            "--smoothing-alpha", "0.2",
            "--model", "sam3-large.pt",
            "--confidence", "0.5",
        ])
        assert args.smoothing_sigma == 1.0
        assert args.smoothing_alpha == 0.2
        assert args.model == "sam3-large.pt"
        assert args.confidence == 0.5

    def test_defaults(self):
        """Default values should match spec."""
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.smoothing_sigma == 0.5
        assert args.smoothing_alpha == 0.15
        assert args.model == "sam3.pt"
        assert args.confidence == 0.25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ImportError` (build_parser doesn't exist)

- [ ] **Step 3: Implement __main__.py with argument parsing**

`clpga_demo/__main__.py`:
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
    parser.add_argument("source", help="Input video file or RTSP stream URL")
    parser.add_argument("-o", "--output", required=True, help="Output video file path")
    parser.add_argument("--live", action="store_true", help="Treat source as a live stream (EMA smoothing)")
    parser.add_argument("--smoothing-sigma", type=float, default=0.5, help="Gaussian sigma in seconds (file mode, default: 0.5)")
    parser.add_argument("--smoothing-alpha", type=float, default=0.15, help="EMA alpha (live mode, default: 0.15)")
    parser.add_argument("--model", default="sam3.pt", help="SAM3 model path (default: sam3.pt)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    return parser


def main() -> None:
    """Run the golf ball tracker CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    from clpga_demo.pipeline import process_stream, process_video

    try:
        if args.live:
            process_stream(
                source=args.source,
                output=args.output,
                model=args.model,
                confidence=args.confidence,
                smoothing_alpha=args.smoothing_alpha,
            )
        else:
            process_video(
                source=args.source,
                output=args.output,
                model=args.model,
                confidence=args.confidence,
                smoothing_sigma_seconds=args.smoothing_sigma,
            )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/__main__.py tests/test_cli.py
git commit -m "feat: add CLI entry point with argument parsing"
```

---

### Task 7: Wire Up __init__.py and Final Integration

**Files:**
- Modify: `clpga_demo/__init__.py`

- [ ] **Step 1: Verify package import works**

Run: `uv run python -c "from clpga_demo.pipeline import process_video, process_stream; print('OK')"`
Expected: `OK`

- [ ] **Step 2: Verify CLI runs with --help**

Run: `uv run python -m clpga_demo --help`
Expected: Help text showing all arguments

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS (smoother: 10, cropper: 7, tracker: 6, pipeline: 4, cli: 4 = ~31 tests)

- [ ] **Step 4: Commit any final adjustments**

```bash
git add -A
git commit -m "feat: wire up package exports and verify integration"
```

---

## Execution Notes

- **Task order matters:** Tasks 1 → 2 → 3 → 4 can be sequential (each builds on the previous). Task 2 and 3 are independent of each other and could run in parallel.
- **SAM3 model download:** The first run with a real video will trigger SAM3 model download (~2.4GB). This is handled automatically by ultralytics.
- **Testing without GPU:** Unit tests (tasks 2-4, 6) use mocked data and don't require a GPU. The pipeline integration test (task 5) mocks the tracker. Only real end-to-end testing requires a GPU with SAM3.
- **Live stream testing:** Testing RTSP streams requires a real camera or test RTSP server. For development, pre-recorded files are sufficient.
