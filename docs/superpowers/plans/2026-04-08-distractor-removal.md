# Distractor Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in cleaning pass that uses SAM3 segment-everything + Netflix void-model to remove distractors from analysis frames, producing a refined trajectory for the final crop.

**Architecture:** Two new modules (`void_model.py` for model download/load/infer, `cleaner.py` for corridor computation + quadmask generation + orchestration) inserted as Pass 2 between the existing raw tracking (Pass 1) and smooth/crop (Pass 3). Enabled via `--clean` CLI flag. Original frames are never modified.

**Tech Stack:** ultralytics SAM3 (segment-everything mode), netflix/void-model (CogVideoX-Fun), huggingface_hub, numpy, OpenCV, scipy

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `clpga_demo/void_model.py` | Create | Download, load, and run void-model inference |
| `clpga_demo/cleaner.py` | Create | Corridor computation, SAM3 segment-everything, quadmask generation, segment orchestration, ball mask identification |
| `clpga_demo/presets.py` | Modify | Add cleaning-related preset parameters |
| `clpga_demo/__main__.py` | Modify | Add `--clean` and related CLI flags |
| `clpga_demo/pipeline.py` | Modify | Wire Pass 2 between existing Pass 1 and Pass 3 |
| `tests/test_void_model.py` | Create | Tests for VoidModelWrapper |
| `tests/test_cleaner.py` | Create | Tests for FrameCleaner |
| `tests/test_presets.py` | Modify | Tests for new preset keys |
| `tests/test_cli.py` | Modify | Tests for new CLI flags |
| `tests/test_pipeline.py` | Modify | Tests for pipeline with cleaning enabled |

---

### Task 1: VoidModelWrapper — Download & Load

**Files:**
- Create: `clpga_demo/void_model.py`
- Create: `tests/test_void_model.py`

- [ ] **Step 1: Write the failing test for download_if_needed**

```python
# tests/test_void_model.py
from unittest.mock import patch, MagicMock
import pytest

from clpga_demo.void_model import VoidModelWrapper


class TestDownload:
    def test_download_if_needed_calls_snapshot_download(self):
        """download_if_needed should call snapshot_download for both repos."""
        wrapper = VoidModelWrapper(model_dir=None, device="cpu")

        with patch("clpga_demo.void_model.snapshot_download") as mock_dl:
            mock_dl.return_value = "/fake/path"
            result = wrapper.download_if_needed()

        assert mock_dl.call_count == 2
        repo_ids = [call.kwargs.get("repo_id") or call.args[0] for call in mock_dl.call_args_list]
        assert "netflix/void-model" in repo_ids
        assert "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP" in repo_ids

    def test_download_skipped_when_model_dir_provided(self):
        """When model_dir is explicitly set, skip download."""
        wrapper = VoidModelWrapper(model_dir="/existing/models", device="cpu")

        with patch("clpga_demo.void_model.snapshot_download") as mock_dl:
            result = wrapper.download_if_needed()

        mock_dl.assert_not_called()
        assert result == "/existing/models"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_void_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clpga_demo.void_model'`

- [ ] **Step 3: Implement VoidModelWrapper with download_if_needed**

```python
# clpga_demo/void_model.py
"""Netflix void-model wrapper: download, load, and run video inpainting."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_VOID_REPO = "netflix/void-model"
_BASE_REPO = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"


class VoidModelWrapper:
    """Wraps Netflix void-model for video inpainting of distractor regions."""

    def __init__(self, model_dir: str | None, device: str = "cuda") -> None:
        self._model_dir = model_dir
        self._device = device
        self._base_model_dir: str | None = None
        self._void_dir: str | None = None
        self._loaded = False

    def download_if_needed(self) -> str:
        """Download void-model and base model from HF if model_dir is not set.

        Returns the directory containing the void-model checkpoints.
        """
        if self._model_dir is not None:
            self._void_dir = self._model_dir
            return self._model_dir

        logger.info("Downloading base model from %s ...", _BASE_REPO)
        self._base_model_dir = snapshot_download(
            repo_id=_BASE_REPO,
            local_dir=str(Path.home() / ".cache" / "clpga" / "CogVideoX-Fun-V1.5-5b-InP"),
        )

        logger.info("Downloading void-model from %s ...", _VOID_REPO)
        self._void_dir = snapshot_download(
            repo_id=_VOID_REPO,
            local_dir=str(Path.home() / ".cache" / "clpga" / "void-model"),
        )

        return self._void_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_void_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/void_model.py tests/test_void_model.py
git commit -m "feat: add VoidModelWrapper with HF download support"
```

---

### Task 2: VoidModelWrapper — Inpaint Interface

**Files:**
- Modify: `clpga_demo/void_model.py`
- Modify: `tests/test_void_model.py`

- [ ] **Step 1: Write the failing test for inpaint**

```python
# Append to tests/test_void_model.py

import numpy as np
import tempfile
from pathlib import Path


class TestInpaint:
    def test_inpaint_returns_cleaned_frames(self):
        """inpaint should return an ndarray of cleaned frames with same shape as input."""
        wrapper = VoidModelWrapper(model_dir="/fake/models", device="cpu")
        wrapper._void_dir = "/fake/models"
        wrapper._base_model_dir = "/fake/base"

        video_segment = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmask_segment = np.full((10, 384, 672), 255, dtype=np.uint8)
        # Mark a region for removal
        quadmask_segment[:, 100:150, 200:250] = 0

        with patch("clpga_demo.void_model.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Mock the output video reading
            with patch("clpga_demo.void_model.cv2.VideoCapture") as mock_cap:
                mock_cap_instance = MagicMock()
                mock_cap.return_value = mock_cap_instance
                mock_cap_instance.isOpened.return_value = True
                mock_cap_instance.get.side_effect = lambda prop: {
                    0: 672.0,  # WIDTH
                    1: 384.0,  # HEIGHT
                    2: 10.0,   # FRAME_COUNT
                }[{3: 0, 4: 1, 7: 2}.get(prop, prop)]
                frames_returned = [True] * 10 + [False]
                mock_cap_instance.read.side_effect = [
                    (ok, np.zeros((384, 672, 3), dtype=np.uint8)) if ok else (False, None)
                    for ok in frames_returned
                ]
                result = wrapper.inpaint(video_segment, quadmask_segment, "golf course background")

        assert result.shape == video_segment.shape
        assert result.dtype == np.uint8

    def test_inpaint_raises_on_subprocess_failure(self):
        """inpaint should raise RuntimeError if void-model process fails."""
        wrapper = VoidModelWrapper(model_dir="/fake/models", device="cpu")
        wrapper._void_dir = "/fake/models"
        wrapper._base_model_dir = "/fake/base"

        video_segment = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmask_segment = np.full((10, 384, 672), 255, dtype=np.uint8)

        with patch("clpga_demo.void_model.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="CUDA OOM")
            with pytest.raises(RuntimeError, match="void-model inference failed"):
                wrapper.inpaint(video_segment, quadmask_segment, "golf course background")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_void_model.py::TestInpaint -v`
Expected: FAIL — `AttributeError: 'VoidModelWrapper' object has no attribute 'inpaint'`

- [ ] **Step 3: Implement inpaint method**

The void-model uses a CLI-based interface with a specific directory structure. We write the video segment and quadmask to temp files, invoke the predict script via subprocess, then read back the result.

```python
# Add to clpga_demo/void_model.py — imports at top:
import json
import subprocess
import tempfile

import cv2
import numpy as np

# Add method to VoidModelWrapper class:

    def inpaint(
        self,
        video_segment: np.ndarray,
        quadmask_segment: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Run void-model on a video segment.

        Args:
            video_segment: (T, H, W, 3) uint8 BGR frames.
            quadmask_segment: (T, H, W) uint8 quadmask (0=remove, 127=affected, 255=keep).
            prompt: Background scene description for inpainting.

        Returns:
            (T, H, W, 3) uint8 BGR cleaned frames.
        """
        num_frames, h, w = video_segment.shape[:3]

        with tempfile.TemporaryDirectory(prefix="void_") as tmpdir:
            seq_dir = Path(tmpdir) / "seq"
            seq_dir.mkdir()

            # Write input video
            input_path = str(seq_dir / "input_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(input_path, fourcc, 30.0, (w, h))
            for i in range(num_frames):
                writer.write(video_segment[i])
            writer.release()

            # Write quadmask video
            mask_path = str(seq_dir / "quadmask_0.mp4")
            mask_writer = cv2.VideoWriter(mask_path, fourcc, 30.0, (w, h), isColor=False)
            for i in range(num_frames):
                mask_writer.write(quadmask_segment[i])
            mask_writer.release()

            # Write prompt
            prompt_path = str(seq_dir / "prompt.json")
            with open(prompt_path, "w") as f:
                json.dump({"bg": prompt}, f)

            # Run void-model
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            cmd = [
                "python",
                str(Path(self._void_dir) / "inference" / "cogvideox_fun" / "predict_v2v.py"),
                "--config", str(Path(self._void_dir) / "config" / "quadmask_cogvideox.py"),
                f"--config.data.data_rootdir={tmpdir}",
                "--config.experiment.run_seqs=seq",
                f"--config.experiment.save_path={output_dir}",
                f"--config.video_model.transformer_path={Path(self._void_dir) / 'void_pass1.safetensors'}",
                f"--config.video_model.model_name={self._base_model_dir}",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"void-model inference failed: {result.stderr}")

            # Read output video
            output_files = list(output_dir.rglob("*.mp4"))
            if not output_files:
                raise RuntimeError("void-model produced no output video")

            cap = cv2.VideoCapture(str(output_files[0]))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot read void-model output: {output_files[0]}")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            # Resize back to original dimensions if void-model changed resolution
            cleaned = np.stack(frames[:num_frames])
            if cleaned.shape[1:3] != (h, w):
                resized = np.empty_like(video_segment)
                for i in range(len(cleaned)):
                    resized[i] = cv2.resize(cleaned[i], (w, h))
                cleaned = resized

            return cleaned
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_void_model.py::TestInpaint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/void_model.py tests/test_void_model.py
git commit -m "feat: add void-model inpaint method with subprocess invocation"
```

---

### Task 3: FrameCleaner — Corridor Computation

**Files:**
- Create: `clpga_demo/cleaner.py`
- Create: `tests/test_cleaner.py`

- [ ] **Step 1: Write the failing test for compute_corridors**

```python
# tests/test_cleaner.py
import numpy as np
import pytest

from clpga_demo.cleaner import FrameCleaner, Corridor


class TestCorridorComputation:
    def test_corridor_size_scales_with_ball_size(self):
        """Corridor radius should be at least ball_size * corridor_multiplier."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        # 5 frames, ball at y=100, moving right, speed=0 (stationary)
        rough_trajectory = [(100.0, 100.0), (100.0, 100.0), (100.0, 100.0), (100.0, 100.0), (100.0, 100.0)]
        ball_sizes = [20.0, 20.0, 20.0, 20.0, 20.0]
        speeds = [0.0, 0.0, 0.0, 0.0, 0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert len(corridors) == 5
        # With speed=0, corridor radius = ball_size * multiplier = 20 * 4 = 80
        for c in corridors:
            assert c.radius == 80.0
            assert c.center_x == 100.0
            assert c.center_y == 100.0

    def test_corridor_size_scales_with_speed(self):
        """Corridor radius should grow with ball speed."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(100.0, 100.0)]
        ball_sizes = [20.0]
        speeds = [50.0]  # Fast-moving ball

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        # speed * corridor_speed_scale * radius_scale = 50 * 1.5 * 4.0 = 300
        # ball_size * corridor_multiplier = 20 * 4 = 80
        # max(80, 300) = 300
        assert corridors[0].radius == 300.0

    def test_corridor_clips_to_frame_bounds(self):
        """Corridor bbox should be clipped to frame edges."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        # Ball near top-left corner
        rough_trajectory = [(10.0, 10.0)]
        ball_sizes = [20.0]
        speeds = [0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert corridors[0].x1 >= 0
        assert corridors[0].y1 >= 0

    def test_corridor_none_trajectory_point_skipped(self):
        """Frames where trajectory is None should produce a None corridor."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(100.0, 100.0), None, (120.0, 100.0)]
        ball_sizes = [20.0, 0.0, 20.0]
        speeds = [0.0, 0.0, 0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert corridors[0] is not None
        assert corridors[1] is None
        assert corridors[2] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestCorridorComputation -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clpga_demo.cleaner'`

- [ ] **Step 3: Implement Corridor and compute_corridors**

```python
# clpga_demo/cleaner.py
"""Distractor removal via SAM3 segment-everything + void-model inpainting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Corridor:
    """A rectangular region around a trajectory point where distractors are searched."""

    center_x: float
    center_y: float
    radius: float
    x1: int
    y1: int
    x2: int
    y2: int


class FrameCleaner:
    """Orchestrates distractor removal: corridor computation, segmentation, quadmask generation."""

    def __init__(
        self,
        sam3_model,
        void_model,
        corridor_config: dict,
    ) -> None:
        self._sam3_model = sam3_model
        self._void_model = void_model
        self._corridor_multiplier = corridor_config.get("corridor_multiplier", 4.0)
        self._corridor_speed_scale = corridor_config.get("corridor_speed_scale", 1.5)
        self._radius_scale = corridor_config.get("radius_scale", 4.0)
        self._mask_dilation_px = corridor_config.get("mask_dilation_px", 5)

    def compute_corridors(
        self,
        rough_trajectory: list[tuple[float, float] | None],
        ball_sizes: list[float],
        speeds: list[float],
        frame_w: int,
        frame_h: int,
    ) -> list[Corridor | None]:
        """Compute per-frame corridors from a rough trajectory.

        Args:
            rough_trajectory: Per-frame (x, y) or None if ball was lost.
            ball_sizes: Per-frame ball diameter estimate (avg of bbox w and h).
            speeds: Per-frame ball speed in px/frame.
            frame_w: Frame width.
            frame_h: Frame height.

        Returns:
            List of Corridor (or None for lost frames), one per frame.
        """
        corridors: list[Corridor | None] = []
        for pos, size, speed in zip(rough_trajectory, ball_sizes, speeds):
            if pos is None:
                corridors.append(None)
                continue

            cx, cy = pos
            size_radius = size * self._corridor_multiplier
            speed_radius = speed * self._corridor_speed_scale * self._radius_scale
            radius = max(size_radius, speed_radius)

            x1 = max(0, int(cx - radius))
            y1 = max(0, int(cy - radius))
            x2 = min(frame_w, int(cx + radius))
            y2 = min(frame_h, int(cy + radius))

            corridors.append(Corridor(
                center_x=cx,
                center_y=cy,
                radius=radius,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

        return corridors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestCorridorComputation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/cleaner.py tests/test_cleaner.py
git commit -m "feat: add FrameCleaner with corridor computation"
```

---

### Task 4: FrameCleaner — Ball Mask Identification

**Files:**
- Modify: `clpga_demo/cleaner.py`
- Modify: `tests/test_cleaner.py`

- [ ] **Step 1: Write the failing tests for identify_ball_mask**

```python
# Append to tests/test_cleaner.py

class TestBallMaskIdentification:
    def _make_cleaner(self):
        return FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

    def test_selects_closest_mask_to_trajectory(self):
        """Should pick the mask whose centroid is closest to trajectory point."""
        cleaner = self._make_cleaner()

        # Two masks: one near trajectory (100, 100), one far (300, 300)
        mask_near = np.zeros((480, 640), dtype=bool)
        mask_near[90:110, 90:110] = True  # centroid ~(100, 100)

        mask_far = np.zeros((480, 640), dtype=bool)
        mask_far[290:310, 290:310] = True  # centroid ~(300, 300)

        masks = [mask_near, mask_far]
        trajectory_point = (100.0, 100.0)
        median_ball_size = 20.0

        result = cleaner.identify_ball_mask(masks, trajectory_point, median_ball_size)
        assert result == 0  # mask_near is index 0

    def test_rejects_mask_with_bad_aspect_ratio(self):
        """Closest mask should be rejected if aspect ratio is too large."""
        cleaner = self._make_cleaner()

        # Closest mask is very elongated (1x40 = aspect ratio 40)
        mask_elongated = np.zeros((480, 640), dtype=bool)
        mask_elongated[99:100, 80:120] = True  # 1px tall, 40px wide — centroid near (100, 99)

        mask_round = np.zeros((480, 640), dtype=bool)
        mask_round[190:210, 190:210] = True  # centroid ~(200, 200)

        masks = [mask_elongated, mask_round]
        trajectory_point = (100.0, 100.0)
        median_ball_size = 20.0

        result = cleaner.identify_ball_mask(masks, trajectory_point, median_ball_size)
        # mask_elongated fails aspect ratio check, so mask_round is selected
        assert result == 1

    def test_rejects_mask_with_bad_size_ratio(self):
        """Closest mask should be rejected if too large compared to median ball size."""
        cleaner = self._make_cleaner()

        # Closest mask is way too big (100x100 vs median ball size of 20)
        mask_big = np.zeros((480, 640), dtype=bool)
        mask_big[50:150, 50:150] = True  # 100x100, centroid ~(100, 100)

        mask_right = np.zeros((480, 640), dtype=bool)
        mask_right[195:205, 195:205] = True  # 10x10, centroid ~(200, 200)

        masks = [mask_big, mask_right]
        trajectory_point = (100.0, 100.0)
        median_ball_size = 20.0

        result = cleaner.identify_ball_mask(masks, trajectory_point, median_ball_size)
        # mask_big fails size ratio check (100 vs 20 = ratio 5.0 > 2.0)
        assert result == 1

    def test_returns_none_when_no_valid_masks(self):
        """Should return None if all masks fail validation."""
        cleaner = self._make_cleaner()

        # Only mask is way too big
        mask_huge = np.zeros((480, 640), dtype=bool)
        mask_huge[0:200, 0:200] = True

        masks = [mask_huge]
        trajectory_point = (100.0, 100.0)
        median_ball_size = 20.0

        result = cleaner.identify_ball_mask(masks, trajectory_point, median_ball_size)
        assert result is None

    def test_empty_masks_returns_none(self):
        """Should return None for empty mask list."""
        cleaner = self._make_cleaner()
        result = cleaner.identify_ball_mask([], (100.0, 100.0), 20.0)
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestBallMaskIdentification -v`
Expected: FAIL — `AttributeError: 'FrameCleaner' object has no attribute 'identify_ball_mask'`

- [ ] **Step 3: Implement identify_ball_mask**

Add to `clpga_demo/cleaner.py` — new config params in `__init__` and the method:

```python
# In __init__, add after existing config reads:
        self._max_aspect_ratio = corridor_config.get("max_aspect_ratio", 2.0)
        self._max_size_ratio = corridor_config.get("max_size_ratio", 2.0)

# New method on FrameCleaner:

    def identify_ball_mask(
        self,
        masks: list[np.ndarray],
        trajectory_point: tuple[float, float],
        median_ball_size: float,
    ) -> int | None:
        """Identify which mask is the ball by proximity to trajectory point.

        Args:
            masks: List of (H, W) boolean masks from SAM3 segment-everything.
            trajectory_point: (x, y) expected ball position from rough trajectory.
            median_ball_size: Median ball diameter for size validation.

        Returns:
            Index of the ball mask, or None if no mask passes validation.
        """
        if not masks:
            return None

        tx, ty = trajectory_point

        # Score each mask: (distance_to_trajectory, index) — only if it passes gates
        candidates: list[tuple[float, int]] = []
        for i, mask in enumerate(masks):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            # Centroid
            cx = float(xs.mean())
            cy = float(ys.mean())

            # Bounding box of mask
            min_x, max_x = float(xs.min()), float(xs.max())
            min_y, max_y = float(ys.min()), float(ys.max())
            w = max_x - min_x + 1
            h = max_y - min_y + 1

            # Aspect ratio gate
            short = min(w, h)
            if short <= 0 or max(w, h) / short > self._max_aspect_ratio:
                continue

            # Size consistency gate
            mask_size = (w + h) / 2
            if median_ball_size > 0:
                ratio = mask_size / median_ball_size
                if ratio > self._max_size_ratio or ratio < 1.0 / self._max_size_ratio:
                    continue

            # Distance to trajectory point
            dist = float(np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2))
            candidates.append((dist, i))

        if not candidates:
            return None

        # Return the closest valid mask
        candidates.sort()
        return candidates[0][1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestBallMaskIdentification -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/cleaner.py tests/test_cleaner.py
git commit -m "feat: add ball mask identification with aspect ratio and size gates"
```

---

### Task 5: FrameCleaner — Quadmask Generation

**Files:**
- Modify: `clpga_demo/cleaner.py`
- Modify: `tests/test_cleaner.py`

- [ ] **Step 1: Write the failing test for generate_quadmask_frame**

```python
# Append to tests/test_cleaner.py
from scipy.ndimage import binary_dilation


class TestQuadmaskGeneration:
    def _make_cleaner(self):
        return FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

    def test_ball_mask_region_is_255(self):
        """Ball mask pixels should be 255 (keep) in the quadmask."""
        cleaner = self._make_cleaner()

        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True

        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[200:220, 200:220] = True

        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)

        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask,
            distractor_masks=[distractor_mask],
            corridor=corridor,
            frame_h=480,
            frame_w=640,
        )

        assert quadmask.shape == (480, 640)
        assert quadmask.dtype == np.uint8
        # Ball region should be 255
        assert np.all(quadmask[90:110, 90:110] == 255)

    def test_distractor_region_is_0(self):
        """Distractor mask pixels (after dilation) should be 0 (remove) in the quadmask."""
        cleaner = self._make_cleaner()

        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True

        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[140:160, 140:160] = True  # Inside corridor

        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)

        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask,
            distractor_masks=[distractor_mask],
            corridor=corridor,
            frame_h=480,
            frame_w=640,
        )

        # Core of distractor should be 0 (remove)
        assert np.all(quadmask[142:158, 142:158] == 0)

    def test_outside_corridor_is_255(self):
        """Everything outside the corridor should be 255 (keep)."""
        cleaner = self._make_cleaner()

        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)

        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=None,
            distractor_masks=[],
            corridor=corridor,
            frame_h=480,
            frame_w=640,
        )

        # Outside corridor should all be 255
        assert np.all(quadmask[0:20, :] == 255)
        assert np.all(quadmask[180:, :] == 255)
        assert np.all(quadmask[:, 0:20] == 255)
        assert np.all(quadmask[:, 180:] == 255)

    def test_no_distractors_returns_all_255(self):
        """If no distractor masks, entire quadmask should be 255."""
        cleaner = self._make_cleaner()

        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True

        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)

        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask,
            distractor_masks=[],
            corridor=corridor,
            frame_h=480,
            frame_w=640,
        )

        assert np.all(quadmask == 255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestQuadmaskGeneration -v`
Expected: FAIL — `AttributeError: 'FrameCleaner' object has no attribute 'generate_quadmask_frame'`

- [ ] **Step 3: Implement generate_quadmask_frame**

Add import at top of `clpga_demo/cleaner.py`:

```python
from scipy.ndimage import binary_dilation
```

Add method to `FrameCleaner`:

```python
    def generate_quadmask_frame(
        self,
        ball_mask: np.ndarray | None,
        distractor_masks: list[np.ndarray],
        corridor: Corridor,
        frame_h: int,
        frame_w: int,
    ) -> np.ndarray:
        """Generate a single-frame quadmask for void-model.

        Args:
            ball_mask: (H, W) boolean mask of the ball (keep), or None if ball not found.
            distractor_masks: List of (H, W) boolean masks to remove.
            corridor: The corridor for this frame.
            frame_h: Frame height.
            frame_w: Frame width.

        Returns:
            (H, W) uint8 quadmask: 0=remove, 127=affected, 255=keep.
        """
        quadmask = np.full((frame_h, frame_w), 255, dtype=np.uint8)

        if not distractor_masks:
            return quadmask

        # Merge distractor masks
        merged = np.zeros((frame_h, frame_w), dtype=bool)
        for m in distractor_masks:
            merged |= m

        # Clip to corridor bounds
        corridor_mask = np.zeros((frame_h, frame_w), dtype=bool)
        corridor_mask[corridor.y1:corridor.y2, corridor.x1:corridor.x2] = True
        merged &= corridor_mask

        # Dilate to avoid artifact halos
        if self._mask_dilation_px > 0:
            merged = binary_dilation(merged, iterations=self._mask_dilation_px)
            merged &= corridor_mask  # Re-clip after dilation

        # Set distractor region to 0 (remove)
        quadmask[merged] = 0

        # Set corridor boundary to 127 (affected) — pixels in corridor but not distractor or ball
        boundary = corridor_mask & ~merged
        if ball_mask is not None:
            boundary &= ~ball_mask
        quadmask[boundary & (quadmask == 255)] = 127

        # Ensure ball region stays 255
        if ball_mask is not None:
            quadmask[ball_mask] = 255

        # Ensure outside corridor is 255
        quadmask[~corridor_mask] = 255

        return quadmask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestQuadmaskGeneration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/cleaner.py tests/test_cleaner.py
git commit -m "feat: add quadmask frame generation with dilation and corridor clipping"
```

---

### Task 6: FrameCleaner — Segment Splitting & Overlap Blending

**Files:**
- Modify: `clpga_demo/cleaner.py`
- Modify: `tests/test_cleaner.py`

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_cleaner.py

class TestSegmentSplitting:
    def test_short_video_single_segment(self):
        """A video shorter than segment_max_frames should produce one segment."""
        segments = FrameCleaner.split_into_segments(
            total_frames=100, max_frames=180, overlap=16,
        )
        assert segments == [(0, 100)]

    def test_exact_max_frames_single_segment(self):
        """A video exactly segment_max_frames should produce one segment."""
        segments = FrameCleaner.split_into_segments(
            total_frames=180, max_frames=180, overlap=16,
        )
        assert segments == [(0, 180)]

    def test_long_video_multiple_segments_with_overlap(self):
        """A longer video should produce overlapping segments."""
        segments = FrameCleaner.split_into_segments(
            total_frames=300, max_frames=180, overlap=16,
        )
        # First segment: 0..180
        assert segments[0] == (0, 180)
        # Second segment should start at 180 - 16 = 164
        assert segments[1][0] == 164
        # Should cover the rest
        assert segments[-1][1] == 300

    def test_segments_cover_all_frames(self):
        """Every frame index should be covered by at least one segment."""
        segments = FrameCleaner.split_into_segments(
            total_frames=500, max_frames=180, overlap=16,
        )
        covered = set()
        for start, end in segments:
            covered.update(range(start, end))
        assert covered == set(range(500))


class TestOverlapBlending:
    def test_blend_two_segments_linear_crossfade(self):
        """Overlapping frames should be linearly blended between segments."""
        seg1 = np.full((20, 4, 4, 3), 100, dtype=np.uint8)  # 20 frames, all pixel value 100
        seg2 = np.full((20, 4, 4, 3), 200, dtype=np.uint8)  # 20 frames, all pixel value 200

        segments = [(0, 20), (12, 32)]  # 8-frame overlap at frames 12..20
        seg_frames = [seg1, seg2]

        result = FrameCleaner.blend_segments(seg_frames, segments, total_frames=32)

        assert result.shape == (32, 4, 4, 3)
        # Before overlap: pure seg1
        assert np.all(result[0] == 100)
        assert np.all(result[11] == 100)
        # After overlap: pure seg2
        assert np.all(result[20] == 200)
        assert np.all(result[31] == 200)
        # Middle of overlap (frame 16 = index 4 of 8 overlap): ~150
        mid_val = result[16, 0, 0, 0]
        assert 140 <= mid_val <= 160

    def test_single_segment_no_blending(self):
        """A single segment should pass through unchanged."""
        seg = np.full((10, 4, 4, 3), 128, dtype=np.uint8)
        result = FrameCleaner.blend_segments([seg], [(0, 10)], total_frames=10)
        assert np.array_equal(result, seg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestSegmentSplitting tests/test_cleaner.py::TestOverlapBlending -v`
Expected: FAIL — `AttributeError: type object 'FrameCleaner' has no attribute 'split_into_segments'`

- [ ] **Step 3: Implement split_into_segments and blend_segments**

Add static methods to `FrameCleaner` in `clpga_demo/cleaner.py`:

```python
    @staticmethod
    def split_into_segments(
        total_frames: int,
        max_frames: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """Split a frame range into overlapping segments for void-model.

        Args:
            total_frames: Total number of frames.
            max_frames: Maximum frames per segment.
            overlap: Number of overlapping frames between consecutive segments.

        Returns:
            List of (start_frame, end_frame) tuples.
        """
        if total_frames <= max_frames:
            return [(0, total_frames)]

        segments: list[tuple[int, int]] = []
        start = 0
        stride = max_frames - overlap

        while start < total_frames:
            end = min(start + max_frames, total_frames)
            segments.append((start, end))
            if end == total_frames:
                break
            start += stride

        return segments

    @staticmethod
    def blend_segments(
        seg_frames: list[np.ndarray],
        segments: list[tuple[int, int]],
        total_frames: int,
    ) -> np.ndarray:
        """Blend overlapping video segments using linear crossfade.

        Args:
            seg_frames: List of (T_i, H, W, 3) uint8 arrays, one per segment.
            segments: List of (start, end) frame ranges corresponding to seg_frames.
            total_frames: Total output frame count.

        Returns:
            (total_frames, H, W, 3) uint8 blended result.
        """
        if len(seg_frames) == 1:
            return seg_frames[0]

        h, w, c = seg_frames[0].shape[1:]
        result = np.zeros((total_frames, h, w, c), dtype=np.float32)
        weights = np.zeros((total_frames, 1, 1, 1), dtype=np.float32)

        for seg_arr, (start, end) in zip(seg_frames, segments):
            seg_len = end - start
            # Create per-frame weights: ramp up at start overlap, ramp down at end overlap
            w_arr = np.ones(seg_len, dtype=np.float32)

            # Find overlap with previous segment
            if start > 0:
                # How many frames overlap with the previous segment's tail
                prev_end = None
                for ps, pe in segments:
                    if pe > start and ps < start:
                        prev_end = pe
                        break
                if prev_end is not None:
                    overlap_len = prev_end - start
                    ramp = np.linspace(0, 1, overlap_len + 2)[1:-1]
                    w_arr[:overlap_len] = ramp

            # Find overlap with next segment
            next_start = None
            for ns, ne in segments:
                if ns > start and ns < end:
                    next_start = ns
                    break
            if next_start is not None:
                overlap_len = end - next_start
                ramp = np.linspace(1, 0, overlap_len + 2)[1:-1]
                w_arr[seg_len - overlap_len:] = ramp

            w_shaped = w_arr.reshape(-1, 1, 1, 1)
            result[start:end] += seg_arr.astype(np.float32) * w_shaped
            weights[start:end] += w_shaped

        # Normalize
        weights = np.maximum(weights, 1e-8)
        result = (result / weights).clip(0, 255).astype(np.uint8)
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestSegmentSplitting tests/test_cleaner.py::TestOverlapBlending -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/cleaner.py tests/test_cleaner.py
git commit -m "feat: add segment splitting and overlap blending for void-model"
```

---

### Task 7: Preset & CLI Configuration

**Files:**
- Modify: `clpga_demo/presets.py`
- Modify: `clpga_demo/__main__.py`
- Modify: `tests/test_presets.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests for presets**

```python
# Append to tests/test_presets.py

class TestCleaningPresets:
    def test_default_preset_has_clean_false(self):
        from clpga_demo.presets import get_preset
        preset = get_preset("default")
        assert preset["clean"] is False

    def test_default_preset_has_corridor_multiplier(self):
        from clpga_demo.presets import get_preset
        preset = get_preset("default")
        assert preset["corridor_multiplier"] == 4.0

    def test_default_preset_has_segment_max_frames(self):
        from clpga_demo.presets import get_preset
        preset = get_preset("default")
        assert preset["segment_max_frames"] == 180

    def test_default_preset_has_clean_prompt(self):
        from clpga_demo.presets import get_preset
        preset = get_preset("default")
        assert preset["clean_prompt"] == "golf course background"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_presets.py::TestCleaningPresets -v`
Expected: FAIL — `KeyError: 'clean'`

- [ ] **Step 3: Add cleaning params to presets**

Edit `clpga_demo/presets.py` — add to the `"default"` dict:

```python
SHOT_PRESETS: dict[str, dict] = {
    "default": {
        "smoothing_sigma_seconds": 0.5,
        "confidence": 0.25,
        "text": ["golf ball"],
        "tracker_type": "momentum",
        "clean": False,
        "corridor_multiplier": 4.0,
        "corridor_speed_scale": 1.5,
        "mask_dilation_px": 5,
        "segment_max_frames": 180,
        "segment_overlap_frames": 16,
        "void_model_dir": None,
        "clean_prompt": "golf course background",
    },
    "putt": {
        "smoothing_sigma_seconds": 0.1,
        "confidence": 0.15,
        "text": ["golf ball on green"],
        "tracker_type": "momentum",
        "momentum_history_size": 5,
        "momentum_radius_scale": 15.0,
        "momentum_confirm_frames": 1,
        "momentum_max_size_ratio": 2.0,
        "momentum_max_aspect_ratio": 6.0,
        "kalman_process_noise": 0.5,
        "kalman_measurement_noise": 1.0,
        "kalman_gate_threshold": 9.0,
        "clean": False,
        "corridor_multiplier": 3.0,
        "corridor_speed_scale": 1.5,
        "mask_dilation_px": 5,
        "segment_max_frames": 180,
        "segment_overlap_frames": 16,
        "void_model_dir": None,
        "clean_prompt": "golf putting green background",
    },
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_presets.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing tests for CLI flags**

```python
# Append to tests/test_cli.py

class TestCleanCLI:
    def test_clean_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--clean"])
        assert args.clean is True

    def test_clean_flag_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.clean is False

    def test_corridor_multiplier_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--corridor-multiplier", "5.0"])
        assert args.corridor_multiplier == 5.0

    def test_mask_dilation_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--mask-dilation", "8"])
        assert args.mask_dilation == 8

    def test_void_model_dir_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--void-model-dir", "/models/void"])
        assert args.void_model_dir == "/models/void"

    def test_clean_prompt_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--clean-prompt", "driving range"])
        assert args.clean_prompt == "driving range"

    def test_clean_flag_resolves_to_preset(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--clean"])
        resolved = resolve_args(args)
        assert resolved["clean"] is True

    def test_corridor_multiplier_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--corridor-multiplier", "6.0"])
        resolved = resolve_args(args)
        assert resolved["corridor_multiplier"] == 6.0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cli.py::TestCleanCLI -v`
Expected: FAIL — `error: unrecognized arguments: --clean`

- [ ] **Step 7: Add CLI flags to __main__.py**

Edit `clpga_demo/__main__.py` — add to `build_parser()` before `return parser`:

```python
    parser.add_argument("--clean", action="store_true", default=False, help="Enable distractor removal cleaning pass")
    parser.add_argument("--corridor-multiplier", type=float, default=None, help="Corridor width multiplier (ball sizes)")
    parser.add_argument("--mask-dilation", type=int, default=None, help="Mask dilation in pixels")
    parser.add_argument("--void-model-dir", type=str, default=None, help="Path to pre-downloaded void-model")
    parser.add_argument("--clean-prompt", type=str, default=None, help="Scene description for void-model inpainting")
```

Edit `resolve_args()` — add to `cli_to_preset` dict:

```python
        "clean": "clean",
        "corridor_multiplier": "corridor_multiplier",
        "mask_dilation": "mask_dilation_px",
        "void_model_dir": "void_model_dir",
        "clean_prompt": "clean_prompt",
```

And update the override loop to handle the `--clean` boolean flag (it's `True`/`False`, not `None`):

```python
    for cli_key, preset_key in cli_to_preset.items():
        cli_val = getattr(args, cli_key, None)
        if cli_key == "clean":
            # Boolean flag — True overrides preset, False does not
            if cli_val:
                preset[preset_key] = True
        elif cli_val is not None:
            preset[preset_key] = cli_val
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cli.py tests/test_presets.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add clpga_demo/presets.py clpga_demo/__main__.py tests/test_presets.py tests/test_cli.py
git commit -m "feat: add cleaning preset params and CLI flags"
```

---

### Task 8: Pipeline Integration — Pass 1 Data Collection

**Files:**
- Modify: `clpga_demo/pipeline.py`
- Modify: `tests/test_pipeline.py`

Pass 1 currently collects `positions` but not `ball_sizes` or `speeds`, which are needed by the cleaner. We need to extract these from the tracker during Pass 1.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_pipeline.py

class TestCleaningPipeline:
    def test_clean_flag_invokes_cleaning_pass(self, tmp_path):
        """When clean=True, process_video should call FrameCleaner."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path, frames=20)

        cleaner_called = {"called": False}

        def mock_clean_and_retrack(cleaner, *args, **kwargs):
            cleaner_called["called"] = True
            # Return the same rough trajectory as refined
            return args[0]  # rough_positions

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            with patch("clpga_demo.pipeline.FrameCleaner") as MockCleaner:
                mock_instance = MockCleaner.return_value
                mock_instance.compute_corridors.return_value = [None] * 20
                mock_instance.generate_quadmasks.return_value = np.zeros((20, 240, 320), dtype=np.uint8)
                mock_instance.clean_segments.return_value = [np.zeros((20, 240, 320, 3), dtype=np.uint8)]
                mock_instance.blend_segments = FrameCleaner.blend_segments
                with patch("clpga_demo.pipeline.VoidModelWrapper") as MockVoid:
                    mock_void = MockVoid.return_value
                    mock_void.download_if_needed.return_value = "/fake"
                    process_video(input_path, output_path, clean=True)

        assert MockCleaner.called

    def test_clean_false_skips_cleaning(self, tmp_path):
        """When clean=False (default), cleaning pass should be skipped."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_video(input_path, output_path, clean=False)

        assert Path(output_path).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_pipeline.py::TestCleaningPipeline -v`
Expected: FAIL — `TypeError: process_video() got an unexpected keyword argument 'clean'`

- [ ] **Step 3: Modify process_video to accept cleaning params and collect Pass 1 metadata**

Edit `clpga_demo/pipeline.py` — add new parameters to `process_video()` signature:

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
    clean: bool = False,
    corridor_multiplier: float = 4.0,
    corridor_speed_scale: float = 1.5,
    mask_dilation_px: int = 5,
    segment_max_frames: int = 180,
    segment_overlap_frames: int = 16,
    void_model_dir: str | None = None,
    clean_prompt: str = "golf course background",
) -> None:
```

Add imports at top:

```python
from clpga_demo.cleaner import FrameCleaner
from clpga_demo.void_model import VoidModelWrapper
```

Collect `ball_sizes` and `speeds` during Pass 1 — after the tracking loop, add:

```python
    # Collect metadata for cleaning pass
    ball_sizes: list[float] = []
    speeds_list: list[float] = []
```

Inside the tracking loop, after `pos = tracker.update(...)`:

```python
            w = result.bbox[2] - result.bbox[0]
            h_box = result.bbox[3] - result.bbox[1]
            ball_sizes.append((w + h_box) / 2)
            speeds_list.append(tracker.speed)
```

In the else branch (no detection):

```python
            ball_sizes.append(ball_sizes[-1] if ball_sizes else 0.0)
            speeds_list.append(tracker.speed)
```

After Pass 1, before smoothing, add Pass 2:

```python
    # --- Pass 2 (optional): Clean distractors and re-track ---
    if clean:
        void_wrapper = VoidModelWrapper(model_dir=void_model_dir)
        void_wrapper.download_if_needed()

        corridor_config = {
            "corridor_multiplier": corridor_multiplier,
            "corridor_speed_scale": corridor_speed_scale,
            "radius_scale": momentum_radius_scale,
            "mask_dilation_px": mask_dilation_px,
            "max_aspect_ratio": momentum_max_aspect_ratio,
            "max_size_ratio": momentum_max_size_ratio,
        }
        cleaner = FrameCleaner(sam3_model=model, void_model=void_wrapper, corridor_config=corridor_config)
        corridors = cleaner.compute_corridors(positions, ball_sizes, speeds_list, frame_w=src_w, frame_h=src_h)

        # Generate quadmasks and run void-model
        segments = FrameCleaner.split_into_segments(frame_count, segment_max_frames, segment_overlap_frames)
        quadmasks = cleaner.generate_quadmasks(source, corridors, positions)
        cleaned_segments = cleaner.clean_segments(source, quadmasks, segments, clean_prompt)
        cleaned_video = FrameCleaner.blend_segments(cleaned_segments, segments, frame_count)

        # Re-track on cleaned frames
        tracker2 = create_tracker(
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
        # Re-run detection on cleaned frames and update positions
        positions = _retrack_cleaned(cleaned_video, model, confidence, text, tracker2, src_w, src_h)

        if all(p is None for p in positions):
            logger.warning("Cleaning pass produced no detections; falling back to rough trajectory")
            # positions stays as the rough trajectory from Pass 1
```

Add a helper function in `pipeline.py`:

```python
def _retrack_cleaned(
    cleaned_frames: np.ndarray,
    model: str,
    confidence: float,
    text: list[str] | None,
    tracker: BallTracker,
    src_w: int,
    src_h: int,
) -> list[tuple[float, float] | None]:
    """Re-run detection and tracking on cleaned frames."""
    from clpga_demo.tracker import select_ball, track_video

    positions: list[tuple[float, float] | None] = []
    selected_obj_id: int | None = None
    frames_since_lost = 0
    fps = 30.0  # Cleaned frames don't have native fps; use default

    for frame_idx in range(len(cleaned_frames)):
        frame = cleaned_frames[frame_idx]
        # Run SAM3 detection on this single cleaned frame
        # This uses the model's single-frame inference
        from ultralytics import SAM3
        sam = SAM3(model)
        results = sam(frame, conf=confidence, texts=text)
        boxes = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None and len(results[0].boxes) > 0 else np.empty((0, 7))

        result = select_ball(boxes, src_w, src_h, preferred_obj_id=selected_obj_id, frame_idx=frame_idx)

        accepted = False
        if result is not None:
            if frames_since_lost > 0 and tracker.has_position:
                if not tracker.accept((result.center_x, result.center_y), result.bbox):
                    result = None
                else:
                    accepted = True
            else:
                accepted = True

        if accepted and result is not None:
            if selected_obj_id is None:
                selected_obj_id = result.obj_id
            pos = tracker.update((result.center_x, result.center_y), result.bbox)
            positions.append(pos)
            frames_since_lost = 0
        else:
            if tracker.has_position:
                pos = tracker.predict()
                positions.append(pos)
            else:
                positions.append(None)
            frames_since_lost += 1
            if frames_since_lost > fps * 3:
                selected_obj_id = None
                tracker.reset()
                frames_since_lost = 0

    return positions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire cleaning Pass 2 into pipeline with ball size and speed collection"
```

---

### Task 9: FrameCleaner — Full generate_quadmasks and clean_segments Orchestration

**Files:**
- Modify: `clpga_demo/cleaner.py`
- Modify: `tests/test_cleaner.py`

This task implements the high-level orchestration methods that read the video, call SAM3 segment-everything per corridor, identify ball masks, generate quadmasks, and run void-model.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_cleaner.py
from unittest.mock import patch, MagicMock


class TestCleanSegments:
    def test_clean_segments_calls_void_model(self):
        """clean_segments should invoke void_model.inpaint for each segment."""
        mock_void = MagicMock()
        mock_void.inpaint.return_value = np.zeros((10, 384, 672, 3), dtype=np.uint8)

        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=mock_void,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmasks = np.full((10, 384, 672), 255, dtype=np.uint8)
        segments = [(0, 10)]

        result = cleaner.clean_segments(video_frames, quadmasks, segments, "golf course")

        assert len(result) == 1
        mock_void.inpaint.assert_called_once()

    def test_clean_segments_skips_all_255_quadmask(self):
        """If a segment's quadmask is all 255 (nothing to remove), skip void-model."""
        mock_void = MagicMock()

        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=mock_void,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmasks = np.full((10, 384, 672), 255, dtype=np.uint8)  # Nothing to remove
        segments = [(0, 10)]

        result = cleaner.clean_segments(video_frames, quadmasks, segments, "golf course")

        # Should return raw frames without calling void-model
        mock_void.inpaint.assert_not_called()
        assert np.array_equal(result[0], video_frames)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestCleanSegments -v`
Expected: FAIL — `TypeError` — `clean_segments` method signature doesn't match yet

- [ ] **Step 3: Implement clean_segments**

Add method to `FrameCleaner` in `clpga_demo/cleaner.py`:

```python
    def clean_segments(
        self,
        video_frames: np.ndarray,
        quadmasks: np.ndarray,
        segments: list[tuple[int, int]],
        prompt: str,
    ) -> list[np.ndarray]:
        """Run void-model inpainting on each video segment.

        Args:
            video_frames: (T, H, W, 3) uint8 full video frames.
            quadmasks: (T, H, W) uint8 quadmask for each frame.
            segments: List of (start, end) frame ranges.
            prompt: Background description for void-model.

        Returns:
            List of (seg_len, H, W, 3) uint8 cleaned segment arrays.
        """
        cleaned: list[np.ndarray] = []

        for start, end in segments:
            seg_video = video_frames[start:end]
            seg_mask = quadmasks[start:end]

            # Skip void-model if nothing to remove in this segment
            if np.all(seg_mask == 255):
                logger.debug("Segment %d-%d: no distractors, skipping inpainting", start, end)
                cleaned.append(seg_video.copy())
                continue

            logger.info("Cleaning segment frames %d-%d with void-model", start, end)
            result = self._void_model.inpaint(seg_video, seg_mask, prompt)
            cleaned.append(result)

        return cleaned
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestCleanSegments -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/cleaner.py tests/test_cleaner.py
git commit -m "feat: add clean_segments orchestration with skip-if-clean optimization"
```

---

### Task 10: FrameCleaner — generate_quadmasks (SAM3 Segment-Everything Integration)

**Files:**
- Modify: `clpga_demo/cleaner.py`
- Modify: `tests/test_cleaner.py`

This is the method that reads video frames, runs SAM3 segment-everything within each corridor, identifies the ball mask, and builds the full quadmask array.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_cleaner.py

class TestGenerateQuadmasks:
    def test_generates_quadmask_array(self):
        """generate_quadmasks should return (T, H, W) uint8 array."""
        mock_sam = MagicMock()
        # SAM3 returns a results object with masks
        mock_result = MagicMock()
        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[95:105, 95:105] = True
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[140:160, 140:160] = True
        mock_result.masks.data = MagicMock()
        mock_result.masks.data.cpu.return_value.numpy.return_value = np.stack([ball_mask, distractor_mask])
        mock_sam.return_value = [mock_result]

        cleaner = FrameCleaner(
            sam3_model=mock_sam,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((3, 480, 640, 3), dtype=np.uint8)
        corridors = [
            Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180),
            Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180),
            None,  # Ball was lost on frame 3
        ]
        trajectory = [(100.0, 100.0), (100.0, 100.0), None]
        median_ball_size = 10.0

        quadmasks = cleaner.generate_quadmasks(video_frames, corridors, trajectory, median_ball_size)

        assert quadmasks.shape == (3, 480, 640)
        assert quadmasks.dtype == np.uint8
        # Frame 3 (no corridor) should be all 255
        assert np.all(quadmasks[2] == 255)

    def test_no_corridor_produces_all_255(self):
        """Frames with no corridor should have all-255 quadmask."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((2, 100, 100, 3), dtype=np.uint8)
        corridors = [None, None]
        trajectory = [None, None]

        quadmasks = cleaner.generate_quadmasks(video_frames, corridors, trajectory, 10.0)

        assert np.all(quadmasks == 255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestGenerateQuadmasks -v`
Expected: FAIL — `TypeError` — `generate_quadmasks` signature doesn't match or doesn't exist

- [ ] **Step 3: Implement generate_quadmasks**

Add method to `FrameCleaner` in `clpga_demo/cleaner.py`:

```python
    def generate_quadmasks(
        self,
        video_frames: np.ndarray,
        corridors: list[Corridor | None],
        trajectory: list[tuple[float, float] | None],
        median_ball_size: float,
    ) -> np.ndarray:
        """Generate quadmask array for all frames using SAM3 segment-everything.

        Args:
            video_frames: (T, H, W, 3) uint8 BGR frames.
            corridors: Per-frame Corridor or None.
            trajectory: Per-frame (x, y) or None.
            median_ball_size: Median ball size for mask validation.

        Returns:
            (T, H, W) uint8 quadmask array.
        """
        num_frames, frame_h, frame_w = video_frames.shape[:3]
        quadmasks = np.full((num_frames, frame_h, frame_w), 255, dtype=np.uint8)

        for i in range(num_frames):
            corridor = corridors[i]
            traj_point = trajectory[i]

            if corridor is None or traj_point is None:
                continue

            # Run SAM3 segment-everything on the corridor region
            frame = video_frames[i]
            crop = frame[corridor.y1:corridor.y2, corridor.x1:corridor.x2]

            results = self._sam3_model(crop)
            if not results or results[0].masks is None or len(results[0].masks) == 0:
                continue

            # Get masks and map back to full frame coordinates
            crop_masks = results[0].masks.data.cpu().numpy().astype(bool)
            full_masks: list[np.ndarray] = []
            for m in crop_masks:
                full_mask = np.zeros((frame_h, frame_w), dtype=bool)
                # Resize crop mask to corridor dimensions if needed
                ch = corridor.y2 - corridor.y1
                cw = corridor.x2 - corridor.x1
                if m.shape != (ch, cw):
                    import cv2
                    m_resized = cv2.resize(m.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    m_resized = m
                full_mask[corridor.y1:corridor.y2, corridor.x1:corridor.x2] = m_resized
                full_masks.append(full_mask)

            # Identify ball mask
            ball_idx = self.identify_ball_mask(full_masks, traj_point, median_ball_size)
            ball_mask = full_masks[ball_idx] if ball_idx is not None else None

            # Distractor masks = everything except ball
            distractor_masks = [m for j, m in enumerate(full_masks) if j != ball_idx]

            quadmasks[i] = self.generate_quadmask_frame(
                ball_mask=ball_mask,
                distractor_masks=distractor_masks,
                corridor=corridor,
                frame_h=frame_h,
                frame_w=frame_w,
            )

        return quadmasks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cleaner.py::TestGenerateQuadmasks -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/cleaner.py tests/test_cleaner.py
git commit -m "feat: add generate_quadmasks with SAM3 segment-everything integration"
```

---

### Task 11: Wire __main__.py to Pass Cleaning Params to process_video

**Files:**
- Modify: `clpga_demo/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_cli.py

class TestCleanPassthrough:
    def test_clean_params_passed_to_process_video(self, tmp_path):
        """All cleaning params should be forwarded from CLI to process_video."""
        from unittest.mock import patch, MagicMock
        import sys

        with patch("clpga_demo.pipeline.process_video") as mock_pv:
            with patch("sys.argv", ["clpga-demo", "in.mp4", "-o", "out.mp4", "--clean", "--corridor-multiplier", "5.0"]):
                from clpga_demo.__main__ import main
                try:
                    main()
                except (SystemExit, Exception):
                    pass

            if mock_pv.called:
                _, kwargs = mock_pv.call_args
                assert kwargs.get("clean") is True or mock_pv.call_args[1].get("clean") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cli.py::TestCleanPassthrough -v`
Expected: FAIL — `process_video()` not being called with `clean` param

- [ ] **Step 3: Update main() to pass cleaning params**

Edit `clpga_demo/__main__.py` — update the `process_video()` call in `main()`:

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
            momentum_confirm_frames=resolved.get("momentum_confirm_frames", 3),
            momentum_max_size_ratio=resolved.get("momentum_max_size_ratio", 2.0),
            momentum_max_aspect_ratio=resolved.get("momentum_max_aspect_ratio", 2.0),
            kalman_process_noise=resolved.get("kalman_process_noise", 1.0),
            kalman_measurement_noise=resolved.get("kalman_measurement_noise", 1.0),
            kalman_gate_threshold=resolved.get("kalman_gate_threshold", 9.0),
            clean=resolved.get("clean", False),
            corridor_multiplier=resolved.get("corridor_multiplier", 4.0),
            corridor_speed_scale=resolved.get("corridor_speed_scale", 1.5),
            mask_dilation_px=resolved.get("mask_dilation_px", 5),
            segment_max_frames=resolved.get("segment_max_frames", 180),
            segment_overlap_frames=resolved.get("segment_overlap_frames", 16),
            void_model_dir=resolved.get("void_model_dir"),
            clean_prompt=resolved.get("clean_prompt", "golf course background"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add clpga_demo/__main__.py tests/test_cli.py
git commit -m "feat: wire cleaning params through CLI to process_video"
```

---

### Task 12: Add void-model Dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Verify current dependencies don't already include what we need**

The `huggingface-hub` is already present. `scipy` is present (for `binary_dilation`). We need to ensure the void-model's CogVideoX-Fun dependencies are documented.

- [ ] **Step 2: Add void-model as an optional dependency group**

Edit `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
]
clean = [
    "huggingface-hub[cli]>=1.7.1",
]
```

No new core dependencies are needed — void-model is invoked via subprocess from its own installed environment. The `huggingface-hub` dependency (for downloading) is already in core deps.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "docs: note cleaning feature dependency on void-model (subprocess)"
```

---

### Task 13: Run Full Test Suite & Verify

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run linter if configured**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/ -v --tb=short`
Expected: No failures

- [ ] **Step 3: Verify the existing pipeline still works without --clean**

Run: `cd /home/stark/Documents/clpga_demo && python -m pytest tests/test_pipeline.py -v`
Expected: All existing tests still pass — no regressions.
