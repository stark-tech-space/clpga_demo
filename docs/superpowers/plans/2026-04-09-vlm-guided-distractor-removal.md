# VLM-Guided Distractor Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace segment-everything with Gemini Flash scene analysis + targeted SAM3 box prompts, so the cleaning pass removes only identified distractors and never the golf ball.

**Architecture:** New `scene_analyzer.py` sends keyframes to Gemini Flash, gets back ball bbox + distractor list. `cleaner.py` replaces its `generate_quadmasks` to use targeted SAM3 box prompts per distractor with a ball protection zone. Pipeline wires the analyzer in. Passes 1 and 3 unchanged.

**Tech Stack:** google-genai (Gemini API), ultralytics SAM3 (box prompts), existing void-model pipeline

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `clpga_demo/scene_analyzer.py` | Create | `SceneAnalysis` dataclass, `SceneAnalyzer` class with Gemini Flash integration |
| `clpga_demo/cleaner.py` | Modify | Replace `generate_quadmasks` with `generate_quadmasks_targeted`, remove old segment-everything methods, add ball protection zone |
| `clpga_demo/pipeline.py` | Modify | Create `SceneAnalyzer`, pass to cleaner, use per-segment scene descriptions as void-model prompts |
| `clpga_demo/presets.py` | Modify | Add `gemini_model` parameter |
| `clpga_demo/__main__.py` | Modify | Add `--gemini-model` CLI flag, wire through to `process_video` |
| `tests/test_scene_analyzer.py` | Create | Tests for SceneAnalyzer |
| `tests/test_cleaner.py` | Modify | Replace segment-everything tests with targeted tests |
| `tests/test_cli.py` | Modify | Add `--gemini-model` test |
| `tests/test_presets.py` | Modify | Add `gemini_model` preset test |
| `tests/test_pipeline.py` | Modify | Update cleaning pipeline mock test |

---

### Task 1: SceneAnalyzer — SceneAnalysis Dataclass & Interface

**Files:**
- Create: `clpga_demo/scene_analyzer.py`
- Create: `tests/test_scene_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scene_analyzer.py
from clpga_demo.scene_analyzer import SceneAnalysis, SceneAnalyzer


class TestSceneAnalysis:
    def test_dataclass_creation(self):
        analysis = SceneAnalysis(
            ball_bbox=(100, 200, 120, 220),
            distractors=[{"label": "flagpole", "bbox": (500, 100, 520, 600)}],
            scene_description="golf putting green with flagpole",
        )
        assert analysis.ball_bbox == (100, 200, 120, 220)
        assert len(analysis.distractors) == 1
        assert analysis.distractors[0]["label"] == "flagpole"
        assert analysis.scene_description == "golf putting green with flagpole"

    def test_empty_analysis(self):
        analysis = SceneAnalysis(
            ball_bbox=None,
            distractors=[],
            scene_description="golf course background",
        )
        assert analysis.ball_bbox is None
        assert analysis.distractors == []

    def test_frozen(self):
        import pytest
        analysis = SceneAnalysis(
            ball_bbox=(100, 200, 120, 220),
            distractors=[],
            scene_description="green",
        )
        with pytest.raises(AttributeError):
            analysis.ball_bbox = (0, 0, 0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scene_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clpga_demo.scene_analyzer'`

- [ ] **Step 3: Implement SceneAnalysis and SceneAnalyzer stub**

```python
# clpga_demo/scene_analyzer.py
"""VLM-based scene analysis for distractor identification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"

_SCENE_PROMPT = """\
You are analyzing a golf putting scene. Identify:
1. The golf ball being putted — provide its bounding box [x1, y1, x2, y2] in pixel coordinates.
2. Any objects that could occlude or be confused with the ball during its path:
   other golf balls, flagpoles/pins, people, club heads, shadows of people,
   tee markers, range markers, etc.
   For each, provide a label and bounding box [x1, y1, x2, y2].
3. A brief description of the background scene (e.g., "golf putting green with
   flagpole and gallery in background").

Only include objects actually visible in the frame.
Return JSON with keys: ball, distractors, scene_description.
Example: {"ball": [540, 620, 560, 640], "distractors": [{"label": "flagpole", "bbox": [800, 100, 820, 700]}], "scene_description": "putting green with flagpole"}
"""


@dataclass(frozen=True)
class SceneAnalysis:
    """Result of VLM scene analysis for a single keyframe."""

    ball_bbox: tuple[int, int, int, int] | None
    distractors: list[dict]
    scene_description: str


class SceneAnalyzer:
    """Analyzes video frames using a VLM to identify the ball and distractors."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._client = None

    def analyze_frame(self, frame: np.ndarray) -> SceneAnalysis:
        """Analyze a single frame and return scene understanding.

        Args:
            frame: (H, W, 3) uint8 BGR frame.

        Returns:
            SceneAnalysis with ball bbox, distractor list, and scene description.
        """
        raise NotImplementedError("Will be implemented in Task 2")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scene_analyzer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/scene_analyzer.py tests/test_scene_analyzer.py
git commit -m "feat: add SceneAnalysis dataclass and SceneAnalyzer stub"
```

---

### Task 2: SceneAnalyzer — Gemini API Integration

**Files:**
- Modify: `clpga_demo/scene_analyzer.py`
- Modify: `tests/test_scene_analyzer.py`

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_scene_analyzer.py
from unittest.mock import patch, MagicMock
import numpy as np


class TestAnalyzeFrame:
    def test_returns_scene_analysis_from_gemini(self):
        """analyze_frame should parse Gemini JSON response into SceneAnalysis."""
        analyzer = SceneAnalyzer(model="gemini-2.5-flash-preview-05-20")

        mock_response = MagicMock()
        mock_response.text = '{"ball": [100, 200, 120, 220], "distractors": [{"label": "flagpole", "bbox": [500, 100, 520, 600]}], "scene_description": "putting green with flagpole"}'

        with patch("clpga_demo.scene_analyzer.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = analyzer.analyze_frame(frame)

        assert result.ball_bbox == (100, 200, 120, 220)
        assert len(result.distractors) == 1
        assert result.distractors[0]["label"] == "flagpole"
        assert result.scene_description == "putting green with flagpole"

    def test_returns_empty_on_api_failure(self):
        """analyze_frame should return empty SceneAnalysis on API error."""
        analyzer = SceneAnalyzer(model="gemini-2.5-flash-preview-05-20")

        with patch("clpga_demo.scene_analyzer.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.side_effect = Exception("API timeout")

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = analyzer.analyze_frame(frame)

        assert result.ball_bbox is None
        assert result.distractors == []
        assert result.scene_description == "golf course background"

    def test_returns_empty_on_invalid_json(self):
        """analyze_frame should handle malformed JSON gracefully."""
        analyzer = SceneAnalyzer(model="gemini-2.5-flash-preview-05-20")

        mock_response = MagicMock()
        mock_response.text = "not valid json {{"

        with patch("clpga_demo.scene_analyzer.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = analyzer.analyze_frame(frame)

        assert result.ball_bbox is None
        assert result.distractors == []

    def test_handles_no_ball_in_response(self):
        """analyze_frame should handle response with no ball detected."""
        analyzer = SceneAnalyzer(model="gemini-2.5-flash-preview-05-20")

        mock_response = MagicMock()
        mock_response.text = '{"ball": null, "distractors": [{"label": "person", "bbox": [200, 100, 400, 600]}], "scene_description": "green with person"}'

        with patch("clpga_demo.scene_analyzer.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = analyzer.analyze_frame(frame)

        assert result.ball_bbox is None
        assert len(result.distractors) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scene_analyzer.py::TestAnalyzeFrame -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement analyze_frame with Gemini API**

Replace the `analyze_frame` method and add imports in `clpga_demo/scene_analyzer.py`:

```python
# Add at top of file:
import cv2
from google import genai

_EMPTY_ANALYSIS = SceneAnalysis(
    ball_bbox=None,
    distractors=[],
    scene_description="golf course background",
)

# Replace analyze_frame in SceneAnalyzer:

    def _get_client(self):
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def analyze_frame(self, frame: np.ndarray) -> SceneAnalysis:
        """Analyze a single frame and return scene understanding.

        Args:
            frame: (H, W, 3) uint8 BGR frame.

        Returns:
            SceneAnalysis with ball bbox, distractor list, and scene description.
            Returns empty analysis on any failure (fail-safe).
        """
        try:
            # Encode frame as JPEG
            _, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            client = self._get_client()
            response = client.models.generate_content(
                model=self._model,
                contents=[
                    {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_bytes.tobytes()}},
                    _SCENE_PROMPT,
                ],
            )

            return self._parse_response(response.text)

        except Exception:
            logger.warning("Scene analysis failed, skipping cleaning for this segment", exc_info=True)
            return _EMPTY_ANALYSIS

    def _parse_response(self, text: str) -> SceneAnalysis:
        """Parse Gemini JSON response into SceneAnalysis."""
        try:
            # Strip markdown code fences if present
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]

            data = json.loads(cleaned)

            ball_bbox = None
            if data.get("ball") is not None:
                b = data["ball"]
                ball_bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

            distractors = []
            for d in data.get("distractors", []):
                bbox = d.get("bbox", [])
                if len(bbox) == 4:
                    distractors.append({
                        "label": str(d.get("label", "unknown")),
                        "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    })

            scene_desc = str(data.get("scene_description", "golf course background"))

            return SceneAnalysis(
                ball_bbox=ball_bbox,
                distractors=distractors,
                scene_description=scene_desc,
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("Failed to parse scene analysis response: %s", text[:200])
            return _EMPTY_ANALYSIS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scene_analyzer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/scene_analyzer.py tests/test_scene_analyzer.py
git commit -m "feat: implement Gemini API scene analysis with fail-safe error handling"
```

---

### Task 3: FrameCleaner — Targeted Quadmask Generation

**Files:**
- Modify: `clpga_demo/cleaner.py`
- Modify: `tests/test_cleaner.py`

This replaces `generate_quadmasks` (segment-everything) with `generate_quadmasks_targeted` (VLM-guided box prompts).

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_cleaner.py
from clpga_demo.scene_analyzer import SceneAnalysis


class TestTargetedQuadmaskGeneration:
    def test_no_distractors_returns_all_255(self):
        """If Gemini found no distractors, all frames should be 255."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((5, 480, 640, 3), dtype=np.uint8)
        corridors = [Corridor(100, 100, 80, 20, 20, 180, 180)] * 5
        analysis = SceneAnalysis(
            ball_bbox=(95, 95, 105, 105),
            distractors=[],
            scene_description="clean green",
        )

        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)

        assert quadmasks.shape == (5, 480, 640)
        assert np.all(quadmasks == 255)

    def test_distractor_region_marked_for_removal(self):
        """Distractor mask from SAM3 should be marked as 0 in quadmask."""
        mock_sam = MagicMock()
        # SAM3 returns a mask for the distractor bbox prompt
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[140:200, 500:530] = True  # Flagpole-shaped region
        mock_result = MagicMock()
        mock_result.masks.data.cpu.return_value.numpy.return_value = np.array([distractor_mask])
        mock_sam.return_value = [mock_result]

        cleaner = FrameCleaner(
            sam3_model=mock_sam,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 0,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        corridors = [Corridor(300, 300, 250, 50, 50, 550, 550)]
        analysis = SceneAnalysis(
            ball_bbox=(290, 290, 310, 310),
            distractors=[{"label": "flagpole", "bbox": (490, 130, 540, 210)}],
            scene_description="green with flagpole",
        )

        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)

        # Distractor region should be 0
        assert np.any(quadmasks[0, 140:200, 500:530] == 0)

    def test_ball_protection_zone_stays_255(self):
        """Ball bbox region should remain 255 even if distractor overlaps."""
        mock_sam = MagicMock()
        # Distractor mask overlaps ball area
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[90:120, 90:120] = True  # Overlaps ball at (95,95)-(105,105)
        mock_result = MagicMock()
        mock_result.masks.data.cpu.return_value.numpy.return_value = np.array([distractor_mask])
        mock_sam.return_value = [mock_result]

        cleaner = FrameCleaner(
            sam3_model=mock_sam,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 0,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        corridors = [Corridor(100, 100, 80, 20, 20, 180, 180)]
        analysis = SceneAnalysis(
            ball_bbox=(95, 95, 105, 105),
            distractors=[{"label": "other ball", "bbox": (85, 85, 125, 125)}],
            scene_description="green",
        )
        median_ball_size = 10.0

        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, median_ball_size)

        # Ball protection zone (expanded by 2x ball size = 20px each direction)
        # Ball bbox (95,95)-(105,105) expanded → (75,75)-(125,125)
        # Center of ball area should be 255
        assert quadmasks[0, 100, 100] == 255

    def test_sam3_no_mask_skips_distractor(self):
        """If SAM3 returns no mask for a distractor, skip it gracefully."""
        mock_sam = MagicMock()
        mock_result = MagicMock()
        mock_result.masks = None  # No mask returned
        mock_sam.return_value = [mock_result]

        cleaner = FrameCleaner(
            sam3_model=mock_sam,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        corridors = [Corridor(100, 100, 80, 20, 20, 180, 180)]
        analysis = SceneAnalysis(
            ball_bbox=(95, 95, 105, 105),
            distractors=[{"label": "person", "bbox": (300, 100, 500, 600)}],
            scene_description="green",
        )

        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)

        # No distractor mask → all 255
        assert np.all(quadmasks == 255)

    def test_none_corridor_produces_all_255(self):
        """Frames with no corridor should have all-255 quadmask."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((2, 100, 100, 3), dtype=np.uint8)
        corridors = [None, None]
        analysis = SceneAnalysis(
            ball_bbox=None,
            distractors=[{"label": "person", "bbox": (10, 10, 50, 50)}],
            scene_description="green",
        )

        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)

        assert np.all(quadmasks == 255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cleaner.py::TestTargetedQuadmaskGeneration -v`
Expected: FAIL — `AttributeError: 'FrameCleaner' object has no attribute 'generate_quadmasks_targeted'`

- [ ] **Step 3: Implement generate_quadmasks_targeted**

Add this method to `FrameCleaner` in `clpga_demo/cleaner.py`:

```python
    def generate_quadmasks_targeted(
        self,
        video_frames: np.ndarray,
        corridors: list[Corridor | None],
        scene_analysis: SceneAnalysis,
        median_ball_size: float,
    ) -> np.ndarray:
        """Generate quadmasks using VLM-guided targeted SAM3 box prompts.

        Args:
            video_frames: (T, H, W, 3) uint8 BGR frames.
            corridors: Per-frame Corridor or None.
            scene_analysis: VLM analysis with ball bbox and distractor list.
            median_ball_size: Median ball diameter for protection zone sizing.

        Returns:
            (T, H, W) uint8 quadmask array.
        """
        num_frames, frame_h, frame_w = video_frames.shape[:3]
        quadmasks = np.full((num_frames, frame_h, frame_w), 255, dtype=np.uint8)

        if not scene_analysis.distractors:
            return quadmasks

        # Compute ball protection zone (expanded bbox)
        ball_protect = None
        if scene_analysis.ball_bbox is not None:
            bx1, by1, bx2, by2 = scene_analysis.ball_bbox
            expand = int(2 * median_ball_size)
            ball_protect = (
                max(0, bx1 - expand),
                max(0, by1 - expand),
                min(frame_w, bx2 + expand),
                min(frame_h, by2 + expand),
            )

        sam = self._get_sam3()

        for i in range(num_frames):
            corridor = corridors[i]
            if corridor is None:
                continue

            frame = video_frames[i]
            distractor_merged = np.zeros((frame_h, frame_w), dtype=bool)

            for dist in scene_analysis.distractors:
                dx1, dy1, dx2, dy2 = dist["bbox"]
                # Expand bbox by 20% for movement tolerance
                dw = dx2 - dx1
                dh = dy2 - dy1
                ex = int(dw * 0.2)
                ey = int(dh * 0.2)
                bx1_e = max(0, dx1 - ex)
                by1_e = max(0, dy1 - ey)
                bx2_e = min(frame_w, dx2 + ex)
                by2_e = min(frame_h, dy2 + ey)

                # SAM3 box prompt
                results = sam(frame, bboxes=[[bx1_e, by1_e, bx2_e, by2_e]])
                if not results or results[0].masks is None or len(results[0].masks) == 0:
                    continue

                mask = results[0].masks.data.cpu().numpy()[0].astype(bool)
                # Resize if needed
                if mask.shape != (frame_h, frame_w):
                    import cv2
                    mask = cv2.resize(
                        mask.astype(np.uint8), (frame_w, frame_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                distractor_merged |= mask

            # Apply ball protection zone — clear distractor pixels near ball
            if ball_protect is not None:
                px1, py1, px2, py2 = ball_protect
                distractor_merged[py1:py2, px1:px2] = False

            if not distractor_merged.any():
                continue

            # Build quadmask using existing generate_quadmask_frame
            # Create ball mask from protection zone for the quadmask builder
            ball_mask = None
            if ball_protect is not None:
                ball_mask = np.zeros((frame_h, frame_w), dtype=bool)
                px1, py1, px2, py2 = ball_protect
                ball_mask[py1:py2, px1:px2] = True

            quadmasks[i] = self.generate_quadmask_frame(
                ball_mask=ball_mask,
                distractor_masks=[distractor_merged],
                corridor=corridor,
                frame_h=frame_h,
                frame_w=frame_w,
            )

            if i % 50 == 0:
                logger.info("Targeted quadmask generation: frame %d / %d", i, num_frames)

        return quadmasks
```

Also add the import at the top of cleaner.py:

```python
from clpga_demo.scene_analyzer import SceneAnalysis
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cleaner.py::TestTargetedQuadmaskGeneration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/cleaner.py tests/test_cleaner.py
git commit -m "feat: add VLM-guided targeted quadmask generation with ball protection zone"
```

---

### Task 4: Remove Old Segment-Everything Code

**Files:**
- Modify: `clpga_demo/cleaner.py`
- Modify: `tests/test_cleaner.py`

- [ ] **Step 1: Remove old methods from cleaner.py**

Remove these methods from `FrameCleaner`:
- `generate_quadmasks` (the segment-everything version)
- `_identify_ball_mask_crop`
- `_generate_quadmask_from_crop`
- `identify_ball_mask`

Keep:
- `compute_corridors`
- `generate_quadmask_frame`
- `generate_quadmasks_targeted` (new from Task 3)
- `clean_segments`
- `split_into_segments`
- `blend_segments`
- `_get_sam3`

- [ ] **Step 2: Remove old tests from tests/test_cleaner.py**

Remove these test classes:
- `TestBallMaskIdentification`
- `TestGenerateQuadmasks`

Keep:
- `TestCorridorComputation`
- `TestQuadmaskGeneration` (tests `generate_quadmask_frame` which is still used)
- `TestSegmentSplitting`
- `TestOverlapBlending`
- `TestCleanSegments`
- `TestTargetedQuadmaskGeneration` (new from Task 3)

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `uv run pytest tests/test_cleaner.py -v`
Expected: All remaining tests PASS

- [ ] **Step 4: Commit**

```bash
git add clpga_demo/cleaner.py tests/test_cleaner.py
git commit -m "refactor: remove segment-everything code, keep targeted approach"
```

---

### Task 5: Preset & CLI — Add gemini_model Parameter

**Files:**
- Modify: `clpga_demo/presets.py`
- Modify: `clpga_demo/__main__.py`
- Modify: `tests/test_presets.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing preset test**

```python
# Append to tests/test_presets.py

class TestGeminiPresets:
    def test_default_preset_has_gemini_model(self):
        from clpga_demo.presets import get_preset
        preset = get_preset("default")
        assert preset["gemini_model"] == "gemini-2.5-flash-preview-05-20"

    def test_putt_preset_has_gemini_model(self):
        from clpga_demo.presets import get_preset
        preset = get_preset("putt")
        assert preset["gemini_model"] == "gemini-2.5-flash-preview-05-20"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_presets.py::TestGeminiPresets -v`
Expected: FAIL — `KeyError: 'gemini_model'`

- [ ] **Step 3: Add gemini_model to presets**

Add to both presets in `clpga_demo/presets.py`:

```python
        "gemini_model": "gemini-2.5-flash-preview-05-20",
```

Add after `"clean_prompt"` line in both `"default"` and `"putt"` dicts.

- [ ] **Step 4: Write failing CLI test**

```python
# Append to tests/test_cli.py

class TestGeminiCLI:
    def test_gemini_model_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--gemini-model", "gemini-2.0-flash"])
        assert args.gemini_model == "gemini-2.0-flash"

    def test_gemini_model_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.gemini_model is None

    def test_gemini_model_resolves_to_preset(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--gemini-model", "gemini-2.0-flash"])
        resolved = resolve_args(args)
        assert resolved["gemini_model"] == "gemini-2.0-flash"
```

- [ ] **Step 5: Add CLI flag and resolve mapping**

In `clpga_demo/__main__.py`, add to `build_parser()` before `return parser`:

```python
    parser.add_argument("--gemini-model", type=str, default=None, help="Gemini model for scene analysis")
```

Add to `cli_to_preset` dict in `resolve_args()`:

```python
        "gemini_model": "gemini_model",
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_presets.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add clpga_demo/presets.py clpga_demo/__main__.py tests/test_presets.py tests/test_cli.py
git commit -m "feat: add gemini_model preset parameter and CLI flag"
```

---

### Task 6: Pipeline Integration — Wire SceneAnalyzer into Pass 2

**Files:**
- Modify: `clpga_demo/pipeline.py`
- Modify: `clpga_demo/__main__.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_pipeline.py (replace existing TestCleaningPipeline)

class TestVLMCleaningPipeline:
    def test_clean_creates_scene_analyzer(self, tmp_path):
        """When clean=True, process_video should create SceneAnalyzer and use targeted quadmasks."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path, frames=20)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            with patch("clpga_demo.pipeline.SceneAnalyzer") as MockAnalyzer:
                mock_analyzer = MockAnalyzer.return_value
                mock_analyzer.analyze_frame.return_value = MagicMock(
                    ball_bbox=(100, 120, 110, 130),
                    distractors=[],
                    scene_description="test green",
                )
                with patch("clpga_demo.pipeline.FrameCleaner") as MockCleaner:
                    mock_cleaner = MockCleaner.return_value
                    mock_cleaner.compute_corridors.return_value = [None] * 20
                    mock_cleaner.generate_quadmasks_targeted.return_value = np.full((20, 240, 320), 255, dtype=np.uint8)
                    mock_cleaner.clean_segments.return_value = [np.zeros((20, 240, 320, 3), dtype=np.uint8)]
                    MockCleaner.split_into_segments.return_value = [(0, 20)]
                    MockCleaner.blend_segments.return_value = np.zeros((20, 240, 320, 3), dtype=np.uint8)
                    with patch("clpga_demo.pipeline.VoidModelWrapper") as MockVoid:
                        mock_void = MockVoid.return_value
                        mock_void.download_if_needed.return_value = "/fake"
                        with patch("clpga_demo.pipeline._retrack_cleaned") as mock_retrack:
                            mock_retrack.return_value = [(100 + i * 10, 120) for i in range(20)]
                            process_video(input_path, output_path, clean=True)

        assert MockAnalyzer.called
        assert mock_cleaner.generate_quadmasks_targeted.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::TestVLMCleaningPipeline -v`
Expected: FAIL — `AttributeError: module 'clpga_demo.pipeline' has no attribute 'SceneAnalyzer'`

- [ ] **Step 3: Update pipeline.py to use SceneAnalyzer**

In `clpga_demo/pipeline.py`, update the Pass 2 section. Replace the current quadmask generation block:

```python
    # --- Pass 2 (optional): Clean distractors and re-track ---
    if clean:
        from clpga_demo.cleaner import FrameCleaner
        from clpga_demo.void_model import VoidModelWrapper
        from clpga_demo.scene_analyzer import SceneAnalyzer

        void_wrapper = VoidModelWrapper(model_dir=void_model_dir)
        void_wrapper.download_if_needed()
        void_wrapper.load()

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

        # Read all frames into memory for cleaning
        cap = cv2.VideoCapture(source)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        video_frames = np.stack(all_frames)

        # Compute median ball size for protection zone sizing
        valid_sizes = [s for s in ball_sizes if s > 0]
        median_ball_size = float(sorted(valid_sizes)[len(valid_sizes) // 2]) if valid_sizes else 20.0

        # Split into segments and process each with VLM-guided cleaning
        segments = FrameCleaner.split_into_segments(frame_count, segment_max_frames, segment_overlap_frames)
        analyzer = SceneAnalyzer(model=gemini_model)

        all_quadmasks = np.full((frame_count, src_h, src_w), 255, dtype=np.uint8)
        segment_prompts: list[str] = []

        for seg_start, seg_end in segments:
            # Extract midpoint keyframe for VLM analysis
            mid_idx = seg_start + (seg_end - seg_start) // 2
            keyframe = video_frames[mid_idx]
            logger.info("Analyzing keyframe %d for segment %d-%d ...", mid_idx, seg_start, seg_end)
            analysis = analyzer.analyze_frame(keyframe)
            segment_prompts.append(analysis.scene_description)

            logger.info(
                "Scene: %s | Ball: %s | Distractors: %d",
                analysis.scene_description,
                analysis.ball_bbox,
                len(analysis.distractors),
            )

            # Generate targeted quadmasks for this segment
            seg_frames = video_frames[seg_start:seg_end]
            seg_corridors = corridors[seg_start:seg_end]
            seg_quadmasks = cleaner.generate_quadmasks_targeted(
                seg_frames, seg_corridors, analysis, median_ball_size,
            )
            all_quadmasks[seg_start:seg_end] = seg_quadmasks

        # Run void-model per segment with scene-aware prompts
        cleaned_segments = []
        for (seg_start, seg_end), prompt in zip(segments, segment_prompts):
            seg_video = video_frames[seg_start:seg_end]
            seg_mask = all_quadmasks[seg_start:seg_end]

            if np.all(seg_mask == 255):
                logger.debug("Segment %d-%d: no distractors, skipping", seg_start, seg_end)
                cleaned_segments.append(seg_video.copy())
                continue

            logger.info("Cleaning segment %d-%d: %s", seg_start, seg_end, prompt)
            result = void_wrapper.inpaint(seg_video, seg_mask, prompt)
            cleaned_segments.append(result)

        cleaned_video = FrameCleaner.blend_segments(cleaned_segments, segments, frame_count)

        # Save debug videos
        output_dir = Path(output).parent
        stem = Path(output).stem
        _save_debug_video(str(output_dir / f"{stem}_cleaned.mp4"), cleaned_video, fps)
        _save_debug_video(
            str(output_dir / f"{stem}_quadmask.mp4"),
            np.stack([cv2.cvtColor(q, cv2.COLOR_GRAY2BGR) for q in all_quadmasks]),
            fps,
        )
        logger.info("Saved debug videos: %s_cleaned.mp4, %s_quadmask.mp4", stem, stem)

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

        refined_positions = _retrack_cleaned(
            cleaned_video, model, confidence, text, tracker2, src_w, src_h, fps,
        )

        if not all(p is None for p in refined_positions):
            positions = refined_positions
        else:
            logger.warning("Cleaning pass produced no detections; falling back to rough trajectory")
```

Also add `gemini_model` parameter to `process_video` signature:

```python
    gemini_model: str = "gemini-2.5-flash-preview-05-20",
```

And update `__main__.py` to pass it:

```python
            gemini_model=resolved.get("gemini_model", "gemini-2.5-flash-preview-05-20"),
```

- [ ] **Step 4: Remove old TestCleaningPipeline test class**

Remove `TestCleaningPipeline` from `tests/test_pipeline.py` (replaced by `TestVLMCleaningPipeline`).

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add clpga_demo/pipeline.py clpga_demo/__main__.py tests/test_pipeline.py
git commit -m "feat: wire SceneAnalyzer into Pass 2 with per-segment VLM analysis"
```

---

### Task 7: Install google-genai Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add google-genai to dependencies**

Add to `dependencies` list in `pyproject.toml`:

```toml
    "google-genai>=1.0",
```

- [ ] **Step 2: Install and verify**

Run: `uv pip install google-genai`
Run: `uv run python -c "from google import genai; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add google-genai for Gemini scene analysis"
```

---

### Task 8: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Verify existing pipeline works without --clean**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: All existing tests pass — no regressions

- [ ] **Step 3: Verify clean=False path is unchanged**

The `test_clean_false_skips_cleaning` test should still pass, confirming the pipeline is identical when `--clean` is not used.
