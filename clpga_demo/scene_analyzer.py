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
        """Analyze a single frame and return scene understanding."""
        raise NotImplementedError("Will be implemented in Task 2")
