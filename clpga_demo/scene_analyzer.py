"""VLM-based scene analysis for distractor identification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from google import genai

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


_EMPTY_ANALYSIS = SceneAnalysis(
    ball_bbox=None,
    distractors=[],
    scene_description="golf course background",
)


class SceneAnalyzer:
    """Analyzes video frames using a VLM to identify the ball and distractors."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._client = None

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
