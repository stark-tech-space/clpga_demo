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
