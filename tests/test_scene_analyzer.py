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
