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
