import pytest

from clpga_demo.presets import SHOT_PRESETS, get_preset


class TestGetPreset:
    def test_returns_default_preset(self):
        preset = get_preset("default")
        assert preset["smoothing_sigma_seconds"] == 0.5
        assert preset["smoothing_alpha"] == 0.15
        assert preset["confidence"] == 0.25
        assert preset["text"] == ["golf ball"]

    def test_returns_putt_preset(self):
        preset = get_preset("putt")
        assert preset["smoothing_sigma_seconds"] == 0.1
        assert preset["smoothing_alpha"] == 0.4
        assert preset["confidence"] == 0.15
        assert preset["text"] == ["golf ball on green"]

    def test_raises_on_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_returns_copy_not_reference(self):
        preset = get_preset("default")
        preset["confidence"] = 999
        assert get_preset("default")["confidence"] == 0.25
