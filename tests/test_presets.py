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
        assert preset["momentum_radius_scale"] == 15.0
        assert "smoothing_alpha" not in preset

    def test_raises_on_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_returns_copy_not_reference(self):
        preset = get_preset("default")
        preset["confidence"] = 999
        assert get_preset("default")["confidence"] == 0.25


class TestTrackerPresets:
    def test_default_has_tracker_type(self):
        preset = get_preset("default")
        assert preset["tracker_type"] == "momentum"

    def test_putt_has_tracker_type(self):
        preset = get_preset("putt")
        assert preset["tracker_type"] == "momentum"

    def test_putt_has_kalman_params(self):
        preset = get_preset("putt")
        assert preset["kalman_process_noise"] == 0.5
        assert preset["kalman_measurement_noise"] == 1.0
        assert preset["kalman_gate_threshold"] == 9.0


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


class TestGeminiPresets:
    def test_default_preset_has_gemini_model(self):
        from clpga_demo.presets import get_preset
        preset = get_preset("default")
        assert preset["gemini_model"] == "gemini-2.5-flash-preview-05-20"

    def test_putt_preset_has_gemini_model(self):
        from clpga_demo.presets import get_preset
        preset = get_preset("putt")
        assert preset["gemini_model"] == "gemini-2.5-flash-preview-05-20"
