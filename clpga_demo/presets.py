"""Named shot presets for pipeline parameter tuning."""

from __future__ import annotations

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
        "segment_max_frames": 85,
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
        "segment_max_frames": 85,
        "segment_overlap_frames": 16,
        "void_model_dir": None,
        "clean_prompt": "golf putting green background",
    },
}


def get_preset(name: str) -> dict:
    """Return a copy of the named preset dict, or raise ValueError."""
    if name not in SHOT_PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Available: {', '.join(SHOT_PRESETS)}")
    return {**SHOT_PRESETS[name]}
