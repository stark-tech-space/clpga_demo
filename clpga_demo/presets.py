"""Named shot presets for pipeline parameter tuning."""

from __future__ import annotations

SHOT_PRESETS: dict[str, dict] = {
    "default": {
        "smoothing_sigma_seconds": 0.5,
        "confidence": 0.25,
        "text": ["golf ball"],
        "tracker_type": "momentum",
    },
    "putt": {
        "smoothing_sigma_seconds": 0.1,
        "confidence": 0.15,
        "text": ["golf ball on green"],
        "tracker_type": "momentum",
        "momentum_history_size": 5,
        "momentum_radius_scale": 2.0,
        "kalman_process_noise": 0.5,
        "kalman_measurement_noise": 1.0,
        "kalman_gate_threshold": 9.0,
    },
}


def get_preset(name: str) -> dict:
    """Return a copy of the named preset dict, or raise ValueError."""
    if name not in SHOT_PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Available: {', '.join(SHOT_PRESETS)}")
    return {**SHOT_PRESETS[name]}
