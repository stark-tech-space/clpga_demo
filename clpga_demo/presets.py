"""Named shot presets for pipeline parameter tuning."""

from __future__ import annotations

SHOT_PRESETS: dict[str, dict] = {
    "default": {
        "smoothing_sigma_seconds": 0.5,
        "smoothing_alpha": 0.15,
        "confidence": 0.25,
        "text": ["golf ball"],
    },
    "putt": {
        "smoothing_sigma_seconds": 0.1,
        "smoothing_alpha": 0.4,
        "confidence": 0.15,
        "text": ["golf ball on green"],
    },
}


def get_preset(name: str) -> dict:
    """Return a copy of the named preset dict, or raise ValueError."""
    if name not in SHOT_PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Available: {', '.join(SHOT_PRESETS)}")
    return {**SHOT_PRESETS[name]}
