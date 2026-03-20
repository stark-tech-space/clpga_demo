# Shot Presets — Design Spec

Add named shot presets to the pipeline so putt videos (and future shot types) get tuned detection and smoothing parameters instead of one-size-fits-all defaults.

## Problem

Putt clips suffer from two issues with the current fixed defaults:
- **Crop lag:** Gaussian sigma of 0.5s (15 frames at 30fps) is too heavy — the crop window can't keep up with a rolling ball on the green.
- **Detection dropout:** Confidence threshold of 0.25 misses weaker detections where the ball blends with the green surface.

## Solution: Named Shot Presets

A `--preset` CLI flag selects a named parameter set. Explicit CLI flags override preset values.

### Preset Dictionary

New file `clpga_demo/presets.py`:

```python
SHOT_PRESETS = {
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
```

A `get_preset(name: str) -> dict` helper returns the preset or raises `ValueError` for unknown names.

### CLI Changes

`__main__.py` gains `--preset` argument (default: `"default"`):

```bash
python -m clpga_demo input.mp4 -o output.mp4 --preset putt
python -m clpga_demo input.mp4 -o output.mp4 --preset putt --smoothing-sigma 0.2
```

Resolution order:
1. Load preset dict by name
2. Overlay any explicitly-provided CLI flags (detected by comparing against argparse defaults)
3. Pass resolved values to `process_video()` / `process_stream()`

### Pipeline Changes

`process_video()` and `process_stream()` in `pipeline.py` gain an optional `text: list[str] | None` parameter. This is passed through to `track_video()`, which already accepts a `text` parameter.

### No Changes Required

- `smoother.py` — already parameterized by sigma
- `cropper.py` — not affected by shot type
- `tracker.py` — already accepts `text` and `confidence` parameters

## Files Changed

| File | Change |
|------|--------|
| `clpga_demo/presets.py` | New — preset dict and `get_preset()` helper |
| `clpga_demo/__main__.py` | Add `--preset` arg, resolve preset + CLI overrides |
| `clpga_demo/pipeline.py` | Add `text` parameter to `process_video()` and `process_stream()`, pass to `track_video()` |

## Testing

- Unit test `get_preset()` for known and unknown preset names
- Unit test CLI arg resolution: preset-only, preset + override, no preset (uses default)
- Integration: reprocess `012_00-13-22_ying-xu-sinks-a-short-par-putt-on-the-6th-hole.mp4` with `--preset putt` and verify improved tracking

## Future

Adding a new shot type (e.g., `chip`, `bunker`) is a one-line addition to the `SHOT_PRESETS` dict. No structural changes needed.
