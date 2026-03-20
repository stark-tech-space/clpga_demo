# Shot Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add named shot presets (`default`, `putt`) so putt videos get tuned smoothing and detection parameters via a `--preset` CLI flag.

**Architecture:** New `presets.py` module holds the preset dictionary and lookup helper. `__main__.py` resolves preset + CLI overrides into final parameter values. `pipeline.py` threads the `text` parameter through to `track_video()`.

**Tech Stack:** Python, argparse, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-shot-presets-design.md`

---

### Task 1: Preset Module

**Files:**
- Create: `clpga_demo/presets.py`
- Create: `tests/test_presets.py`

- [ ] **Step 1: Write failing tests for `get_preset()`**

```python
# tests/test_presets.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_presets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clpga_demo.presets'`

- [ ] **Step 3: Implement `presets.py`**

```python
# clpga_demo/presets.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_presets.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/presets.py tests/test_presets.py
git commit -m "feat: add shot presets module with default and putt presets"
```

---

### Task 2: CLI `--preset` Argument

**Files:**
- Modify: `clpga_demo/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for `--preset` argument and override resolution**

Add to `tests/test_cli.py`:

```python
from clpga_demo.__main__ import build_parser, resolve_args


class TestPresetArg:
    def test_preset_default(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.preset == "default"

    def test_preset_putt(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt"])
        assert args.preset == "putt"


class TestResolveArgs:
    def test_default_preset_values(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        resolved = resolve_args(args)
        assert resolved["smoothing_sigma_seconds"] == 0.5
        assert resolved["confidence"] == 0.25
        assert resolved["text"] == ["golf ball"]

    def test_putt_preset_values(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt"])
        resolved = resolve_args(args)
        assert resolved["smoothing_sigma_seconds"] == 0.1
        assert resolved["smoothing_alpha"] == 0.4
        assert resolved["confidence"] == 0.15
        assert resolved["text"] == ["golf ball on green"]

    def test_cli_override_beats_preset(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt", "--smoothing-sigma", "0.3"])
        resolved = resolve_args(args)
        assert resolved["smoothing_sigma_seconds"] == 0.3
        # Non-overridden values come from putt preset
        assert resolved["confidence"] == 0.15
        assert resolved["text"] == ["golf ball on green"]

    def test_confidence_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt", "--confidence", "0.5"])
        resolved = resolve_args(args)
        assert resolved["confidence"] == 0.5
        assert resolved["smoothing_sigma_seconds"] == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `AttributeError: 'Namespace' has no attribute 'preset'` and `ImportError: cannot import name 'resolve_args'`

- [ ] **Step 3: Add `--preset` arg and `resolve_args()` to `__main__.py`**

Change overrideable arg defaults to `None` so we can detect explicit CLI usage. Update `build_parser()`:

```python
parser.add_argument("--preset", default="default", help="Shot preset: default, putt (default: default)")
parser.add_argument("--smoothing-sigma", type=float, default=None, help="Gaussian sigma in seconds (file mode)")
parser.add_argument("--smoothing-alpha", type=float, default=None, help="EMA alpha (live mode)")
parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
```

**Note:** The existing `--smoothing-sigma`, `--smoothing-alpha`, and `--confidence` arguments must have their defaults changed from their old values (0.5, 0.15, 0.25) to `None`. This is required so `resolve_args()` can distinguish "user explicitly passed this flag" from "argparse used its default." The preset system now owns the defaults.

Update existing `TestCLIParser.test_defaults` test to reflect `None` defaults:

```python
def test_defaults(self):
    """Default values should be None (presets supply real defaults)."""
    parser = build_parser()
    args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
    assert args.smoothing_sigma is None
    assert args.smoothing_alpha is None
    assert args.model == "sam3.pt"
    assert args.confidence is None
```

Add new function `resolve_args()`:

```python
def resolve_args(args: argparse.Namespace) -> dict:
    """Resolve preset defaults with explicit CLI overrides."""
    from clpga_demo.presets import get_preset

    preset = get_preset(args.preset)

    # CLI flags that map to preset keys — only override if explicitly set (not None)
    cli_to_preset = {
        "smoothing_sigma": "smoothing_sigma_seconds",
        "smoothing_alpha": "smoothing_alpha",
        "confidence": "confidence",
    }

    for cli_key, preset_key in cli_to_preset.items():
        cli_val = getattr(args, cli_key)
        if cli_val is not None:
            preset[preset_key] = cli_val

    return preset
```

Update `main()` to use `resolve_args()`:

```python
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    resolved = resolve_args(args)

    from clpga_demo.pipeline import process_stream, process_video

    try:
        if args.live:
            process_stream(
                source=args.source,
                output=args.output,
                model=args.model,
                confidence=resolved["confidence"],
                smoothing_alpha=resolved["smoothing_alpha"],
                text=resolved["text"],
            )
        else:
            process_video(
                source=args.source,
                output=args.output,
                model=args.model,
                confidence=resolved["confidence"],
                smoothing_sigma_seconds=resolved["smoothing_sigma_seconds"],
                text=resolved["text"],
            )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All passed (existing + new tests)

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/__main__.py tests/test_cli.py
git commit -m "feat: add --preset CLI flag with override resolution"
```

---

### Task 3: Pipeline `text` Parameter

**Files:**
- Modify: `clpga_demo/pipeline.py:18-77` (`process_video`)
- Modify: `clpga_demo/pipeline.py:80-149` (`process_stream`)
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for `text` passthrough**

Add to `tests/test_pipeline.py`:

```python
class TestProcessVideoText:
    def test_passes_text_to_tracker(self, tmp_path):
        """process_video should forward the text parameter to track_video."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        captured_kwargs = {}

        def capturing_tracker(source, **kwargs):
            captured_kwargs.update(kwargs)
            return _mock_track_video(source)

        with patch("clpga_demo.pipeline.track_video", side_effect=capturing_tracker):
            process_video(input_path, output_path, text=["golf ball on green"])

        assert captured_kwargs["text"] == ["golf ball on green"]

    def test_default_text_is_none(self, tmp_path):
        """process_video without text param should pass None (tracker uses its default)."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        captured_kwargs = {}

        def capturing_tracker(source, **kwargs):
            captured_kwargs.update(kwargs)
            return _mock_track_video(source)

        with patch("clpga_demo.pipeline.track_video", side_effect=capturing_tracker):
            process_video(input_path, output_path)

        assert captured_kwargs.get("text") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py::TestProcessVideoText -v`
Expected: FAIL — `TypeError: process_video() got an unexpected keyword argument 'text'`

- [ ] **Step 3: Add `text` parameter to `process_video()` and `process_stream()`**

In `pipeline.py`, update `process_video` signature:

```python
def process_video(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_sigma_seconds: float = 0.5,
    text: list[str] | None = None,
) -> None:
```

Update the `track_video` call on line 46:

```python
for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence, text=text):
```

Similarly update `process_stream` signature:

```python
def process_stream(
    source: str,
    output: str,
    model: str = "sam3.pt",
    confidence: float = 0.25,
    smoothing_alpha: float = 0.15,
    text: list[str] | None = None,
) -> None:
```

Update the `track_video` call on line 104:

```python
for frame_idx, orig_frame, boxes in track_video(source, model=model, confidence=confidence, text=text):
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add clpga_demo/pipeline.py tests/test_pipeline.py
git commit -m "feat: thread text parameter through pipeline to tracker"
```

---

### Task 4: Full Integration Verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Smoke test with putt preset**

Run: `uv run --env-file .env python -m clpga_demo clips/012_00-13-22_ying-xu-sinks-a-short-par-putt-on-the-6th-hole.mp4 -o clips_processed/012_putt_test.mp4 --preset putt`
Expected: Completes successfully, output video shows improved ball tracking with less crop lag

- [ ] **Step 3: Smoke test with default preset (regression check)**

Run: `uv run --env-file .env python -m clpga_demo clips/001_00-01-03_li-gengshan-hits-a-tee-shot-on-the-8th-hole-landin.mp4 -o clips_processed/001_default_test.mp4`
Expected: Same behavior as before — default preset matches original hardcoded values

- [ ] **Step 4: Commit any fixes, then final commit if clean**

```bash
git status
```
