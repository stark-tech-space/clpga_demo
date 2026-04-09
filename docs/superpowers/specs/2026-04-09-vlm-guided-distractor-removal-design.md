# VLM-Guided Distractor Removal

**Date:** 2026-04-09
**Status:** Approved
**Supersedes:** Segment-everything approach in `2026-04-08-distractor-removal-design.md` (Pass 2 internals only)

## Problem

The segment-everything approach from the initial distractor removal design produces 100-800+ masks per frame on putting greens. The heuristic ball mask identification (proximity + size/aspect ratio) cannot reliably distinguish the golf ball from similar-looking grass patches, shadows, and divots. This causes the pipeline to frequently remove the ball itself, making tracking worse than the uncleaned baseline.

## Solution

Replace segment-everything with **VLM-guided targeted segmentation**. Use Gemini Flash to analyze one keyframe per void-model segment, identifying the ball and specific distractors by name and bounding box. SAM3 then segments only those named distractors with precise box prompts. The ball gets an explicit protection zone derived from Gemini's ball bounding box.

## Architecture: Revised Pass 2

Passes 1 and 3 are unchanged from the original design. Only the internals of Pass 2 change.

```
PASS 2: VLM-Guided Cleaning + Refined Tracking

  1. Split video into segments (85 frames, 16-frame overlap)
  2. Per segment:
     a. Extract midpoint keyframe (frame at segment_start + segment_length/2)
     b. Send keyframe to Gemini Flash → SceneAnalysis:
        - ball bounding box (or None if not visible)
        - distractor list: [{label, bbox}, ...]
        - scene description (for void-model prompt)
     c. If no distractors → skip segment (quadmask all-255)
     d. Per frame in segment:
        - For each distractor: SAM3 box prompt → precise mask
        - Merge distractor masks (logical OR)
        - Apply ball protection zone (zero out distractor pixels near ball)
        - Build quadmask (0=remove, 127=corridor, 255=keep)
     e. Run void-model on segment with scene-aware prompt
  3. Blend overlapping segments (linear crossfade)
  4. Write cleaned frames to temp video
  5. Re-track with SAM3 text-prompt detection → refined trajectory
```

## SceneAnalyzer Module

### Interface

```python
@dataclass(frozen=True)
class SceneAnalysis:
    ball_bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2) or None
    distractors: list[dict]  # [{"label": str, "bbox": (x1,y1,x2,y2)}, ...]
    scene_description: str   # Brief scene summary for void-model prompt

class SceneAnalyzer:
    def __init__(self, model: str = "gemini-2.5-flash-preview-05-20")
    def analyze_frame(self, frame: np.ndarray) -> SceneAnalysis
```

### Gemini API Details

- **SDK:** `google-genai` (pip installable)
- **Auth:** `GEMINI_API_KEY` environment variable
- **Input:** JPEG-compressed frame (~50-100KB)
- **Output:** Structured JSON via Gemini's response schema
- **Timeout:** 30 seconds per call

### Prompt

```
You are analyzing a golf putting scene. Identify:
1. The golf ball being putted — provide its bounding box [x1, y1, x2, y2] in pixel coordinates.
2. Any objects that could occlude or be confused with the ball during its path:
   other golf balls, flagpoles/pins, people, club heads, shadows of people,
   tee markers, range markers, etc.
   For each, provide a label and bounding box [x1, y1, x2, y2].
3. A brief description of the background scene (e.g., "golf putting green with
   flagpole and gallery in background").

Only include objects actually visible in the frame.
Return JSON with keys: ball, distractors, scene_description.
```

### Failure Handling

If Gemini fails (timeout, rate limit, invalid JSON, network error):
- Return empty `SceneAnalysis(ball_bbox=None, distractors=[], scene_description="golf course background")`
- Segment is skipped for cleaning (quadmask all-255)
- Pipeline falls back to raw tracking for that segment
- Log warning but do not raise

This is **fail-safe**: a Gemini failure never makes tracking worse. At worst, a segment gets no cleaning.

### Provider Abstraction

`SceneAnalyzer.__init__` accepts a model string. The implementation starts with Gemini Flash but the interface is model-agnostic. Future providers (GPT-4o, local VLMs) only need to implement `analyze_frame(frame) -> SceneAnalysis`.

## Targeted SAM3 Segmentation

### Per-Frame Distractor Masking

For each frame in a segment, given the `SceneAnalysis` from the segment's keyframe:

1. **For each distractor:**
   - Take distractor bbox from Gemini
   - Expand by 20% of bbox width/height in all directions (movement tolerance within ~3s segment)
   - Clip to frame bounds
   - Call SAM3 with box prompt → get single precise mask
   - If SAM3 returns no mask (object not in frame): skip this distractor for this frame

2. **Merge distractor masks:** Logical OR of all individual distractor masks

3. **Ball protection zone:**
   - Take Gemini's ball bbox
   - Expand by 2x `median_ball_size` in all directions
   - Any distractor mask pixels inside this zone → cleared (set to False)
   - This ensures the ball is never accidentally removed even if a distractor bbox overlaps

4. **Build quadmask:**
   - Dilate merged distractor mask by `mask_dilation_px` (existing parameter)
   - 0 = dilated distractor region (remove)
   - 127 = corridor interior, non-distractor, non-ball (affected)
   - 255 = ball protection zone + outside corridor (keep)

### Performance

- SAM3 box prompt: ~100ms per call (vs ~1.5s for segment-everything grid)
- Typical scene: 1-5 distractors × 85 frames = 85-425 SAM3 calls per segment ≈ 10-45 seconds
- Previous approach: 85 frames × 1.5s = 128 seconds per segment
- **~3-5x faster** for segmentation step

### Scene Description as void-model Prompt

Gemini's `scene_description` replaces the static `clean_prompt` preset parameter. void-model gets a per-segment, scene-aware inpainting prompt (e.g., "golf putting green with trees in background") instead of a generic "golf course background". This should improve inpainting quality.

## Code Changes

### New Files

| File | Responsibility |
|------|---------------|
| `clpga_demo/scene_analyzer.py` | `SceneAnalyzer` class, `SceneAnalysis` dataclass, Gemini API integration |

### Modified Files

| File | Changes |
|------|---------|
| `clpga_demo/cleaner.py` | Replace `generate_quadmasks` internals: remove point grid + `_identify_ball_mask_crop`, add `generate_quadmasks_targeted` using `SceneAnalysis` with box prompts + ball protection zone |
| `clpga_demo/pipeline.py` | Create `SceneAnalyzer` in Pass 2, pass to `FrameCleaner` |
| `clpga_demo/presets.py` | Add `gemini_model` parameter to presets |
| `clpga_demo/__main__.py` | Add `--gemini-model` CLI flag |

### Removed Code

| What | Why |
|------|-----|
| `_identify_ball_mask_crop` method | Replaced by Gemini's ball bbox protection zone |
| `identify_ball_mask` method | No longer needed — distractors are known by name |
| Point grid in `generate_quadmasks` | Replaced by targeted box prompts from Gemini |
| `_generate_quadmask_from_crop` method | Replaced by new targeted quadmask builder |

### Preserved Code (Unchanged)

- `void_model.py` — inpainting unchanged
- `Corridor` dataclass and `compute_corridors` — still defines region of interest
- `split_into_segments` and `blend_segments` — segment management unchanged
- `generate_quadmask_frame` — low-level quadmask builder reused
- `clean_segments` — void-model orchestration unchanged
- `_retrack_cleaned` — re-tracking via temp video unchanged
- `_save_debug_video` — debug output unchanged
- Pass 1 and Pass 3 — completely unchanged

## Preset Configuration

```python
# New parameter added to SHOT_PRESETS
"gemini_model": "gemini-2.5-flash-preview-05-20",
```

## CLI Flags

```
--gemini-model TEXT    # Override Gemini model name (default from preset)
```

## Dependencies

```
google-genai          # Gemini API SDK
```

`GEMINI_API_KEY` environment variable must be set when `--clean` is used.

## Cost Estimate

- Gemini Flash: ~$0.10 per 1M input tokens
- One 1080p frame ≈ 750 image tokens + ~200 prompt tokens
- 4 segments × 1 keyframe = ~3,800 tokens ≈ $0.0004 per video
- 12 putt clips ≈ $0.005 total

## Future Optimizations

### Local VLM Replacement
Replace Gemini with a local vision-language model (Qwen-VL, LLaVA) to eliminate API dependency and latency. The `SceneAnalyzer` interface is designed for this swap.

### Distractor Tracking Across Frames
Instead of re-prompting SAM3 every frame with the same static bbox, use SAM3's video tracking mode to propagate distractor masks temporally after the first detection. Faster and more consistent.

### Adaptive Keyframe Selection
Instead of fixed midpoint, select keyframes where the scene changes most (camera movement, new objects entering frame). Could use optical flow or frame differencing.
