# Distractor Removal via SAM3 + Netflix void-model

**Date:** 2026-04-08
**Status:** Approved

## Problem

Golf ball tracking is disrupted by distractors — other balls, flagpoles, people, equipment, and environmental clutter — that either occlude the ball or confuse the SAM3 detector into following the wrong target. The existing gating in `MomentumTracker.accept()` mitigates some false associations, but cannot prevent the detector itself from being confused by nearby objects.

## Solution

A two-pass cleaning pipeline that uses SAM3 (segment-everything mode) and Netflix's void-model (video inpainting) to remove distractors from an analysis copy of the video before re-running detection/tracking. Original frames are preserved for the final output crop.

## Architecture: Three-Pass Pipeline

```
INPUT VIDEO
    │
    ▼
┌─────────────────────────────────────────────────┐
│  PASS 1: Rough Tracking on Raw Frames           │
│  (Existing pipeline, unchanged)                  │
│                                                  │
│  SAM3 text-prompt detection → select_ball()      │
│  → BallTracker (momentum/kalman) update/predict  │
│                                                  │
│  Output: rough_trajectory[]                      │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  PASS 2: Corridor Cleaning + Refined Tracking    │
│  (New, opt-in via --clean flag)                  │
│                                                  │
│  1. Smooth rough_trajectory to define corridors  │
│  2. SAM3 segment-everything within each corridor │
│  3. Identify ball mask → keep (255)              │
│  4. Distractor masks → remove (0)                │
│  5. Generate quadmask video                      │
│  6. void-model inpaints video segments           │
│  7. Re-run detection/tracking on cleaned frames  │
│                                                  │
│  Output: refined_trajectory[]                    │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  PASS 3: Smooth + Crop Originals                 │
│  (Existing pipeline, unchanged)                  │
│                                                  │
│  GaussianSmoother.smooth(refined_trajectory)     │
│  → calculate_crop() → VideoWriter                │
│                                                  │
│  Output: 9:16 portrait video (original quality)  │
└─────────────────────────────────────────────────┘
```

## Corridor Sizing

Corridors are speed-adaptive regions around the rough trajectory where distractors are searched and removed.

**Per-frame corridor width:**
```
corridor_radius = max(ball_size * corridor_multiplier, speed_at_frame * corridor_speed_scale * radius_scale)
```

- `corridor_multiplier` (default 4.0): ball-size multiplier
- `corridor_speed_scale` (default 1.5): speed multiplier
- Wider where the ball moves fast (mid-flight), narrower at rest (address, green)
- Corridor is a square bounding box clipped to frame edges
- Precomputed for all frames before Pass 2 begins

## Ball Mask Identification

When SAM3 segments everything in a corridor, we must decide which mask is the ball (keep) and which are distractors (remove).

**Strategy:** The ball mask is the one whose centroid is closest to the rough trajectory point for that frame.

**Validation gates (reuse existing logic from MomentumTracker.accept()):**
- Aspect ratio: `max(w, h) / min(w, h) <= max_aspect_ratio`
- Size consistency: compared to median historical ball size within `max_size_ratio` tolerance

**Edge case — ball fully occluded:**
If no mask passes validation checks:
1. Inpaint ALL masks in the corridor (remove everything)
2. Re-run detection on the cleaned frame
3. If still no detection, tracker predicts through the gap (existing behavior)

This handles cases like a flagpole directly in front of the ball — removing the pole may reveal the ball behind it.

## Quadmask Generation

void-model uses a four-value mask format:

| Value | Meaning | Usage |
|-------|---------|-------|
| 0     | Remove  | Distractor masks |
| 63    | Overlap | Not used in initial implementation |
| 127   | Affected | Corridor boundary region |
| 255   | Keep    | Ball region + everything outside corridor |

**Mask preparation:**
1. Merge all distractor masks via logical OR
2. Dilate merged mask by `mask_dilation_px` (default 5px) to avoid artifact halos
3. Ball region set to 255
4. Corridor boundary pixels set to 127
5. Everything outside corridor set to 255

## void-model Integration

**Model:** Netflix void-model, built on CogVideoX-Fun 3D video transformer.

**Source:**
- Checkpoints: `netflix/void-model` on Hugging Face (Apache 2.0)
- Base model: `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`
- Download via `huggingface_hub.snapshot_download()`

**Hardware:** 40GB+ VRAM (A100-class GPU)

**Video segmentation:**
- void-model processes up to 197 frames per invocation
- Split video into segments of `segment_max_frames` (default 180)
- Overlap segments by `segment_overlap_frames` (default 16) for temporal continuity
- Blend overlapping regions via linear crossfade: in the overlap zone, frame weights transition linearly from 1.0/0.0 (start of overlap) to 0.0/1.0 (end of overlap) between the preceding and following segments

**Invocation:**
- Input: video segment (RGB) + quadmask segment + text prompt (e.g., "golf course background")
- Output: cleaned video segment with distractors inpainted
- Model loaded once at pipeline start, shared across all segments

**When to skip inpainting for a frame:**
- Only one mask found in corridor (presumably the ball) — nothing to remove
- No masks found in corridor — region is already clean
- In both cases, use the raw frame for detection

## New Code

**New files:**
- `clpga_demo/cleaner.py` — `FrameCleaner` class: corridor computation, SAM3 segment-everything orchestration, quadmask generation, segment management
- `clpga_demo/void_model.py` — `VoidModelWrapper` class: download, loading, inference interface

**Modified files:**
- `clpga_demo/pipeline.py` — Add Pass 2 between existing Pass 1 and Pass 3
- `clpga_demo/presets.py` — Add cleaning parameters to preset dicts
- `clpga_demo/__main__.py` — Add CLI flags for cleaning configuration

### FrameCleaner API

```python
class FrameCleaner:
    def __init__(self, sam3_model, void_model: VoidModelWrapper, corridor_config: dict)
    def compute_corridors(self, rough_trajectory, ball_sizes, speeds) -> list[Corridor]
    def generate_quadmasks(self, video_path, corridors, rough_trajectory) -> quadmask_video
    def clean_segments(self, video_path, quadmasks) -> list[ndarray]
    def identify_ball_mask(self, masks, trajectory_point, ball_size) -> int | None
```

### VoidModelWrapper API

```python
class VoidModelWrapper:
    def __init__(self, model_dir: str | None, device: str)
    def download_if_needed(self) -> str
    def load(self) -> Self
    def inpaint(self, video_segment, quadmask_segment, prompt: str) -> ndarray
```

### Pipeline integration

```python
# In process_video():
rough_trajectory = track_pass(video, sam3_model, tracker, ...)

if cleaning_enabled:
    cleaner = FrameCleaner(sam3_model, void_model, corridor_config)
    corridors = cleaner.compute_corridors(rough_trajectory, ...)
    quadmasks = cleaner.generate_quadmasks(video_path, corridors, rough_trajectory)
    cleaned_segments = cleaner.clean_segments(video_path, quadmasks)
    refined_trajectory = track_pass(cleaned_segments, sam3_model, tracker, ...)
else:
    refined_trajectory = rough_trajectory

smoothed = smoother.smooth(refined_trajectory)
crop_and_write(video_path, smoothed, output_path)
```

## Preset Configuration

```python
# New parameters added to SHOT_PRESETS
"clean": False,                     # Enable cleaning pass
"corridor_multiplier": 4.0,        # Ball-size multiplier for corridor width
"corridor_speed_scale": 1.5,       # Speed multiplier for corridor width
"mask_dilation_px": 5,             # Dilate distractor masks before inpainting
"segment_max_frames": 180,         # Max frames per void-model segment
"segment_overlap_frames": 16,      # Overlap between segments
"void_model_dir": None,            # Custom model path (auto-downloads if None)
"clean_prompt": "golf course background",  # Text prompt for void-model
```

## CLI Flags

```
--clean                           # Enable the cleaning pass
--corridor-multiplier FLOAT       # Override corridor width multiplier
--mask-dilation INT               # Override mask dilation pixels
--void-model-dir PATH             # Path to pre-downloaded void-model
--clean-prompt TEXT               # Scene description for void-model
```

## Future Optimizations (Not Implemented)

### 1. Online Cleaning in Pass 1
Use a lightweight per-frame inpainter (not void-model) during Pass 1 to clean frames online as the tracker predicts ahead. This would produce a better rough trajectory, making Pass 2's corridors more accurate. void-model is unsuitable for this because it operates on video clips, not individual frames.

### 2. Selective Re-processing (Option C)
Instead of cleaning the entire video in Pass 2, identify frames where tracking was poor (large jumps, lost track, low confidence) and only re-process those sections. This dramatically reduces void-model invocations.

### 3. Prompt-Based Distractor Segmentation (Approach 2)
Instead of SAM3 segment-everything, use text prompts for known distractor classes ("person", "flagpole", "golf ball") and exclude the closest ball to trajectory. Faster but requires maintaining a distractor class list and misses unlisted objects.
