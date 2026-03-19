# Golf Ball Tracker — Design Spec

Track a golf ball across video using SAM3 and output a 9:16 portrait-cropped video centered on the ball.

## Requirements

- **Detection:** Use SAM3 text prompt ("golf ball") as primary detection, bbox fallback if text fails
- **Ball selection:** When multiple balls visible, pick the one being hit using heuristic (most centered + largest). Gemini fallback deferred to later phase
- **Crop:** 9:16 portrait aspect ratio using full source height (e.g., 607x1080 for 1080p input). No black bars — native resolution crop
- **Smoothing:** Smooth crop window movement for cinematic feel
- **Input:** Pre-recorded video files and live camera/RTSP streams
- **Output:** Cropped video file
- **Interface:** CLI tool + importable Python API

## Architecture

### Approach: Hybrid Pipeline

Two-pass for pre-recorded files (full-trajectory Gaussian smoothing), single-pass for live streams (causal EMA smoothing). Shared core for tracking and cropping.

### Module Structure

```
clpga_demo/
├── __init__.py          # Package init, exports public API
├── __main__.py          # CLI entry point (argparse), enables python -m clpga_demo
├── tracker.py           # SAM3 ball detection & tracking
├── cropper.py           # Portrait crop extraction & video writing
├── smoother.py          # Trajectory smoothing (Gaussian for files, EMA for live)
└── pipeline.py          # Orchestrates tracker -> smoother -> cropper
```

## Tracking

### How SAM3VideoSemanticPredictor Works

`SAM3VideoSemanticPredictor` is a coupled detect+track system. It detects all matching objects AND tracks them internally across frames — you cannot select one detection and ask it to track only that one. The predictor manages object IDs (`obj_id`) internally.

### Detection Flow

1. Instantiate `SAM3VideoSemanticPredictor` and call `predictor(source="video.mp4", text=["golf ball"], stream=True)` — this iterates all frames, detecting and tracking golf balls automatically
2. For each frame's results, the predictor returns all tracked object instances with masks and bounding boxes
3. **Post-processing ball selection:** Apply heuristic to pick which tracked `obj_id` to center the crop on: `score = (1 / distance_from_center) * bbox_area` — favors centered, large balls (the one being hit)
4. **Sticky selection:** Once an `obj_id` is selected in the first frame, follow that same `obj_id` across subsequent frames. Only re-evaluate the heuristic if the tracked object is lost (no longer in results)
5. **Fallback:** If no detection in the first 30 frames via text prompt, use the same `SAM3VideoSemanticPredictor` instance with `add_prompt(frame_idx=0, bboxes=center_bbox)` to provide a manual bbox hint, then re-run. Warn the user.

### Predictor Invocation

- **Pre-recorded files:** `predictor(source="video.mp4", text=["golf ball"], stream=True)` — iterates all frames via the predictor's internal video dataset loader
- **Live streams:** `predictor(source="rtsp://camera/stream", text=["golf ball"], stream=True)` — same interface, the predictor's dataset handles RTSP sources

### Output Per Frame

`TrackResult(frame_idx, center_x, center_y, bbox, confidence, mask, obj_id)` — mask retained for future use (e.g., background effects), crop uses center coordinates only. `obj_id` used for sticky selection.

## Smoothing

### Pre-recorded (Gaussian)

Collect all center positions, apply `scipy.ndimage.gaussian_filter1d` with configurable sigma. Default sigma is time-based: `0.5 seconds * fps` (e.g., 15 frames at 30fps, 30 frames at 60fps). Bidirectional — looks forward and backward for ultra-smooth motion.

### Live (EMA)

`smoothed = alpha * current + (1 - alpha) * previous` with configurable alpha (default 0.15). Causal-only, responsive but dampens jitter.

## Cropping

- Crop width: `crop_w = source_height * 9 / 16`
- Center horizontally on smoothed ball position
- Edge clamping: `crop_x = max(0, min(smoothed_x - crop_w/2, source_width - crop_w))`
- No resize — native resolution output
- Write output video at source FPS, codec auto-detected or configurable
- Edge case: if source is already narrower than 9:16, crop vertically to maintain 9:16 ratio (`crop_h = source_width * 16 / 9`), centered on the ball's y-position

## Occlusion & Tracking Loss

When the selected `obj_id` disappears from results (ball occluded or out of frame):

- **Pre-recorded (two-pass):** After collecting all positions, interpolate missing frames linearly between the last known and next known positions before applying Gaussian smoothing
- **Live (single-pass):** Hold the last known smoothed position until the ball reappears. If lost for more than 3 seconds, re-run the ball selection heuristic on the next frame with detections

## Error Handling

- **Invalid video input:** Raise `ValueError` with descriptive message if file doesn't exist, codec unsupported, or video has zero frames
- **RTSP disconnect:** Log warning, attempt reconnection up to 3 times with 1-second backoff. After 3 failures, finalize the output video with frames collected so far
- **No ball detected in entire video:** Raise `RuntimeError` after processing completes with no detections. Output no file.

## CLI Interface

```bash
# Pre-recorded
python -m clpga_demo input.mp4 -o output.mp4

# Live stream
python -m clpga_demo rtsp://camera:554/stream -o output.mp4 --live

# Options
python -m clpga_demo input.mp4 -o output.mp4 \
  --smoothing-sigma 0.5 \       # Gaussian sigma in seconds (file mode)
  --smoothing-alpha 0.1 \       # EMA alpha (live mode)
  --model sam3.pt \              # SAM3 model path
  --confidence 0.25              # Detection confidence threshold
```

## Python API

```python
from clpga_demo.pipeline import process_video, process_stream

# Pre-recorded
process_video("input.mp4", "output.mp4", smoothing_sigma_seconds=0.5)

# Live
process_stream("rtsp://camera/stream", "output.mp4", smoothing_alpha=0.15)
```

## Dependencies

- `ultralytics` (SAM3) — already in pyproject.toml
- `opencv-python` — video I/O
- `scipy` — Gaussian smoothing
- `numpy` — array operations (transitive via ultralytics)

Note: `opencv-python` and `scipy` must be added to `pyproject.toml`.

## Future Enhancements (Not In Scope)

- Gemini API integration for intelligent ball selection
- Background blur/effects using segmentation mask
- Multiple ball tracking with separate output videos
- Frame-by-frame image export alongside video
