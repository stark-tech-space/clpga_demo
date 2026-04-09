"""CLI entry point for clpga_demo. Run with: python -m clpga_demo"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="clpga-demo",
        description="Track a golf ball in video and output a 9:16 portrait crop.",
    )
    parser.add_argument("source", help="Input video file")
    parser.add_argument("-o", "--output", required=True, help="Output video file path")
    parser.add_argument("--preset", default="default", help="Shot preset: default, putt (default: default)")
    parser.add_argument("--smoothing-sigma", type=float, default=None, help="Gaussian sigma in seconds")
    parser.add_argument("--model", default="sam3.pt", help="SAM3 model path (default: sam3.pt)")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--momentum-history", type=int, default=None, help="Momentum tracker history size")
    parser.add_argument("--momentum-radius", type=float, default=None, help="Momentum acceptance radius scale factor")
    parser.add_argument("--confirm-frames", type=int, default=None, help="Re-acquisition confirmation frames")
    parser.add_argument("--max-size-ratio", type=float, default=None, help="Max size ratio for shape gate")
    parser.add_argument("--max-aspect-ratio", type=float, default=None, help="Max aspect ratio for shape gate")
    parser.add_argument("--tracker", default=None, choices=["momentum", "kalman"], help="Tracker type: momentum or kalman")
    parser.add_argument("--kalman-process-noise", type=float, default=None, help="Kalman process noise")
    parser.add_argument("--kalman-measurement-noise", type=float, default=None, help="Kalman measurement noise")
    parser.add_argument("--kalman-gate", type=float, default=None, help="Kalman gate threshold")
    parser.add_argument("--clean", action="store_true", default=False, help="Enable distractor removal cleaning pass")
    parser.add_argument("--corridor-multiplier", type=float, default=None, help="Corridor width multiplier (ball sizes)")
    parser.add_argument("--mask-dilation", type=int, default=None, help="Mask dilation in pixels")
    parser.add_argument("--void-model-dir", type=str, default=None, help="Path to pre-downloaded void-model")
    parser.add_argument("--clean-prompt", type=str, default=None, help="Scene description for void-model inpainting")
    parser.add_argument("--gemini-model", type=str, default=None, help="Gemini model for scene analysis")
    return parser


def resolve_args(args: argparse.Namespace) -> dict:
    """Resolve preset defaults with explicit CLI overrides."""
    from clpga_demo.presets import get_preset

    preset = get_preset(args.preset)

    cli_to_preset = {
        "smoothing_sigma": "smoothing_sigma_seconds",
        "confidence": "confidence",
        "momentum_history": "momentum_history_size",
        "momentum_radius": "momentum_radius_scale",
        "confirm_frames": "momentum_confirm_frames",
        "max_size_ratio": "momentum_max_size_ratio",
        "max_aspect_ratio": "momentum_max_aspect_ratio",
        "tracker": "tracker_type",
        "kalman_process_noise": "kalman_process_noise",
        "kalman_measurement_noise": "kalman_measurement_noise",
        "kalman_gate": "kalman_gate_threshold",
        "corridor_multiplier": "corridor_multiplier",
        "mask_dilation": "mask_dilation_px",
        "void_model_dir": "void_model_dir",
        "clean_prompt": "clean_prompt",
        "gemini_model": "gemini_model",
    }

    for cli_key, preset_key in cli_to_preset.items():
        cli_val = getattr(args, cli_key, None)
        if cli_val is not None:
            preset[preset_key] = cli_val

    # Handle boolean --clean flag separately
    if getattr(args, "clean", False):
        preset["clean"] = True

    return preset


def main() -> None:
    """Run the golf ball tracker CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    resolved = resolve_args(args)

    from clpga_demo.pipeline import process_video

    try:
        process_video(
            source=args.source,
            output=args.output,
            model=args.model,
            confidence=resolved["confidence"],
            smoothing_sigma_seconds=resolved["smoothing_sigma_seconds"],
            text=resolved["text"],
            tracker_type=resolved.get("tracker_type", "momentum"),
            momentum_history_size=resolved.get("momentum_history_size", 5),
            momentum_radius_scale=resolved.get("momentum_radius_scale", 4.0),
            momentum_confirm_frames=resolved.get("momentum_confirm_frames", 3),
            momentum_max_size_ratio=resolved.get("momentum_max_size_ratio", 2.0),
            momentum_max_aspect_ratio=resolved.get("momentum_max_aspect_ratio", 2.0),
            kalman_process_noise=resolved.get("kalman_process_noise", 1.0),
            kalman_measurement_noise=resolved.get("kalman_measurement_noise", 1.0),
            kalman_gate_threshold=resolved.get("kalman_gate_threshold", 9.0),
            clean=resolved.get("clean", False),
            corridor_multiplier=resolved.get("corridor_multiplier", 4.0),
            corridor_speed_scale=resolved.get("corridor_speed_scale", 1.5),
            mask_dilation_px=resolved.get("mask_dilation_px", 5),
            segment_max_frames=resolved.get("segment_max_frames", 180),
            segment_overlap_frames=resolved.get("segment_overlap_frames", 16),
            void_model_dir=resolved.get("void_model_dir"),
            clean_prompt=resolved.get("clean_prompt", "golf course background"),
            gemini_model=resolved.get("gemini_model", "gemini-2.5-flash-preview-05-20"),
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
