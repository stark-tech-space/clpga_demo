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
    parser.add_argument("source", help="Input video file or RTSP stream URL")
    parser.add_argument("-o", "--output", required=True, help="Output video file path")
    parser.add_argument("--live", action="store_true", help="Treat source as a live stream (EMA smoothing)")
    parser.add_argument("--preset", default="default", help="Shot preset: default, putt (default: default)")
    parser.add_argument("--smoothing-sigma", type=float, default=None, help="Gaussian sigma in seconds (file mode)")
    parser.add_argument("--smoothing-alpha", type=float, default=None, help="EMA alpha (live mode)")
    parser.add_argument("--model", default="sam3.pt", help="SAM3 model path (default: sam3.pt)")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
    return parser


def resolve_args(args: argparse.Namespace) -> dict:
    """Resolve preset defaults with explicit CLI overrides."""
    from clpga_demo.presets import get_preset

    preset = get_preset(args.preset)

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


def main() -> None:
    """Run the golf ball tracker CLI."""
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


if __name__ == "__main__":
    main()
