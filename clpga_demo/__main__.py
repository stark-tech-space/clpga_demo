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
    parser.add_argument("--smoothing-sigma", type=float, default=0.5, help="Gaussian sigma in seconds (file mode, default: 0.5)")
    parser.add_argument("--smoothing-alpha", type=float, default=0.15, help="EMA alpha (live mode, default: 0.15)")
    parser.add_argument("--model", default="sam3.pt", help="SAM3 model path (default: sam3.pt)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    return parser


def main() -> None:
    """Run the golf ball tracker CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    from clpga_demo.pipeline import process_stream, process_video

    try:
        if args.live:
            process_stream(
                source=args.source,
                output=args.output,
                model=args.model,
                confidence=args.confidence,
                smoothing_alpha=args.smoothing_alpha,
            )
        else:
            process_video(
                source=args.source,
                output=args.output,
                model=args.model,
                confidence=args.confidence,
                smoothing_sigma_seconds=args.smoothing_sigma,
            )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
