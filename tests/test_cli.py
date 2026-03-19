from clpga_demo.__main__ import build_parser


class TestCLIParser:
    def test_minimal_args(self):
        """Minimum required args: source and -o output."""
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "-o", "output.mp4"])
        assert args.source == "input.mp4"
        assert args.output == "output.mp4"
        assert args.live is False

    def test_live_flag(self):
        """--live flag should be parsed."""
        parser = build_parser()
        args = parser.parse_args(["rtsp://cam/stream", "-o", "out.mp4", "--live"])
        assert args.live is True

    def test_all_options(self):
        """All optional arguments should be parsed correctly."""
        parser = build_parser()
        args = parser.parse_args([
            "input.mp4", "-o", "output.mp4",
            "--smoothing-sigma", "1.0",
            "--smoothing-alpha", "0.2",
            "--model", "sam3-large.pt",
            "--confidence", "0.5",
        ])
        assert args.smoothing_sigma == 1.0
        assert args.smoothing_alpha == 0.2
        assert args.model == "sam3-large.pt"
        assert args.confidence == 0.5

    def test_defaults(self):
        """Default values should match spec."""
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.smoothing_sigma == 0.5
        assert args.smoothing_alpha == 0.15
        assert args.model == "sam3.pt"
        assert args.confidence == 0.25
