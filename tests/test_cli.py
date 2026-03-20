from clpga_demo.__main__ import build_parser, resolve_args


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
        """Default values should be None (presets supply real defaults)."""
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.smoothing_sigma is None
        assert args.smoothing_alpha is None
        assert args.model == "sam3.pt"
        assert args.confidence is None


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
        assert resolved["confidence"] == 0.15
        assert resolved["text"] == ["golf ball on green"]

    def test_confidence_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt", "--confidence", "0.5"])
        resolved = resolve_args(args)
        assert resolved["confidence"] == 0.5
        assert resolved["smoothing_sigma_seconds"] == 0.1
