from clpga_demo.__main__ import build_parser, resolve_args


class TestCLIParser:
    def test_minimal_args(self):
        """Minimum required args: source and -o output."""
        parser = build_parser()
        args = parser.parse_args(["input.mp4", "-o", "output.mp4"])
        assert args.source == "input.mp4"
        assert args.output == "output.mp4"

    def test_all_options(self):
        """All optional arguments should be parsed correctly."""
        parser = build_parser()
        args = parser.parse_args([
            "input.mp4", "-o", "output.mp4",
            "--smoothing-sigma", "1.0",
            "--model", "sam3-large.pt",
            "--confidence", "0.5",
            "--momentum-history", "3",
            "--momentum-radius", "6.0",
        ])
        assert args.smoothing_sigma == 1.0
        assert args.model == "sam3-large.pt"
        assert args.confidence == 0.5
        assert args.momentum_history == 3
        assert args.momentum_radius == 6.0

    def test_defaults(self):
        """Default values should be None (presets supply real defaults)."""
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.smoothing_sigma is None
        assert args.model == "sam3.pt"
        assert args.confidence is None
        assert args.momentum_history is None
        assert args.momentum_radius is None

    def test_no_live_flag(self):
        """--live flag should no longer exist."""
        import pytest
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["in.mp4", "-o", "out.mp4", "--live"])


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
        assert resolved["confidence"] == 0.15
        assert resolved["text"] == ["golf ball on green"]
        assert resolved["momentum_history_size"] == 5
        assert resolved["momentum_radius_scale"] == 2.0

    def test_cli_override_beats_preset(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt", "--smoothing-sigma", "0.3"])
        resolved = resolve_args(args)
        assert resolved["smoothing_sigma_seconds"] == 0.3
        assert resolved["confidence"] == 0.15

    def test_momentum_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "in.mp4", "-o", "out.mp4", "--preset", "putt",
            "--momentum-history", "3", "--momentum-radius", "6.0",
        ])
        resolved = resolve_args(args)
        assert resolved["momentum_history_size"] == 3
        assert resolved["momentum_radius_scale"] == 6.0

    def test_confidence_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--preset", "putt", "--confidence", "0.5"])
        resolved = resolve_args(args)
        assert resolved["confidence"] == 0.5
        assert resolved["smoothing_sigma_seconds"] == 0.1
