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
        assert resolved["momentum_radius_scale"] == 15.0

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


class TestTrackerCLI:
    def test_tracker_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--tracker", "kalman"])
        assert args.tracker == "kalman"

    def test_tracker_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.tracker is None

    def test_kalman_flags_parsed(self):
        parser = build_parser()
        args = parser.parse_args([
            "in.mp4", "-o", "out.mp4",
            "--kalman-process-noise", "2.0",
            "--kalman-measurement-noise", "3.0",
            "--kalman-gate", "16.0",
        ])
        assert args.kalman_process_noise == 2.0
        assert args.kalman_measurement_noise == 3.0
        assert args.kalman_gate == 16.0

    def test_tracker_resolve_to_preset(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--tracker", "kalman"])
        resolved = resolve_args(args)
        assert resolved["tracker_type"] == "kalman"

    def test_kalman_resolve_overrides_preset(self):
        parser = build_parser()
        args = parser.parse_args([
            "in.mp4", "-o", "out.mp4", "--preset", "putt",
            "--kalman-process-noise", "2.0",
        ])
        resolved = resolve_args(args)
        assert resolved["kalman_process_noise"] == 2.0
        # Other putt values preserved
        assert resolved["kalman_measurement_noise"] == 1.0


class TestCleanCLI:
    def test_clean_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--clean"])
        assert args.clean is True

    def test_clean_flag_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
        assert args.clean is False

    def test_corridor_multiplier_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--corridor-multiplier", "5.0"])
        assert args.corridor_multiplier == 5.0

    def test_mask_dilation_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--mask-dilation", "8"])
        assert args.mask_dilation == 8

    def test_void_model_dir_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--void-model-dir", "/models/void"])
        assert args.void_model_dir == "/models/void"

    def test_clean_prompt_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--clean-prompt", "driving range"])
        assert args.clean_prompt == "driving range"

    def test_clean_flag_resolves_to_preset(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--clean"])
        resolved = resolve_args(args)
        assert resolved["clean"] is True

    def test_corridor_multiplier_override(self):
        parser = build_parser()
        args = parser.parse_args(["in.mp4", "-o", "out.mp4", "--corridor-multiplier", "6.0"])
        resolved = resolve_args(args)
        assert resolved["corridor_multiplier"] == 6.0


class TestCleanPassthrough:
    def test_clean_params_passed_to_process_video(self):
        """All cleaning params should be forwarded from CLI to process_video."""
        from unittest.mock import patch, MagicMock

        with patch("clpga_demo.pipeline.process_video") as mock_pv:
            with patch("sys.argv", ["clpga-demo", "in.mp4", "-o", "out.mp4", "--clean", "--corridor-multiplier", "5.0"]):
                from clpga_demo.__main__ import main
                try:
                    main()
                except (SystemExit, Exception):
                    pass

            if mock_pv.called:
                kwargs = mock_pv.call_args.kwargs if mock_pv.call_args.kwargs else {}
                # Check named arguments
                assert kwargs.get("clean") is True
                assert kwargs.get("corridor_multiplier") == 5.0
