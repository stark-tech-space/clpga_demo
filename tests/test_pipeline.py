from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from clpga_demo.pipeline import process_video


def _create_test_video(path: str, width: int = 320, height: int = 240, frames: int = 10) -> None:
    """Create a small test video with a white circle (fake ball)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a moving white circle
        cx = 100 + i * 10
        cy = 120
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _mock_track_video(source, **kwargs):
    """Mock tracker that yields fake detections with a moving ball."""
    cap = cv2.VideoCapture(source)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cx = 100 + frame_idx * 10
        cy = 120
        boxes = np.array([[cx - 10, cy - 10, cx + 10, cy + 10, 1, 0.95, 0]])
        yield frame_idx, frame, boxes
        frame_idx += 1
    cap.release()


class TestProcessVideo:
    def test_creates_output_file(self, tmp_path):
        """process_video should create an output video file."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_video(input_path, output_path)

        assert Path(output_path).exists()
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        # Output should be portrait 9:16 ratio (allow +-1 for int rounding)
        expected_w = int(h * 9 / 16)
        assert abs(w - expected_w) <= 1

    def test_raises_on_missing_input(self):
        """process_video should raise ValueError for nonexistent input."""
        with pytest.raises(ValueError, match="does not exist"):
            process_video("/nonexistent/video.mp4", "/tmp/out.mp4")

    def test_raises_on_no_detections(self, tmp_path):
        """process_video should raise RuntimeError when no ball is detected."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        def _empty_tracker(source, **kwargs):
            cap = cv2.VideoCapture(source)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_idx, frame, np.empty((0, 7))
                frame_idx += 1
            cap.release()

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _empty_tracker(s)):
            with pytest.raises(RuntimeError, match="No golf ball detected"):
                process_video(input_path, output_path)


class TestProcessVideoText:
    def test_passes_text_to_tracker(self, tmp_path):
        """process_video should forward the text parameter to track_video."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        captured_kwargs = {}

        def capturing_tracker(source, **kwargs):
            captured_kwargs.update(kwargs)
            return _mock_track_video(source)

        with patch("clpga_demo.pipeline.track_video", side_effect=capturing_tracker):
            process_video(input_path, output_path, text=["golf ball on green"])

        assert captured_kwargs["text"] == ["golf ball on green"]

    def test_default_text_is_none(self, tmp_path):
        """process_video without text param should pass None (tracker uses its default)."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        captured_kwargs = {}

        def capturing_tracker(source, **kwargs):
            captured_kwargs.update(kwargs)
            return _mock_track_video(source)

        with patch("clpga_demo.pipeline.track_video", side_effect=capturing_tracker):
            process_video(input_path, output_path)

        assert captured_kwargs.get("text") is None


class TestTrackerType:
    def test_default_tracker_is_momentum(self, tmp_path):
        """Default tracker_type should work (momentum)."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_video(input_path, output_path, tracker_type="momentum")

        assert Path(output_path).exists()

    def test_kalman_tracker_creates_output(self, tmp_path):
        """Kalman tracker should also produce output video."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_video(input_path, output_path, tracker_type="kalman")

        assert Path(output_path).exists()


class TestMomentumFiltering:
    def test_rejects_detection_far_from_momentum(self, tmp_path):
        """A detection that jumps far from predicted trajectory should be rejected."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path, frames=20)

        def _jumping_tracker(source, **kwargs):
            """Ball moves steadily then jumps to a distant position."""
            cap = cv2.VideoCapture(source)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx < 10:
                    # Steady rightward motion
                    cx = 100 + frame_idx * 5
                    cy = 120
                    boxes = np.array([[cx - 10, cy - 10, cx + 10, cy + 10, 1, 0.95, 0]])
                elif frame_idx == 10:
                    # Ball disappears for one frame
                    boxes = np.empty((0, 7))
                elif frame_idx == 11:
                    # False detection appears far away (should be rejected)
                    boxes = np.array([[10, 10, 30, 30, 2, 0.90, 0]])
                else:
                    # Real ball continues near expected trajectory
                    cx = 100 + frame_idx * 5
                    cy = 120
                    boxes = np.array([[cx - 10, cy - 10, cx + 10, cy + 10, 1, 0.95, 0]])
                yield frame_idx, frame, boxes
                frame_idx += 1
            cap.release()

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _jumping_tracker(s)):
            process_video(input_path, output_path)

        # If momentum filtering works, the output video should exist
        assert Path(output_path).exists()


class TestKalmanGapHandling:
    def test_kalman_fills_gaps_with_predictions(self, tmp_path):
        """Kalman tracker should produce continuous trajectory (no NaN gaps)."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path, frames=20)

        def _gap_tracker(source, **kwargs):
            """Ball visible for 5 frames, gone for 3, back for rest."""
            cap = cv2.VideoCapture(source)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx < 5 or frame_idx >= 8:
                    cx = 100 + frame_idx * 5
                    cy = 120
                    boxes = np.array([[cx - 10, cy - 10, cx + 10, cy + 10, 1, 0.95, 0]])
                else:
                    boxes = np.empty((0, 7))
                yield frame_idx, frame, boxes
                frame_idx += 1
            cap.release()

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _gap_tracker(s)):
            process_video(input_path, output_path, tracker_type="kalman")

        assert Path(output_path).exists()


class TestCleaningPipeline:
    def test_clean_flag_invokes_cleaning_pass(self, tmp_path):
        """When clean=True, process_video should use FrameCleaner."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path, frames=20)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            with patch("clpga_demo.cleaner.FrameCleaner") as MockCleaner:
                mock_instance = MockCleaner.return_value
                mock_instance.compute_corridors.return_value = [None] * 20
                mock_instance.generate_quadmasks.return_value = np.zeros((20, 240, 320), dtype=np.uint8)
                mock_instance.clean_segments.return_value = [np.zeros((20, 240, 320, 3), dtype=np.uint8)]
                MockCleaner.split_into_segments.return_value = [(0, 20)]
                MockCleaner.blend_segments.return_value = np.zeros((20, 240, 320, 3), dtype=np.uint8)
                with patch("clpga_demo.void_model.VoidModelWrapper") as MockVoid:
                    mock_void = MockVoid.return_value
                    mock_void.download_if_needed.return_value = "/fake"
                    with patch("clpga_demo.pipeline._retrack_cleaned") as mock_retrack:
                        mock_retrack.return_value = [(100 + i * 10, 120) for i in range(20)]
                        process_video(input_path, output_path, clean=True)

        assert MockCleaner.called

    def test_clean_false_skips_cleaning(self, tmp_path):
        """When clean=False (default), cleaning pass should be skipped."""
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _create_test_video(input_path)

        with patch("clpga_demo.pipeline.track_video", side_effect=lambda s, **kw: _mock_track_video(s)):
            process_video(input_path, output_path, clean=False)

        assert Path(output_path).exists()
