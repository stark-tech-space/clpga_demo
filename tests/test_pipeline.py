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
