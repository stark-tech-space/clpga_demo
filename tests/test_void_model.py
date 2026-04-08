from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
from pathlib import Path
import pytest

from clpga_demo.void_model import VoidModelWrapper


class TestInpaint:
    def test_inpaint_returns_cleaned_frames(self):
        """inpaint should return an ndarray of cleaned frames with same shape as input."""
        wrapper = VoidModelWrapper(model_dir="/fake/models", device="cpu")
        wrapper._void_dir = "/fake/models"
        wrapper._base_model_dir = "/fake/base"

        video_segment = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmask_segment = np.full((10, 384, 672), 255, dtype=np.uint8)
        quadmask_segment[:, 100:150, 200:250] = 0

        with patch("clpga_demo.void_model.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("clpga_demo.void_model.cv2.VideoCapture") as mock_cap:
                mock_cap_instance = MagicMock()
                mock_cap.return_value = mock_cap_instance
                mock_cap_instance.isOpened.return_value = True
                frames_returned = [True] * 10 + [False]
                mock_cap_instance.read.side_effect = [
                    (ok, np.zeros((384, 672, 3), dtype=np.uint8)) if ok else (False, None)
                    for ok in frames_returned
                ]
                with patch("clpga_demo.void_model.Path") as mock_path_cls:
                    mock_path_instance = MagicMock()
                    mock_path_cls.return_value = mock_path_instance
                    mock_path_instance.rglob.return_value = [Path("/fake/output/seq/output.mp4")]
                    result = wrapper.inpaint(video_segment, quadmask_segment, "golf course background")

        assert result.shape == video_segment.shape
        assert result.dtype == np.uint8

    def test_inpaint_raises_on_subprocess_failure(self):
        """inpaint should raise RuntimeError if void-model process fails."""
        wrapper = VoidModelWrapper(model_dir="/fake/models", device="cpu")
        wrapper._void_dir = "/fake/models"
        wrapper._base_model_dir = "/fake/base"

        video_segment = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmask_segment = np.full((10, 384, 672), 255, dtype=np.uint8)

        with patch("clpga_demo.void_model.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="CUDA OOM")
            with pytest.raises(RuntimeError, match="void-model inference failed"):
                wrapper.inpaint(video_segment, quadmask_segment, "golf course background")


class TestDownload:
    def test_download_if_needed_calls_snapshot_download(self):
        """download_if_needed should call snapshot_download for both repos."""
        wrapper = VoidModelWrapper(model_dir=None, device="cpu")

        with patch("clpga_demo.void_model.snapshot_download") as mock_dl:
            mock_dl.return_value = "/fake/path"
            result = wrapper.download_if_needed()

        assert mock_dl.call_count == 2
        repo_ids = [call.kwargs.get("repo_id") or call.args[0] for call in mock_dl.call_args_list]
        assert "netflix/void-model" in repo_ids
        assert "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP" in repo_ids

    def test_download_skipped_when_model_dir_provided(self):
        """When model_dir is explicitly set, skip download."""
        wrapper = VoidModelWrapper(model_dir="/existing/models", device="cpu")

        with patch("clpga_demo.void_model.snapshot_download") as mock_dl:
            result = wrapper.download_if_needed()

        mock_dl.assert_not_called()
        assert result == "/existing/models"
