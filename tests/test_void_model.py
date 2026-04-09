from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from clpga_demo.void_model import VoidModelWrapper


class TestInpaint:
    def test_inpaint_returns_cleaned_frames(self):
        """inpaint should return an ndarray of cleaned frames."""
        wrapper = VoidModelWrapper(model_dir="/fake/models", device="cpu")
        wrapper._void_dir = "/fake/models"
        wrapper._base_model_dir = "/fake/base"
        wrapper._loaded = True

        # Mock the pipeline
        import torch
        mock_pipe = MagicMock()
        # Pipeline returns (1, C, T, H, W) tensor in [0, 1]
        out_tensor = torch.zeros(1, 3, 5, 384, 672)
        mock_pipe.return_value = MagicMock(videos=out_tensor)
        wrapper._pipe = mock_pipe
        wrapper._generator = torch.Generator()

        video_segment = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmask_segment = np.full((10, 384, 672), 255, dtype=np.uint8)
        quadmask_segment[:, 100:150, 200:250] = 0

        result = wrapper.inpaint(video_segment, quadmask_segment, "golf course background")

        assert result.dtype == np.uint8
        assert result.shape[1:3] == (384, 672)
        assert result.shape[3] == 3
        mock_pipe.assert_called_once()

    def test_inpaint_calls_load_if_not_loaded(self):
        """inpaint should call load() if not already loaded."""
        wrapper = VoidModelWrapper(model_dir="/fake/models", device="cpu")
        wrapper._void_dir = "/fake/models"
        wrapper._base_model_dir = "/fake/base"
        wrapper._loaded = False

        with patch.object(wrapper, "load", return_value=wrapper) as mock_load:
            # After load, set up the mock pipeline
            import torch

            def setup_after_load():
                mock_pipe = MagicMock()
                out_tensor = torch.zeros(1, 3, 5, 384, 672)
                mock_pipe.return_value = MagicMock(videos=out_tensor)
                wrapper._pipe = mock_pipe
                wrapper._generator = torch.Generator()
                wrapper._loaded = True
                return wrapper

            mock_load.side_effect = setup_after_load

            video_segment = np.zeros((10, 384, 672, 3), dtype=np.uint8)
            quadmask_segment = np.full((10, 384, 672), 255, dtype=np.uint8)

            result = wrapper.inpaint(video_segment, quadmask_segment, "golf course")

        mock_load.assert_called_once()


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
