from unittest.mock import patch, MagicMock
import pytest

from clpga_demo.void_model import VoidModelWrapper


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
