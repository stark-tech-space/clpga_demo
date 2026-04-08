"""Netflix void-model wrapper: download, load, and run video inpainting."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_VOID_REPO = "netflix/void-model"
_BASE_REPO = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"


class VoidModelWrapper:
    """Wraps Netflix void-model for video inpainting of distractor regions."""

    def __init__(self, model_dir: str | None, device: str = "cuda") -> None:
        self._model_dir = model_dir
        self._device = device
        self._base_model_dir: str | None = None
        self._void_dir: str | None = None
        self._loaded = False

    def download_if_needed(self) -> str:
        """Download void-model and base model from HF if model_dir is not set.

        Returns the directory containing the void-model checkpoints.
        """
        if self._model_dir is not None:
            self._void_dir = self._model_dir
            return self._model_dir

        logger.info("Downloading base model from %s ...", _BASE_REPO)
        self._base_model_dir = snapshot_download(
            repo_id=_BASE_REPO,
            local_dir=str(Path.home() / ".cache" / "clpga" / "CogVideoX-Fun-V1.5-5b-InP"),
        )

        logger.info("Downloading void-model from %s ...", _VOID_REPO)
        self._void_dir = snapshot_download(
            repo_id=_VOID_REPO,
            local_dir=str(Path.home() / ".cache" / "clpga" / "void-model"),
        )

        return self._void_dir
