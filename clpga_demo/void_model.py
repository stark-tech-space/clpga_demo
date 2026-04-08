"""Netflix void-model wrapper: download, load, and run video inpainting."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
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

    def inpaint(
        self,
        video_segment: np.ndarray,
        quadmask_segment: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Run void-model inpainting on a video segment.

        Args:
            video_segment: (T, H, W, 3) uint8 RGB frames.
            quadmask_segment: (T, H, W) uint8 mask (255=keep, 0=inpaint).
            prompt: Text prompt describing the background scene.

        Returns:
            (T, H, W, 3) uint8 cleaned frames.
        """
        T, H, W, _ = video_segment.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30.0

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_dir = Path(tmpdir) / "seq"
            seq_dir.mkdir()
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Write input_video.mp4
            input_video_path = str(seq_dir / "input_video.mp4")
            writer = cv2.VideoWriter(input_video_path, fourcc, fps, (W, H))
            for frame in video_segment:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            # Write quadmask_0.mp4 (grayscale written as BGR single-channel video)
            mask_video_path = str(seq_dir / "quadmask_0.mp4")
            mask_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (W, H), isColor=False)
            for mask_frame in quadmask_segment:
                mask_writer.write(mask_frame)
            mask_writer.release()

            # Write prompt.json
            prompt_path = seq_dir / "prompt.json"
            prompt_path.write_text(json.dumps({"bg": prompt}))

            # Build CLI command
            void_dir = self._void_dir
            base_model_dir = self._base_model_dir or ""
            cmd = [
                "python",
                f"{void_dir}/inference/cogvideox_fun/predict_v2v.py",
                f"--config={void_dir}/config/quadmask_cogvideox.py",
                f"--config.data.data_rootdir={tmpdir}",
                "--config.experiment.run_seqs=seq",
                f"--config.experiment.save_path={str(output_dir)}",
                f"--config.video_model.transformer_path={void_dir}/void_pass1.safetensors",
                f"--config.video_model.model_name={base_model_dir}",
            ]

            logger.info("Running void-model inference ...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"void-model inference failed (exit {result.returncode}): {result.stderr}"
                )

            # Discover output video
            output_files = list(Path(output_dir).rglob("*.mp4"))
            if not output_files:
                raise RuntimeError("void-model produced no output video files.")
            output_video_path = str(output_files[0])

            # Read output frames
            cap = cv2.VideoCapture(output_video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open output video: {output_video_path}")

            frames: list[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        if not frames:
            raise RuntimeError("No frames read from void-model output video.")

        out = np.stack(frames, axis=0)  # (T', H', W', 3)

        # Resize to original dimensions if needed
        _T, out_H, out_W, _ = out.shape
        if out_H != H or out_W != W:
            resized = np.empty((len(frames), H, W, 3), dtype=np.uint8)
            for i, f in enumerate(frames):
                resized[i] = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
            out = resized

        return out.astype(np.uint8)
