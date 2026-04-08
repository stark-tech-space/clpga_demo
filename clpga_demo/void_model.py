"""Netflix void-model wrapper: download, load, and run video inpainting."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_VOID_REPO = "netflix/void-model"
_BASE_REPO = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"
_VOID_CODE_REPO = "https://github.com/netflix/void-model.git"

# Default inference settings matching VOID notebook
_SAMPLE_SIZE = (384, 672)  # (H, W)
_MAX_VIDEO_LENGTH = 197
_TEMPORAL_WINDOW_SIZE = 85
_NUM_INFERENCE_STEPS = 50
_GUIDANCE_SCALE = 1.0
_SEED = 42


class VoidModelWrapper:
    """Wraps Netflix void-model for video inpainting of distractor regions."""

    def __init__(self, model_dir: str | None, device: str = "cuda") -> None:
        self._model_dir = model_dir
        self._device = device
        self._base_model_dir: str | None = None
        self._void_dir: str | None = None
        self._code_dir: str | None = None
        self._pipe = None
        self._generator = None
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

    def _ensure_code(self) -> str:
        """Ensure the void-model code repo is available and on sys.path."""
        code_dir = Path.home() / ".cache" / "clpga" / "void-model-code"
        if not (code_dir / "videox_fun").exists():
            import subprocess

            logger.info("Cloning void-model code repo ...")
            subprocess.run(
                ["git", "clone", "--depth", "1", _VOID_CODE_REPO, str(code_dir)],
                check=True,
                env={"GIT_LFS_SKIP_SMUDGE": "1", **__import__("os").environ},
            )
        self._code_dir = str(code_dir)
        if self._code_dir not in sys.path:
            sys.path.insert(0, self._code_dir)
        return self._code_dir

    def load(self) -> VoidModelWrapper:
        """Load the void-model pipeline into GPU memory."""
        if self._loaded:
            return self

        self._ensure_code()

        base_path = self._base_model_dir or str(
            Path.home() / ".cache" / "clpga" / "CogVideoX-Fun-V1.5-5b-InP"
        )
        void_ckpt = str(Path(self._void_dir) / "void_pass1.safetensors")

        weight_dtype = torch.bfloat16

        from safetensors.torch import load_file
        from diffusers import DDIMScheduler
        from videox_fun.models import (
            AutoencoderKLCogVideoX,
            CogVideoXTransformer3DModel,
            T5EncoderModel,
            T5Tokenizer,
        )
        from videox_fun.pipeline import CogVideoXFunInpaintPipeline
        from videox_fun.utils.fp8_optimization import convert_weight_dtype_wrapper

        logger.info("Loading VAE ...")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            base_path, subfolder="vae"
        ).to(weight_dtype)

        logger.info("Loading transformer ...")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            base_path,
            subfolder="transformer",
            low_cpu_mem_usage=True,
            use_vae_mask=True,
        ).to(weight_dtype)

        logger.info("Loading VOID checkpoint from %s ...", void_ckpt)
        state_dict = load_file(void_ckpt)

        # Handle channel dimension mismatch for VAE mask
        param_name = "patch_embed.proj.weight"
        if state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1):
            latent_ch, feat_scale = 16, 8
            feat_dim = latent_ch * feat_scale
            new_weight = transformer.state_dict()[param_name].clone()
            new_weight[:, :feat_dim] = state_dict[param_name][:, :feat_dim]
            new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
            state_dict[param_name] = new_weight

        m, u = transformer.load_state_dict(state_dict, strict=False)
        logger.info("Loaded VOID weights (missing: %d, unexpected: %d)", len(m), len(u))

        tokenizer = T5Tokenizer.from_pretrained(base_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            base_path, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        scheduler = DDIMScheduler.from_pretrained(base_path, subfolder="scheduler")

        self._pipe = CogVideoXFunInpaintPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        convert_weight_dtype_wrapper(self._pipe.transformer, weight_dtype)
        self._pipe.enable_model_cpu_offload(device=self._device)
        self._generator = torch.Generator(device=self._device).manual_seed(_SEED)

        self._loaded = True
        logger.info("void-model pipeline ready.")
        return self

    def inpaint(
        self,
        video_segment: np.ndarray,
        quadmask_segment: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Run void-model inpainting on a video segment.

        Args:
            video_segment: (T, H, W, 3) uint8 BGR frames.
            quadmask_segment: (T, H, W) uint8 quadmask (0=remove, 127=affected, 255=keep).
            prompt: Text prompt describing the background scene.

        Returns:
            (T, H, W, 3) uint8 BGR cleaned frames.
        """
        if not self._loaded:
            self.load()

        T_frames, orig_H, orig_W, _ = video_segment.shape

        # Match get_video_mask_input format exactly:
        # input_video: (1, C, T, H, W) float [0, 1]
        # input_mask: (1, 1, T, H, W) float [0, 1]

        # Prepare video: BGR -> RGB, resize, (T,H,W,C) -> (C,T,H,W) -> (1,C,T,H,W)
        frames_rgb = []
        for i in range(T_frames):
            frame_rgb = cv2.cvtColor(video_segment[i], cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (_SAMPLE_SIZE[1], _SAMPLE_SIZE[0]))
            frames_rgb.append(frame_resized)
        video_np = np.stack(frames_rgb)  # (T, H, W, 3)
        video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W) [0,1]
        video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)

        # Prepare mask: resize, quantize quadmask values, invert (void-model convention),
        # then (T, H, W) -> (1, 1, T, H, W) [0, 1]
        masks_resized = []
        for i in range(T_frames):
            mask_resized = cv2.resize(
                quadmask_segment[i], (_SAMPLE_SIZE[1], _SAMPLE_SIZE[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            masks_resized.append(mask_resized)
        mask_np = np.stack(masks_resized).astype(np.float32)  # (T, H, W)

        # Quantize to quadmask values (matching void-model's get_video_mask_input)
        mask_tensor = torch.from_numpy(mask_np)
        mask_tensor = torch.where(mask_tensor <= 31, 0, mask_tensor)
        mask_tensor = torch.where((mask_tensor > 31) * (mask_tensor <= 95), 63, mask_tensor)
        mask_tensor = torch.where((mask_tensor > 95) * (mask_tensor <= 191), 127, mask_tensor)
        mask_tensor = torch.where(mask_tensor > 191, 255, mask_tensor)
        # Invert: void-model convention is 0=keep, 255=remove (opposite of our quadmask)
        mask_tensor = 255 - mask_tensor
        # Normalize to [0, 1] and reshape to (1, 1, T, H, W)
        mask_tensor = (mask_tensor / 255.0).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)

        negative_prompt = (
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. The background is solid. "
            "Strange body and strange trajectory. Distortion. "
        )

        # void-model requires num_frames == video length, and both must be 4n+1.
        # Clamp to temporal window size (85 frames) — the caller should split
        # longer videos into overlapping segments of this size.
        effective_T = min(T_frames, _TEMPORAL_WINDOW_SIZE)
        # Round down to nearest 4n+1 for VAE compatibility
        effective_T = ((effective_T - 1) // 4) * 4 + 1

        video_input = video_tensor[:, :, :effective_T]
        mask_input = mask_tensor[:, :, :effective_T]

        logger.info("Running void-model inference on %d frames ...", effective_T)
        with torch.no_grad():
            sample = self._pipe(
                prompt,
                num_frames=effective_T,
                negative_prompt=negative_prompt,
                height=_SAMPLE_SIZE[0],
                width=_SAMPLE_SIZE[1],
                generator=self._generator,
                guidance_scale=_GUIDANCE_SCALE,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                video=video_input,
                mask_video=mask_input,
                strength=1.0,
                use_trimask=True,
                use_vae_mask=True,
            ).videos

        # Convert output back to numpy BGR frames at original resolution
        # Pipeline output .videos is typically (B, C, T, H, W) in [0, 1]
        s = sample[0]  # Remove batch dim -> (C, T, H, W)
        if s.shape[0] == 3:
            # (C, T, H, W) -> (T, H, W, C)
            out_tensor = s.permute(1, 2, 3, 0)
        elif s.shape[1] == 3:
            # (T, C, H, W) -> (T, H, W, C)
            out_tensor = s.permute(0, 2, 3, 1)
        else:
            out_tensor = s

        # Denormalize from [0, 1] to [0, 255]
        out_np = (out_tensor.cpu().float().clamp(0, 1) * 255).numpy().astype(np.uint8)

        # Convert (T, H, W, C) RGB -> BGR and resize to original
        result_frames = []
        for i in range(out_np.shape[0]):
            frame_bgr = cv2.cvtColor(out_np[i], cv2.COLOR_RGB2BGR)
            if frame_bgr.shape[:2] != (orig_H, orig_W):
                frame_bgr = cv2.resize(frame_bgr, (orig_W, orig_H))
            result_frames.append(frame_bgr)

        # Pad to original frame count if void-model returned fewer frames
        while len(result_frames) < T_frames:
            result_frames.append(video_segment[len(result_frames)].copy())

        return np.stack(result_frames).astype(np.uint8)
