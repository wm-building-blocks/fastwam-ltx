"""LTX-2 video-only component loader (Task 7 / 2026-05-19).

Single entry point ``load_ltx2_video_only_components`` builds + (optionally)
weight-loads the three FastWAM-side adapters from one LTX-2.3 safetensors
file plus a Gemma-3-12B folder. All Wan-specific plumbing (DiffSynth redirect,
multi-shard merging, ModelConfig.download_if_necessary) is removed: LTX-2.3
ships everything in one ``ltx-2.3-22b-dev.safetensors`` file.

Weight provenance summary (see Tasks 4/5/6 for details):
    model.diffusion_model.*  -> LTXVideoDiT (after audio key filter)
    vae.encoder.*            -> LTXVideoVAE.encoder
    vae.decoder.*            -> LTXVideoVAE.decoder
    vae.per_channel_statistics.* -> per_channel_statistics.*
    text_embedding_projection.* -> LTXTextEncoder.embeddings_processor.feature_extractor.*
    model.diffusion_model.video_embeddings_connector.*
                             -> LTXTextEncoder.embeddings_processor.video_connector.*
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from fastwam.utils.logging_config import get_logger

from ..ltx_video_dit import LTXVideoDiT, load_ltx_config_from_safetensors
from ..ltx_video_vae import LTXVideoVAE
from ..ltx_text_encoder import LTXTextEncoder

logger = get_logger(__name__)

SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"

DEFAULT_LTX_CKPT_PATH = "checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_PATH = "checkpoints/google/gemma-3-12b-it-qat-q4_0-unquantized"


@dataclass
class LTXLoadedComponents:
    """Bundle of loaded LTX components ready to be plugged into FastWAM."""
    dit: LTXVideoDiT
    vae: LTXVideoVAE
    text_encoder: Optional[LTXTextEncoder]
    config: Dict[str, Any]
    dit_path: str
    vae_path: str
    text_encoder_path: Optional[str]


def _resolve(path: str, root_hint: Optional[str] = None) -> str:
    """Resolve a path. Accepts absolute paths verbatim; relative paths are
    looked up under (a) cwd, (b) ``root_hint`` if provided. Raises
    ``FileNotFoundError`` if no candidate exists."""
    if os.path.isabs(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    candidates = [path]
    if root_hint:
        candidates.append(os.path.join(root_hint, path))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"could not locate {path!r}; tried: {candidates}")


def load_ltx2_video_only_components(
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    ckpt_path: str = DEFAULT_LTX_CKPT_PATH,
    gemma_path: str = DEFAULT_GEMMA_PATH,
    load_text_encoder: bool = False,
    attach_gemma_to_text_encoder: bool = False,
    skip_dit_load_from_pretrain: bool = False,
    use_gradient_checkpointing: bool = False,
    project_root: Optional[str] = None,
) -> LTXLoadedComponents:
    """Load LTX-2.3 video-only DiT + VAE (+ optional text encoder).

    Args:
        device: target torch device for the DiT and VAE. ``"cpu"`` is supported
            (slow but useful for testing).
        torch_dtype: model parameter dtype. ``bfloat16`` is the upstream default.
        ckpt_path: path to ``ltx-2.3-22b-dev.safetensors`` (single-file
            checkpoint that bundles DiT, VAE, and embeddings-connector weights).
        gemma_path: HF-format folder for Gemma-3-12B-it. Only used when
            ``attach_gemma_to_text_encoder=True``.
        load_text_encoder: build LTXTextEncoder (EmbeddingsProcessor) and load
            its weights from the same ``ckpt_path``. Note: this does *not*
            load Gemma itself (12B weights). Pass ``attach_gemma_to_text_encoder=True``
            to additionally bring Gemma into VRAM (only do this for cache
            precomputation; the training loop should read pre-computed
            embeddings from disk).
        attach_gemma_to_text_encoder: implies ``load_text_encoder=True``.
        skip_dit_load_from_pretrain: random-init the DiT (debug only).
        use_gradient_checkpointing: pass-through to LTXVideoDiT.
        project_root: if set, used as a fallback root for resolving relative
            ``ckpt_path`` / ``gemma_path``. Typically the FastWAM_LTX repo dir.
    """
    if attach_gemma_to_text_encoder:
        load_text_encoder = True

    ckpt_path = _resolve(ckpt_path, project_root)
    logger.info("Loading LTX-2.3 components from %s (device=%s, dtype=%s)",
                ckpt_path, device, torch_dtype)
    start = time.time()

    # 1) Parse safetensors metadata once.
    config = load_ltx_config_from_safetensors(ckpt_path)

    # 2) DiT
    if skip_dit_load_from_pretrain:
        logger.info("Skipping DiT pretrained load (random init).")
        dit = LTXVideoDiT(config=config, use_gradient_checkpointing=use_gradient_checkpointing)
        dit = dit.to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        logger.info("Building LTXVideoDiT + loading pretrained weights...")
        dit = LTXVideoDiT(config=config, use_gradient_checkpointing=use_gradient_checkpointing)
        dit = dit.to(dtype=torch_dtype)
        missing, unexpected = dit.load_pretrained_safetensors(
            ckpt_path, strict=False, device="cpu"
        )
        if missing:
            raise RuntimeError(
                f"LTX DiT missing {len(missing)} keys after load: {missing[:5]}"
            )
        if unexpected:
            raise RuntimeError(
                f"LTX DiT has {len(unexpected)} unexpected keys after load: {unexpected[:5]}"
            )
        dit = dit.to(device=device)
        dit_path = ckpt_path

    # 3) VAE (same ckpt, different prefixes)
    logger.info("Building LTXVideoVAE + loading pretrained weights...")
    vae = LTXVideoVAE.from_config(config, build_encoder=True, build_decoder=True)
    vae = vae.to(dtype=torch_dtype)
    vae_result = vae.load_pretrained_safetensors(ckpt_path, strict=False, device="cpu")
    for sub, (m, u) in vae_result.items():
        if m or u:
            raise RuntimeError(
                f"LTX VAE {sub}: missing={len(m)} ({m[:3]}), unexpected={len(u)} ({u[:3]})"
            )
    vae = vae.to(device=device)
    vae_path = ckpt_path

    # 4) Text encoder (optional)
    text_encoder: Optional[LTXTextEncoder] = None
    text_encoder_path: Optional[str] = None
    if load_text_encoder:
        logger.info("Building LTXTextEncoder + loading EmbeddingsProcessor weights...")
        text_encoder = LTXTextEncoder.from_config(config).to(dtype=torch_dtype)
        m, u = text_encoder.load_pretrained_safetensors(ckpt_path, strict=False)
        if m or u:
            raise RuntimeError(
                f"LTXTextEncoder: missing={len(m)} ({m[:3]}), unexpected={len(u)} ({u[:3]})"
            )
        text_encoder = text_encoder.to(device=device)
        text_encoder_path = ckpt_path
        if attach_gemma_to_text_encoder:
            gemma_path = _resolve(gemma_path, project_root)
            logger.info("Attaching Gemma from %s ...", gemma_path)
            text_encoder.attach_gemma(gemma_path, torch_dtype=torch_dtype)
            if device != "cpu":
                text_encoder.gemma = text_encoder.gemma.to(device)

    logger.info("Loaded LTX-2.3 components in %.1fs", time.time() - start)
    return LTXLoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        config=config,
        dit_path=dit_path,
        vae_path=vae_path,
        text_encoder_path=text_encoder_path,
    )


__all__ = [
    "load_ltx2_video_only_components",
    "LTXLoadedComponents",
    "SKIPPED_PRETRAIN_SENTINEL",
    "DEFAULT_LTX_CKPT_PATH",
    "DEFAULT_GEMMA_PATH",
]
