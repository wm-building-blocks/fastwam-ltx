from dataclasses import dataclass
import inspect
from typing import Any

import torch
import time

from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import (
    wan_video_vae_state_dict_converter,
)
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


WAN22_MODEL_REGISTRY = [
    # UMT5-XXL text encoder (shared between Wan2.2-TI2V-5B and Wan2.2-A14B)
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    # Wan2.1-VAE (z_dim=16, used by Wan2.2-A14B series)
    {
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE,
        "model_class_kwargs": {"z_dim": 16},
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )

    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("model_class_kwargs", {}))
    model_kwargs.update(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")

    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
            "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[text_config.origin_file_pattern]
        vae_config.model_id, vae_config.origin_file_pattern = redirect_dict[vae_config.origin_file_pattern]
    return dit_config, text_config, vae_config, tokenizer_config


def load_wan22_a14b_high_noise_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-T2V-A14B",
    tokenizer_model_id: str = "Wan-AI/Wan2.2-T2V-A14B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
):
    """Load only the high-noise expert from Wan2.2-T2V-A14B + Wan2.1-VAE + UMT5-XXL."""
    logger.info("Loading Wan2.2-T2V-A14B (high-noise expert) components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-T2V-A14B loading.")
    validated_dit_config = _validate_dit_config(dit_config)

    # VAE: Wan2.1_VAE.pth at the A14B repo root (no DiffSynth redirect — safetensors not available)
    vae_config = ModelConfig(
        model_id=model_id,
        origin_file_pattern="Wan2.1_VAE.pth",
    )
    # DiT high-noise shards: directory under model_id
    dit_model_config = ModelConfig(
        model_id=model_id,
        origin_file_pattern="high_noise_model/*.safetensors",
    )
    # Text encoder: redirect to DiffSynth safetensors if available
    text_config = ModelConfig(
        model_id=model_id,
        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
    )
    tokenizer_config = ModelConfig(
        model_id=tokenizer_model_id,
        origin_file_pattern="google/umt5-xxl/",
    )
    if redirect_common_files:
        text_config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
        text_config.origin_file_pattern = "models_t5_umt5-xxl-enc-bf16.safetensors"

    vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    # DiT: multi-shard safetensors under high_noise_model/
    if skip_dit_load_from_pretrain:
        logger.info("Skipping pretrained video DiT load; random init.")
        dit = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        from .state_dict_converters import wan_video_dit_from_diffusers
        shard_files = dit_model_config.path if isinstance(dit_model_config.path, list) else [dit_model_config.path]
        if not shard_files:
            raise FileNotFoundError(f"No safetensors shards found for {model_id}/high_noise_model/")
        logger.info("Loading A14B DiT from %d shard(s): %s ...", len(shard_files), shard_files[0])
        merged_state: dict = {}
        for sf in shard_files:
            sd = load_state_dict(sf, torch_dtype=torch_dtype, device="cpu")
            merged_state.update(sd)
        merged_state = wan_video_dit_from_diffusers(merged_state)
        dit = WanVideoDiT(**validated_dit_config)
        missing, unexpected = dit.load_state_dict(merged_state, strict=False)
        logger.info(
            "DiT load: missing=%d, unexpected=%d (first missing: %s)",
            len(missing), len(unexpected), missing[:3],
        )
        dit = dit.to(device=device, dtype=torch_dtype)
        dit_path = str(shard_files[0])

    text_encoder = None
    tokenizer = None
    text_encoder_path = None
    tokenizer_path = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path, "wan_video_text_encoder",
            torch_dtype=torch_dtype, device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)

    vae = _load_registered_model(
        vae_config.path, "wan_video_vae",
        torch_dtype=torch_dtype, device=device,
    )
    logger.info("Loaded A14B high-noise components in %.2fs.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer,
        dit_path=dit_path, vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path, tokenizer_path=tokenizer_path,
    )
