"""Preprocess LTXAlignedActionDiT backbone weights from LTX video DiT.

Mirror of scripts/preprocess_action_dit_backbone.py (the wan22 path), adapted
for the LTX MoT design:

  - Video expert: LTXVideoDiT (ltx-core LTXModel, hidden_dim=inner_dim=4096).
  - Action expert: LTXAlignedActionDiT — BasicAVTransformerBlock with
    hidden_dim=1024 / inner_dim=4096 / num_heads=32 / head_dim=128, plus
    action_encoder + head + AdaLN MLPs.

Mapping (action key on the left, video key on the right):
  blocks.{i}.X            <- transformer_blocks.{i}.X
  adaln_single.X          <- adaln_single.X
  prompt_adaln_single.X   <- prompt_adaln_single.X  (action.cross_attention_adaln only)
  scale_shift_table       <- scale_shift_table

Skip prefixes (kept at random init): action_encoder.*, head.*, context_proj.*
  context_proj is the Option-F text-context reprojection (text_dim -> hidden_dim);
  it has no LTX video-DiT counterpart, so it stays randomly initialized. The
  skip set is read from LTXAlignedActionDiT.ACTION_BACKBONE_SKIP_PREFIXES via
  backbone_key_set(), so this requires no code change here.

Per-tensor resize: rank is aligned by squeeze/unsqueeze, then each dim that
differs is interpolated via 1D linear (align_corners=True) in float32 -- same
recipe wan22 uses. When the *last* dim is resized, an alpha = sqrt(d_v/d_a)
scaling is applied so attention-input-magnitude is preserved on average.

Per-block `scale_shift_table` row-count mismatch:
  Video runs with cross_attention_adaln=True (coeff=9: 3 msa + 3 mlp + 3 ca).
  Action keeps cross_attention_adaln=False (coeff=6: 3 msa + 3 mlp).
  We slice the first 6 rows of the video table (drop the cross-attn AdaLN rows
  that action doesn't have) before resizing the hidden dim. If shapes already
  agree we just copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected float32/float16/bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        cur = out.shape[dim]
        if cur == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv = [0] * out.ndim
        for i, p in enumerate(perm):
            inv[p] = i
        out_p = out.permute(*perm).contiguous()
        prefix = out_p.shape[:-1]
        out_p = _interpolate_last_dim(out_p, new_size)
        out_p = out_p.reshape(*prefix, new_size)
        out = out_p.permute(*inv).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _load_model_config(path: Path) -> tuple[dict[str, Any], str | None]:
    cfg = OmegaConf.load(str(path))
    if "action_dit_config" not in cfg:
        raise ValueError(f"`{path}` must contain `action_dit_config` at top level.")
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if not isinstance(action_cfg, dict):
        raise ValueError("`action_dit_config` must resolve to a dict.")
    if _is_unresolved_interpolation(action_cfg.get("action_dim")):
        print("[WARN] `action_dit_config.action_dim` unresolved; defaulting to 14 for preprocessing.")
        action_cfg["action_dim"] = 14
    yaml_ckpt = cfg.get("ckpt_path", None)
    if yaml_ckpt is not None:
        yaml_ckpt = str(yaml_ckpt)
    return action_cfg, yaml_ckpt


def _map_action_key_to_video(key: str) -> str | None:
    """Translate an action-expert state-dict key to the matching LTXModel key.

    Returns None for keys that have no video counterpart (caller should skip).
    """
    if key.startswith("blocks."):
        return "transformer_blocks." + key[len("blocks."):]
    if key.startswith("adaln_single.") or key.startswith("prompt_adaln_single."):
        return key
    if key == "scale_shift_table":
        return key
    # norm_out / head / action_encoder / context_proj / etc. have no carryover.
    return None


def _prepare_src_tensor(
    action_key: str, src: torch.Tensor, target_shape: tuple[int, ...]
) -> torch.Tensor:
    """For per-block scale_shift_table the row count differs (video coeff=9 vs
    action coeff=6). Slice the leading rows that correspond to msa+mlp before
    the generic resize handles the hidden dim."""
    if action_key.endswith("scale_shift_table") and src.ndim >= 2:
        if src.shape[0] > target_shape[0]:
            src = src[: target_shape[0]].contiguous()
    return src


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess LTXAlignedActionDiT backbone weights from LTX video DiT."
    )
    parser.add_argument("--model-config", required=True, help="Path to model yaml, e.g. configs/model/fastwam.yaml")
    parser.add_argument("--ckpt-path", default=None,
                        help="Override ckpt_path in model yaml (LTX-2.3 safetensors).")
    parser.add_argument("--output", required=True, help="Output .pt path for preprocessed ActionDiT backbone.")
    parser.add_argument("--device", default="cpu", help="Device for tensor work (cpu is fine; we move to CPU on save).")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Apply alpha=sqrt(d_v/d_a) when the last dim is resized (true/false). Default: true.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.model_config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha = _parse_bool(args.apply_alpha_scaling)
    torch_dtype = _parse_dtype(args.dtype)

    action_cfg, yaml_ckpt = _load_model_config(cfg_path)
    int_fields = ["action_dim", "hidden_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim"]
    for k in int_fields:
        if _is_unresolved_interpolation(action_cfg.get(k)):
            raise ValueError(f"action_dit_config.{k} unresolved: {action_cfg.get(k)}")
        action_cfg[k] = int(action_cfg[k])
    action_cfg["eps"] = float(action_cfg.get("eps", 1.0e-6))
    if "cross_attention_adaln" in action_cfg:
        action_cfg["cross_attention_adaln"] = bool(action_cfg["cross_attention_adaln"])
    # Strip non-constructor knobs (yaml-only) we don't pass to LTXAlignedActionDiT.
    action_cfg.pop("use_gradient_checkpointing", None)

    ckpt_path = args.ckpt_path or yaml_ckpt
    if ckpt_path is None:
        raise ValueError("ckpt_path not in yaml and --ckpt-path not provided.")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parents[1] / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LTX checkpoint not found: {ckpt_path}")

    from fastwam.models.ltx.action_dit import LTXAlignedActionDiT
    from fastwam.models.ltx.ltx_video_dit import LTXVideoDiT

    print(f"[INFO] Loading LTX video DiT from {ckpt_path} (device={args.device}, dtype={torch_dtype}).")
    video = LTXVideoDiT.from_safetensors(str(ckpt_path), use_gradient_checkpointing=False)
    video = video.to(device=args.device, dtype=torch_dtype)
    missing, unexpected = video.load_pretrained_safetensors(str(ckpt_path), strict=False, device=args.device)
    print(f"[INFO] Loaded video DiT (missing={len(missing)}, unexpected={len(unexpected)}).")
    video_state = video._inner.state_dict()

    # Sanity: action inner_dim must match video inner_dim for joint attention.
    inner_dim_action = int(action_cfg["num_heads"]) * int(action_cfg["attn_head_dim"])
    if inner_dim_action != int(video.hidden_dim):
        raise ValueError(
            f"Joint attention requires action inner_dim == video.hidden_dim; "
            f"got action {inner_dim_action} vs video {video.hidden_dim}."
        )
    if int(action_cfg["num_layers"]) != int(len(video.blocks)):
        raise ValueError(
            f"action num_layers={action_cfg['num_layers']} must match video {len(video.blocks)}."
        )

    print(f"[INFO] Building action expert (alpha_scaling={apply_alpha}).")
    action = LTXAlignedActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
    action_state = action.state_dict()
    backbone_keys = LTXAlignedActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    no_video_counterpart: list[str] = []
    for key in sorted(backbone_keys):
        v_key = _map_action_key_to_video(key)
        if v_key is None or v_key not in video_state:
            no_video_counterpart.append(key)
            continue

        src = video_state[v_key]
        tgt = action_state[key]
        src = _prepare_src_tensor(key, src, tuple(tgt.shape))

        if tuple(src.shape) == tuple(tgt.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(tgt.shape))
            if apply_alpha and src.ndim >= 2 and src.shape[-1] != tgt.shape[-1]:
                alpha = (float(src.shape[-1]) / float(tgt.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(dtype=tgt.dtype, device="cpu").contiguous()

    payload = {
        "policy": {
            "skip_prefixes": list(LTXAlignedActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "block_key_remap": "blocks.{i}.X<-transformer_blocks.{i}.X",
            "scale_shift_table_row_slice": "first target.shape[0] rows of video table",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "eps": float(action_cfg["eps"]),
            "source_ckpt": str(ckpt_path),
            "video_inner_dim": int(video.hidden_dim),
        },
    }
    torch.save(payload, str(out_path))

    skipped_by_prefix = len(action_state) - len(backbone_keys)
    print(
        f"[INFO] Saved LTXAlignedActionDiT backbone payload to {out_path}\n"
        f"       copied={copied}, interpolated={interpolated}, "
        f"skipped_by_prefix={skipped_by_prefix}, no_video_counterpart={len(no_video_counterpart)}."
    )
    if no_video_counterpart:
        print(f"[INFO] no-counterpart action keys (random-init kept): {no_video_counterpart[:8]}"
              + ("..." if len(no_video_counterpart) > 8 else ""))


if __name__ == "__main__":
    main()
