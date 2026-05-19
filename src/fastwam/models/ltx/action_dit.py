"""
LTX-isomorphic ActionDiT for FastWAM-LTX MoT.

Design (post Plan-B redirect, 2026-05-19):
  - Preserve FastWAM's two-expert MoT design (small action expert paired with
    big LTX video DiT, joint self-attention with FastWAM mask).
  - Drop Wan2.2-specific block internals (Wan DiTBlock with modulation/norm1/2/3,
    Wan complex-number RoPE). Use LTX BasicAVTransformerBlock directly with
    video.dim = hidden_dim = 1024 (smaller residual stream; inner dim still
    matches LTX video's 4096 for joint-attention cat compatibility).
  - Single RoPE convention (LTX SPLIT) across both streams so joint attention
    Q.K geometry is consistent.

Public API matches the dict-style pre_dit / post_dit contract used by
fastwam.py (renamed copies live in models/ltx/fastwam.py), so the surrounding
MoT scaffolding stays drop-in.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


try:
    from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
    from ltx_core.model.transformer.rope import LTXRopeType, precompute_freqs_cis
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
    _LTX_OK = True
    _LTX_ERR = None
except Exception as _e:  # noqa: BLE001
    _LTX_OK = False
    _LTX_ERR = _e


class LTXAlignedActionDiT(nn.Module):
    """ActionDiT built from LTX primitives (BasicAVTransformerBlock with dim=hidden).

    Block structure is *identical* to LTX video block (so MoT helpers reuse one
    code path for both streams), but parameter dimensions are configured to
    keep the action expert small.
    """

    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "eps",
    )

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 32,
        attn_head_dim: int = 128,
        num_layers: int = 48,
        text_dim: int = 4096,
        eps: float = 1.0e-6,
        apply_gated_attention: bool = True,
        cross_attention_adaln: bool = True,
        rope_type: str = "split",
        timestep_scale_multiplier: int = 1000,
        positional_embedding_theta: float = 10000.0,
        use_middle_indices_grid: bool = True,
        double_precision_rope: bool = False,
        max_action_frames: int = 64,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not _LTX_OK:
            raise ImportError(
                "ltx-core unavailable; pip install -e third_party/ltx-2/packages/ltx-core. "
                f"Original error: {_LTX_ERR!r}"
            )
        if num_heads <= 0 or attn_head_dim <= 0:
            raise ValueError("num_heads and attn_head_dim must be positive")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"attn_head_dim must be even for RoPE, got {attn_head_dim}")

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.text_dim = text_dim
        self.eps = eps
        self.cross_attention_adaln = cross_attention_adaln
        self.apply_gated_attention = apply_gated_attention
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.use_middle_indices_grid = use_middle_indices_grid
        self.double_precision_rope = double_precision_rope
        self.max_action_frames = max_action_frames

        rope_enum = getattr(LTXRopeType, rope_type.upper()) if isinstance(rope_type, str) else rope_type
        self.rope_type = rope_enum

        # ---- input head: action vectors -> hidden ----
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # ---- timestep AdaLN MLPs ----
        # adaln_embedding_coefficient(cross_attention_adaln=True) = 9 in ltx-core:
        #   3 (msa: shift/scale/gate) + 3 (mlp: shift/scale/gate) + 3 (cross-attn AdaLN)
        coeff = adaln_embedding_coefficient(cross_attention_adaln)
        self.adaln_single = AdaLayerNormSingle(hidden_dim, embedding_coefficient=coeff)
        self.prompt_adaln_single = (
            AdaLayerNormSingle(hidden_dim, embedding_coefficient=2) if cross_attention_adaln else None
        )

        # ---- transformer blocks (LTX BasicAVTransformerBlock with dim=hidden_dim) ----
        video_cfg = TransformerConfig(
            dim=hidden_dim,
            heads=num_heads,
            d_head=attn_head_dim,
            context_dim=text_dim,
            apply_gated_attention=apply_gated_attention,
            cross_attention_adaln=cross_attention_adaln,
        )
        self.blocks = nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=i,
                    video=video_cfg,
                    audio=None,
                    rope_type=self.rope_type,
                    norm_eps=eps,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

        # ---- output head ----
        # scale_shift_table for final norm; head projects back to action_dim
        self.scale_shift_table = nn.Parameter(torch.empty(2, hidden_dim))
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        self.head = nn.Linear(hidden_dim, action_dim)

        # Default init for the new buffers we own (the blocks' init is handled
        # inside ltx_core; leaving it for now).
        nn.init.normal_(self.scale_shift_table, std=0.02)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    # ----- public attribute MoT consumes -----
    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.attn_head_dim

    # ----- pre_dit / post_dit: FastWAM-side contract -----
    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Build the per-block ingredients MoT needs.

        Args:
            action_tokens: (B, T, action_dim) noisy action tokens.
            timestep: (B,) batch-shared sigma.
            context: (B, T_text, text_dim) text embeddings (already in
                LTX cross-attn dim; we don't reproject).
            context_mask: (B, T_text) binary mask, 1 for valid tokens.

        Returns a dict with keys: tokens, freqs, t_mod, embedded_timestep,
        context, context_mask, scale_shift_table, meta. MoT helpers read this
        dict the same way they read video_expert.pre_dit's output.
        """
        if action_tokens.ndim != 3:
            raise ValueError(f"action_tokens must be 3D (B,T,action_dim); got {tuple(action_tokens.shape)}")
        B, T, _ = action_tokens.shape
        device = action_tokens.device
        dtype = action_tokens.dtype

        # 1) Embed action vectors to hidden_dim
        x = self.action_encoder(action_tokens)  # (B, T, hidden)

        # 2) Build positions for LTX 3D RoPE: action token i has temporal index i,
        # height=0, width=0 (action lives on the temporal axis only).
        i_idx = torch.arange(T, device=device, dtype=torch.float32)
        zero = torch.zeros(T, device=device, dtype=torch.float32)
        positions = torch.stack([i_idx, zero, zero], dim=0)  # (3, T)
        positions = positions.unsqueeze(0).expand(B, -1, -1).contiguous()      # (B, 3, T)
        positions = positions.unsqueeze(-1).expand(-1, -1, -1, 2).contiguous() # (B, 3, T, 2)

        # 3) RoPE freqs via LTX precompute_freqs_cis (same machinery as LTX video).
        # max_pos: action lives only on temporal axis, so [max_action_frames, 1, 1].
        from ltx_core.model.transformer.rope import (
            generate_freq_grid_np,
            generate_freq_grid_pytorch,
        )
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            indices_grid=positions,
            dim=self.inner_dim,
            out_dtype=dtype,
            theta=self.positional_embedding_theta,
            max_pos=[max(self.max_action_frames, T), 1, 1],
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

        # 4) Timestep embedding via adaln_single (matches LTX video's setup).
        if timestep.ndim == 0:
            timestep = timestep.expand(B)
        timestep_scaled = timestep * self.timestep_scale_multiplier
        t_mod_flat, embedded_timestep = self.adaln_single(
            timestep_scaled.flatten(), hidden_dtype=dtype
        )
        # Reshape to (B, 1, coeff*hidden) so block's get_ada_values can split.
        t_mod = t_mod_flat.view(B, -1, t_mod_flat.shape[-1])
        embedded_timestep = embedded_timestep.view(B, -1, embedded_timestep.shape[-1])

        # 5) Cross-attn AdaLN prompt timestep (if enabled)
        prompt_timestep = None
        if self.prompt_adaln_single is not None:
            sigma = timestep  # use same sigma for now
            p_flat, _ = self.prompt_adaln_single(sigma.flatten(), hidden_dtype=dtype)
            prompt_timestep = p_flat.view(B, -1, p_flat.shape[-1])

        return {
            "tokens": x,
            "freqs": pe,
            "t_mod": t_mod,
            "embedded_timestep": embedded_timestep,
            "prompt_timestep": prompt_timestep,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "T": T,
                "B": B,
                "hidden_dim": self.hidden_dim,
                "inner_dim": self.inner_dim,
                "num_action_frames": T,
            },
        }

    def post_dit(self, tokens_out: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """Apply final scale_shift + norm + head. Returns (B, T, action_dim)."""
        embedded_timestep = pre_state["embedded_timestep"]
        scale_shift = (
            self.scale_shift_table[None, None].to(device=tokens_out.device, dtype=tokens_out.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
        x = self.norm_out(tokens_out)
        x = x * (1 + scale) + shift
        return self.head(x)

    # ----- preferred construction path used by fastwam.py -----
    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: Dict[str, Any],
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LTXAlignedActionDiT":
        if not action_dit_config:
            raise ValueError("`action_dit_config` is required for LTXAlignedActionDiT.from_pretrained")
        model = cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            if not skip_dit_load_from_pretrain:
                logger.info("No action_dit_pretrained_path; ActionDiT initialized randomly.")
            return model

        # Load alpha-scale init payload (Task 10 will produce this).
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        if not p.is_file():
            raise FileNotFoundError(f"action_dit_pretrained_path not found: {p}")
        payload = torch.load(str(p), map_location="cpu")
        if "backbone_state_dict" not in payload:
            raise ValueError(f"payload missing backbone_state_dict: {p}")
        missing, unexpected = model.load_state_dict(payload["backbone_state_dict"], strict=False)
        logger.info(
            "LTXAlignedActionDiT load: missing=%d, unexpected=%d (first missing: %s)",
            len(missing), len(unexpected), missing[:3],
        )
        return model

    # ----- helpers used by Task 10 preprocess script -----
    @classmethod
    def backbone_key_set(cls, keys):
        return {k for k in keys if not any(k.startswith(p) for p in cls.ACTION_BACKBONE_SKIP_PREFIXES)}


__all__ = [
    "LTXAlignedActionDiT",
]
