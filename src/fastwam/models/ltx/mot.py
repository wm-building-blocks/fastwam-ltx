"""
LTX joint-MoT (FastWAM-LTX, post Plan-B redirect 2026-05-19).

Preserves FastWAM's two-expert joint self-attention design and the original
``_build_mot_attention_mask`` (video doesn't see action; action sees first-frame
video + all action). Block-internal API is uniformly LTX (BasicAVTransformerBlock
+ LTX SPLIT RoPE + apply_gated_attention + cross_attention_adaln) on BOTH the
video and action streams, eliminating Wan/LTX RoPE incompatibility and
collapsing the MoT helpers to a single code path.

FastWAM-compatible MoT.forward signature (drop-in for the existing dict-style
contract used by ``fastwam.py``):

    mot(
        embeds_all={"video": (B,Tv,4096), "action": (B,Ta,1024)},
        attention_mask=(Tv+Ta, Tv+Ta) bool   # built by FastWAM._build_mot_attention_mask
        freqs_all={"video": (cos_v, sin_v), "action": (cos_a, sin_a)},  # LTX rope tuples
        context_all={"video": {"context","mask"}, "action": {"context","mask"}},
        t_mod_all={"video": (B,Tv,coeff*4096), "action": (B,1,coeff*1024)},
    )

Inference prefill (``prefill_video_cache`` / ``forward_action_with_video_cache``)
is deferred — Plan B §7 marks it out-of-scope for the first FSDP smoke. The
training joint path is sufficient to validate end-to-end.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


try:
    from ltx_core.model.transformer.rope import apply_rotary_emb
    from ltx_core.utils import rms_norm
    _LTX_OK = True
    _LTX_ERR = None
except Exception as _e:  # noqa: BLE001
    _LTX_OK = False
    _LTX_ERR = _e


# ============================================================================
# Helpers — single API (both video and action experts use LTX block structure)
# ============================================================================


def _get_ada_values_ltx(
    block, t_mod: torch.Tensor, slice_range: slice
) -> Tuple[torch.Tensor, ...]:
    """Slice AdaLN-Zero ada values from ``block.scale_shift_table + t_mod``.

    Mirrors ``BasicAVTransformerBlock.get_ada_values`` semantics. Returns a
    tuple of (shift, scale, gate) when slice_range = slice(0,3); 3 tensors in
    general. Output shape is (B, T_or_1, dim) per element.
    """
    B = t_mod.shape[0]
    sst = block.scale_shift_table
    num_ada_params = sst.shape[0]
    inner_dim = sst.shape[1]
    # t_mod shape: (B, T_or_1, num_ada_params * inner_dim)
    ada = (
        sst[slice_range].unsqueeze(0).unsqueeze(0).to(device=t_mod.device, dtype=t_mod.dtype)
        + t_mod.reshape(B, t_mod.shape[1], num_ada_params, -1)[:, :, slice_range, :]
    ).unbind(dim=2)
    return ada


def _build_expert_attention_io_ltx(
    block,
    x: torch.Tensor,
    pe: Tuple[torch.Tensor, torch.Tensor],
    t_mod: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run a block's pre-attention path manually, stopping after Q/K/V + RoPE.

    Works for both video and action experts (both are LTX BasicAVTransformerBlock).
    Returns a dict with q/k/v ready for joint attention concat, plus the
    intermediate signals needed in the post-block step.
    """
    # MSA ada values (slice 0:3) — shift, scale, gate
    vshift_msa, vscale_msa, vgate_msa = _get_ada_values_ltx(block, t_mod, slice(0, 3))
    # MLP ada values (slice 3:6) — shift, scale, gate
    vshift_mlp, vscale_mlp, vgate_mlp = _get_ada_values_ltx(block, t_mod, slice(3, 6))

    # Pre-attention RMSNorm + AdaLN modulate (LTX style)
    norm_x = rms_norm(x, eps=block.norm_eps) * (1 + vscale_msa) + vshift_msa

    # Q/K/V projections + RMSNorm
    q = block.attn1.to_q(norm_x)
    k = block.attn1.to_k(norm_x)
    v = block.attn1.to_v(norm_x)
    q = block.attn1.q_norm(q)
    k = block.attn1.k_norm(k)

    # LTX SPLIT RoPE — same for both streams (no Wan/LTX rope mismatch)
    q = apply_rotary_emb(q, pe, block.attn1.rope_type)
    k = apply_rotary_emb(k, pe, block.attn1.rope_type)

    return {
        "q": q, "k": k, "v": v,
        "residual_x": x,
        "norm_x": norm_x,           # needed for to_gate_logits gating
        "gate_msa": vgate_msa,
        "shift_mlp": vshift_mlp,
        "scale_mlp": vscale_mlp,
        "gate_mlp": vgate_mlp,
    }


def _apply_expert_post_block_ltx(
    block,
    io_state: Dict[str, torch.Tensor],
    mixed_attn_out: torch.Tensor,
    t_mod: torch.Tensor,
    context: Optional[torch.Tensor],
    context_mask: Optional[torch.Tensor],
    prompt_timestep: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply per-head gating + to_out + text cross-attn + FF for one stream's
    output from the joint attention call.
    """
    norm_x = io_state["norm_x"]
    x_residual = io_state["residual_x"]

    out = mixed_attn_out
    # Per-head attention gating (apply_gated_attention=True paths)
    if block.attn1.to_gate_logits is not None:
        gate_logits = block.attn1.to_gate_logits(norm_x)  # (B, T, heads)
        B, T, _ = out.shape
        heads = block.attn1.heads
        dim_head = block.attn1.dim_head
        out = out.view(B, T, heads, dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(B, T, heads * dim_head)

    # to_out projection (Sequential(Linear, Identity))
    out = block.attn1.to_out(out)

    # Residual + per-token MSA gate
    x = x_residual + out * io_state["gate_msa"]

    # Text cross-attention (block.attn2). Use the LTX block's own helper to
    # keep the cross_attention_adaln pathway identical.
    if context is not None:
        x = x + block._apply_text_cross_attention(
            x,
            context,
            block.attn2,
            block.scale_shift_table,
            getattr(block, "prompt_scale_shift_table", None),
            t_mod,
            prompt_timestep,
            context_mask,
            cross_attention_adaln=block.cross_attention_adaln,
        )

    # Feed-forward (AdaLN + ff + per-token MLP gate)
    ff_in = rms_norm(x, eps=block.norm_eps) * (1 + io_state["scale_mlp"]) + io_state["shift_mlp"]
    x = x + block.ff(ff_in) * io_state["gate_mlp"]

    return x


def _fastwam_mask_to_additive(
    bool_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Convert FastWAM's 2-D bool mask (V+A, V+A) into the (1, 1, S, S) additive
    log-bias form that ``F.scaled_dot_product_attention`` accepts as ``attn_mask``."""
    if bool_mask.dtype != torch.bool:
        return bool_mask  # already additive
    additive = torch.zeros_like(bool_mask, dtype=dtype)
    additive[~bool_mask] = float("-inf")
    return additive.unsqueeze(0).unsqueeze(0)


# ============================================================================
# MoTBlock — per-layer joint forward
# ============================================================================


class MoTBlock(nn.Module):
    """Pairs one LTX video block with one LTX action block; FSDP wrap unit."""

    def __init__(
        self,
        mot: "MoT",
        video_block: nn.Module,
        action_block: nn.Module,
        do_ckpt: bool,
        expert_use_gc_video: bool,
        expert_use_gc_action: bool,
    ) -> None:
        super().__init__()
        # Stash via object.__setattr__ so nn.Module.__setattr__ does NOT register
        # `mot` as a submodule — otherwise MoT -> MoTBlock -> mot creates a cycle
        # that breaks `.to(device)` with infinite recursion.
        object.__setattr__(self, "_mot_ref", mot)
        self.video_block = video_block
        self.action_block = action_block
        self.do_ckpt = bool(do_ckpt)
        self._expert_use_gc_video = bool(expert_use_gc_video)
        self._expert_use_gc_action = bool(expert_use_gc_action)

    def _block_for(self, name: str) -> nn.Module:
        if name == "video":
            return self.video_block
        if name == "action":
            return self.action_block
        raise KeyError(f"Unknown expert name: {name}")

    def forward(self, mode: str, **kwargs):
        if mode == "joint":
            return self._joint(**kwargs)
        raise ValueError(f"Unsupported MoTBlock mode: {mode!r}")

    def _joint(
        self,
        *,
        embeds_all: Dict[str, torch.Tensor],
        freqs_all: Dict[str, Any],
        t_mod_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        attention_mask: torch.Tensor,
        prompt_timestep_all: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        mot = self._mot_ref
        expert_order = mot.expert_order

        # Build per-stream Q/K/V via the single-API helper.
        io_states: Dict[str, Dict[str, torch.Tensor]] = {}
        q_chunks: List[torch.Tensor] = []
        k_chunks: List[torch.Tensor] = []
        v_chunks: List[torch.Tensor] = []
        seq_lens: List[int] = []

        for name in expert_order:
            block = self._block_for(name)
            io = _build_expert_attention_io_ltx(
                block,
                x=embeds_all[name],
                pe=freqs_all[name],
                t_mod=t_mod_all[name],
            )
            io_states[name] = io
            q_chunks.append(io["q"])
            k_chunks.append(io["k"])
            v_chunks.append(io["v"])
            seq_lens.append(io["q"].shape[1])

        q_cat = torch.cat(q_chunks, dim=1)
        k_cat = torch.cat(k_chunks, dim=1)
        v_cat = torch.cat(v_chunks, dim=1)

        # Joint attention with FastWAM mask (bool → additive).
        attn_additive = _fastwam_mask_to_additive(attention_mask.to(q_cat.device), q_cat.dtype)
        num_heads = mot.num_heads
        mixed = self._joint_attention(q_cat, k_cat, v_cat, num_heads, attn_additive)

        # Split + per-stream post block.
        out: Dict[str, torch.Tensor] = {}
        start = 0
        for name, seq_len in zip(expert_order, seq_lens):
            end = start + seq_len
            mixed_slice = mixed[:, start:end, :]
            block = self._block_for(name)
            ctx_payload = context_all.get(name) or {}
            prompt_t = (prompt_timestep_all or {}).get(name)
            out[name] = _apply_expert_post_block_ltx(
                block=block,
                io_state=io_states[name],
                mixed_attn_out=mixed_slice,
                t_mod=t_mod_all[name],
                context=ctx_payload.get("context"),
                context_mask=ctx_payload.get("mask"),
                prompt_timestep=prompt_t,
            )
            start = end
        return out

    def _joint_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Joint SDPA over concatenated Q/K/V from both experts."""
        # q/k/v shape: (B, S, H*D); rearrange to (B, H, S, D) for SDPA
        q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)

        def _attn(q_, k_, v_):
            out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask)
            return rearrange(out, "b h s d -> b s (h d)")

        mot = self._mot_ref
        ckpt = mot.mot_checkpoint_mixed_attn and self.do_ckpt
        if ckpt and self.training:
            return torch.utils.checkpoint.checkpoint(_attn, q, k, v, use_reentrant=False)
        return _attn(q, k, v)


# ============================================================================
# MoT — top-level orchestrator (FastWAM-compatible signature)
# ============================================================================


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        mot_checkpoint_stride: int = 1,
    ) -> None:
        super().__init__()
        if not _LTX_OK:
            raise ImportError(
                "ltx-core unavailable for LTX joint MoT; "
                f"original error: {_LTX_ERR!r}"
            )
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = ["video", "action"]  # deterministic order; matches FastWAM mask layout
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        if int(mot_checkpoint_stride) < 1:
            raise ValueError(f"`mot_checkpoint_stride` must be >= 1, got {mot_checkpoint_stride}")
        self.mot_checkpoint_stride = int(mot_checkpoint_stride)

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must share num_layers; got {self.num_layers} vs {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must share num_heads; got {self.num_heads} vs {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    f"All experts must share attn_head_dim; got {self.attn_head_dim} vs {expert.attn_head_dim}"
                )

        logger.info(
            "Initialized LTX joint MoT (single-API): experts=%s, num_layers=%d, num_heads=%d, attn_head_dim=%d",
            self.expert_order, self.num_layers, self.num_heads, self.attn_head_dim,
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                "  Expert '%s': num_params=%.2fB",
                name, sum(p.numel() for p in expert.parameters()) / 1e9,
            )

        # Per-layer MoTBlock pair; FSDP wraps each as the boundary.
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        gc_v = bool(getattr(video_expert, "use_gradient_checkpointing", False))
        gc_a = bool(getattr(action_expert, "use_gradient_checkpointing", False))
        self.blocks = nn.ModuleList(
            [
                MoTBlock(
                    mot=self,
                    video_block=video_expert.blocks[i],
                    action_block=action_expert.blocks[i],
                    do_ckpt=self._layer_does_ckpt(i),
                    expert_use_gc_video=gc_v,
                    expert_use_gc_action=gc_a,
                )
                for i in range(self.num_layers)
            ]
        )

    def _layer_does_ckpt(self, layer_idx: int) -> bool:
        return (layer_idx % self.mot_checkpoint_stride) == 0

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, Any],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        prompt_timestep_all: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        for k in self.expert_order:
            if k not in embeds_all:
                raise ValueError(f"Missing expert tokens for {k!r}")
            if k not in freqs_all:
                raise ValueError(f"Missing expert freqs for {k!r}")
            if k not in t_mod_all:
                raise ValueError(f"Missing expert t_mod for {k!r}")

        if attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be 2D square; got shape {tuple(attention_mask.shape)}"
            )

        tokens_all = dict(embeds_all)
        for layer_idx in range(self.num_layers):
            tokens_all = self.blocks[layer_idx](
                "joint",
                embeds_all=tokens_all,
                freqs_all=freqs_all,
                t_mod_all=t_mod_all,
                context_all=context_all,
                attention_mask=attention_mask,
                prompt_timestep_all=prompt_timestep_all,
            )
        return tokens_all


__all__ = [
    "MoT",
    "MoTBlock",
]
