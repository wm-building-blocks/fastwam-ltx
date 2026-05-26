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

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .helpers.attention_backend import sdpa_backend_ctx
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


# A MoT block param is reachable through several aliased module paths, all the
# same nn.Module (see MoT.__init__):
#   * `mixtures.<expert>.blocks.<i>.*`
#   * `mixtures.video._inner.transformer_blocks.<i>.*` — the video expert sets
#     `self.blocks = self._inner.transformer_blocks` (ltx_video_dit.py), so the
#     LTX DiT's own `transformer_blocks` is a third alias of the same modules.
# Only the canonical `blocks.<i>.{video,action}_block.*` path goes through the
# MoTBlock FSDP unit; every alias path bypasses FSDP and yields broken sharded
# flat-param shards (size [0] / partial) on save. Drop them all.
_MIXTURES_BLOCKS_ALIAS_RE = re.compile(
    r"mixtures\.[^.]+\.blocks\.\d+\."
    r"|mixtures\.[^.]+\._inner\.transformer_blocks\.\d+\."
)


def _drop_mixtures_blocks_alias(module, state_dict, prefix, local_metadata):
    """Drop MoT block alias keys from a MoT state_dict.

    See MoT.__init__ and `_MIXTURES_BLOCKS_ALIAS_RE` for rationale. Keys are
    full paths including `prefix` (e.g.
    "mot.mixtures.video.blocks.0.attn1.to_q.weight" when this MoT lives under a
    parent module that called state_dict with prefix="mot."). The parent's
    traversal is unaffected — only this MoT's own emitted keys are filtered.
    Non-block params under mixtures.<expert>.* and mixtures.video._inner.*
    (patchify_proj, adaln_single, scale_shift_table, norm_out, proj_out) are
    kept — they are NOT aliased through `self.blocks`.
    """
    for key in list(state_dict.keys()):
        rel = key[len(prefix):] if key.startswith(prefix) else key
        if _MIXTURES_BLOCKS_ALIAS_RE.match(rel):
            del state_dict[key]
    return state_dict


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
        # `video_block` / `action_block` ARE registered as submodules of
        # MoTBlock. This produces two paths to the same nn.Module in the
        # model tree (`mot.mixtures.<expert>.blocks[i]` AND
        # `mot.blocks[i].<>_block`); state_dict() therefore emits both. We
        # filter the `mixtures.<>.blocks.*` alias on save (see MoT.__init__).
        # FSDP wraps at `MoTBlock` granularity (see accelerate_fsdp.yaml);
        # attempting to drop the submodule registration here and wrap at the
        # inner block class triggers a FSDP recursive-wrap AssertionError.
        # Resume from dcp `state/` currently fails because dcp records the
        # alias side with shape (0,) — load uses `weights/.pt` + `strict=False`
        # via `load_checkpoint` instead (see fastwam.py:1132).
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
        if mode == "video_prefill":
            return self._video_prefill(**kwargs)
        if mode == "action_with_kv":
            return self._action_with_kv(**kwargs)
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

    def _video_prefill(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: Tuple[torch.Tensor, torch.Tensor],
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_prompt_timestep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run video stream alone through this layer; cache post-rope K/V for
        cross-stream attention in `_action_with_kv`. Mirrors the Wan-style
        prefill API but uses LTX block internals.
        """
        mot = self._mot_ref
        block = self.video_block
        io = _build_expert_attention_io_ltx(
            block, x=video_tokens, pe=video_freqs, t_mod=video_t_mod,
        )
        attn_additive = _fastwam_mask_to_additive(
            video_attention_mask.to(io["q"].device), io["q"].dtype
        )
        mixed = self._joint_attention(
            io["q"], io["k"], io["v"], mot.num_heads, attn_additive
        )
        ctx = video_context_payload or {}
        new_video_tokens = _apply_expert_post_block_ltx(
            block=block,
            io_state=io,
            mixed_attn_out=mixed,
            t_mod=video_t_mod,
            context=ctx.get("context"),
            context_mask=ctx.get("mask"),
            prompt_timestep=video_prompt_timestep,
        )
        return new_video_tokens, io["k"], io["v"]

    def _action_with_kv(
        self,
        *,
        action_tokens: torch.Tensor,
        action_freqs: Tuple[torch.Tensor, torch.Tensor],
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        k_video: torch.Tensor,
        v_video: torch.Tensor,
        action_attention_mask: torch.Tensor,
        action_prompt_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run action stream against cached video K/V at this layer. The
        cached K/V already had q_norm/k_norm and LTX SPLIT RoPE applied in
        `_video_prefill`, so we just cat them onto action's K/V and run a
        single SDPA call. Attention mask shape is `(T_action, T_video + T_action)`.
        """
        mot = self._mot_ref
        block = self.action_block
        io = _build_expert_attention_io_ltx(
            block, x=action_tokens, pe=action_freqs, t_mod=action_t_mod,
        )
        k_cat = torch.cat([k_video, io["k"]], dim=1)
        v_cat = torch.cat([v_video, io["v"]], dim=1)
        attn_additive = _fastwam_mask_to_additive(
            action_attention_mask.to(io["q"].device), io["q"].dtype
        )
        mixed = self._joint_attention(
            io["q"], k_cat, v_cat, mot.num_heads, attn_additive
        )
        ctx = action_context_payload or {}
        new_action_tokens = _apply_expert_post_block_ltx(
            block=block,
            io_state=io,
            mixed_attn_out=mixed,
            t_mod=action_t_mod,
            context=ctx.get("context"),
            context_mask=ctx.get("mask"),
            prompt_timestep=action_prompt_timestep,
        )
        return new_action_tokens

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
            with sdpa_backend_ctx():
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

        # `mixtures.<expert>.blocks[i]` and `blocks[i].{video,action}_block`
        # are the same nn.Module — state_dict traverses both, producing 4
        # keys per block param. Under FSDP with
        # `transformer_layer_cls_to_wrap=[MoTBlock]`, only the `blocks.*` path
        # goes through MoTBlock's FSDP unit and receives the unsharded gather;
        # the `mixtures.<>.blocks.*` path bypasses FSDP and yields sharded
        # flat-param shards. Filtering the alias on save halves ckpt size
        # and avoids saving broken sharded tensors. Load uses `strict=False`
        # and the alias is restored by the shared nn.Module references
        # reconstructed in `__init__`.
        self._register_state_dict_hook(_drop_mixtures_blocks_alias)

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

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: Tuple[torch.Tensor, torch.Tensor],
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_prompt_timestep: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Two-pass inference, phase 1: run the full video stack alone and
        return the post-rope K/V from every layer. Phase 2
        (`forward_action_with_video_cache`) reuses these cached K/V so the
        action denoiser can attend to a fixed prefix without re-encoding
        video at every action denoise step.
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [V, V], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        x = video_tokens
        kv_cache: List[Dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            x, k, v = self.blocks[layer_idx](
                "video_prefill",
                video_tokens=x,
                video_freqs=video_freqs,
                video_t_mod=video_t_mod,
                video_context_payload=video_context_payload,
                video_attention_mask=video_attention_mask,
                video_prompt_timestep=video_prompt_timestep,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: Tuple[torch.Tensor, torch.Tensor],
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: List[Dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_prompt_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Two-pass inference, phase 2. `attention_mask` must be the full
        [V+A, V+A] joint mask used at training time; the action-row slice is
        taken internally so the caller passes the same mask object as
        `_build_mot_attention_mask` produced.
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [V+A, V+A], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        x = action_tokens
        for layer_idx in range(self.num_layers):
            cache = video_kv_cache[layer_idx]
            if "k" not in cache or "v" not in cache:
                raise ValueError(f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`.")
            k_video = cache["k"]
            v_video = cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )
            x = self.blocks[layer_idx](
                "action_with_kv",
                action_tokens=x,
                action_freqs=action_freqs,
                action_t_mod=action_t_mod,
                action_context_payload=action_context_payload,
                k_video=k_video,
                v_video=v_video,
                action_attention_mask=action_attention_mask,
                action_prompt_timestep=action_prompt_timestep,
            )
        return x


__all__ = [
    "MoT",
    "MoTBlock",
]
