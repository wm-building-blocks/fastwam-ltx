import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from einops import rearrange
from .helpers.gradient import gradient_checkpoint_forward

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, ctx_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
        
        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
            
        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class _LegacyWanLikeDiT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )
        
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False
        assert require_vae_embedding == False and fuse_vae_embedding_in_latents == True, "Only support fusing vae embedding in latents"

        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")
            

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = (num_latent_frames == 1 and action is None)
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}")

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")
            
            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, 0, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]

        context = self.text_embedding(context) # (B, L, dim)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action) # (B, action_len, dim)
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim, 
                torch.arange(action_len, device=action_emb.device)) # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0) # (B, action_len, dim)
            context = torch.cat([context, action_emb], dim=1) # (B, context_len + action_len, dim)

            # new mask
            num_temporal_groups = f - 1 # first latent frame do not attend to actions
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            assert action_emb.shape[1] % num_temporal_groups == 0, \
                f"Action embedding length {action_emb.shape[1]} must be divisible by number of temporal groups {num_temporal_groups}"
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device) # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w # query length
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device) # (B, seq_len, L + action_len)
            # all latent frames attend to text tokens
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, seq_len, L)
            # latent frames from the 2nd one attend to action tokens
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode with num_latent_frames=1."
                )
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None # special rule for faster speed

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask
                )
            else:
                x_tokens = block(x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask)

        return self.post_dit(x_tokens, pre_state)


# ============================================================================
# LTX-2 backbone adapter (Task 4 / 2026-05-19)
# ============================================================================
# Wraps ltx-core's LTXModel(VideoOnly) to fit FastWAM's video-backbone interface.
# The legacy Wan-style classes above (DiTBlock, _LegacyWanLikeDiT, helpers like
# flash_attention/modulate/rope_apply) remain for ActionDiT + MoT during the
# transition; Task 9 will replace those.

import json as _json
from pathlib import Path as _Path
from typing import Any as _Any, Dict as _Dict, Optional as _Optional, Tuple as _Tuple

try:
    import safetensors as _safetensors
    from ltx_core.model.transformer import (
        LTXModel as _LTXModel,
        Modality as _Modality,
        LTXV_MODEL_COMFY_RENAMING_MAP as _LTXV_RENAMING,
        LTXVideoOnlyModelConfigurator as _LTXVideoOnlyConfigurator,
    )
    from ltx_core.model.transformer.model import LTXModelType as _LTXModelType
    _LTX_CORE_OK = True
    _LTX_CORE_ERR = None
except Exception as _e:  # noqa: BLE001
    _LTX_CORE_OK = False
    _LTX_CORE_ERR = _e


# ---- Audio key filter ----------------------------------------------------------
# Tokens that, when present anywhere in a state-dict key, mark it as belonging
# to the audio sub-model or to AV cross-attention layers. In LTX-2.3 VideoOnly
# none of these modules are instantiated, so they must be filtered out before
# load_state_dict(strict=...).
_AUDIO_FILTER_TOKENS: _Tuple[str, ...] = (
    "audio_patchify_proj",
    "audio_adaln_single",
    "audio_scale_shift_table",
    "audio_norm_out",
    "audio_proj_out",
    "audio_args_preprocessor",
    "audio_caption_projection",
    "audio_prompt_adaln_single",
    "audio_prompt_scale_shift_table",
    "audio_attn1",
    "audio_attn2",
    "audio_ff",
    "video_to_audio_attn",
    "audio_to_video_attn",
    "audio_embeddings_connector",
    "video_embeddings_connector",
    "embeddings_connector",
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_v2a_gate_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_video_scale_shift_adaln_single",
    "scale_shift_table_a2v_ca_audio",
    "scale_shift_table_a2v_ca_video",
)


def ltx_video_dit_filter_audio(state_dict: _Dict[str, _Any]) -> _Dict[str, _Any]:
    """Drop every key whose name contains an audio / AV-cross-attn token.

    Reason: LTX-2.3's safetensors file holds both the 14B video DiT and the
    5B audio DiT in a single shard, but VideoOnly model instantiation never
    creates the audio submodules. Loading without filtering causes hundreds
    of unexpected_keys.
    """
    out: _Dict[str, _Any] = {}
    dropped = 0
    for k, v in state_dict.items():
        if any(tok in k for tok in _AUDIO_FILTER_TOKENS):
            dropped += 1
            continue
        out[k] = v
    return out


def _strip_comfy_prefix(state_dict: _Dict[str, _Any]) -> _Dict[str, _Any]:
    """Select DiT keys (``model.diffusion_model.X``) and strip the prefix.

    LTX-2.3 safetensors files bundle multiple modules (DiT, VAE, audio_vae,
    vocoder, text_embedding_projection). Only ``model.diffusion_model.*``
    belongs to the LTXModel transformer; everything else is discarded here
    and loaded by its own adapter.
    """
    pref = "model.diffusion_model."
    return {k[len(pref):]: v for k, v in state_dict.items() if k.startswith(pref)}


def load_ltx_config_from_safetensors(path: str) -> _Dict[str, _Any]:
    """Read the ``__metadata__["config"]`` JSON blob from an LTX safetensors file.

    Returns the parsed dict (top-level keys typically ``transformer``, ``vae``,
    possibly ``audio_vae`` etc.). Raises if the file has no embedded config.
    """
    with _safetensors.safe_open(path, framework="pt") as f:
        md = f.metadata() or {}
    if "config" not in md:
        raise ValueError(
            f"safetensors file {path!s} has no '__metadata__[\"config\"]' blob"
        )
    return _json.loads(md["config"])


def load_ltx_safetensors_state_dict(path: str, device: str = "cpu") -> _Dict[str, _Any]:
    """Stream the safetensors tensors into a state-dict (CPU by default)."""
    out: _Dict[str, _Any] = {}
    with _safetensors.safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


# ---- LTXVideoDiT wrapper -------------------------------------------------------
class LTXVideoDiT(torch.nn.Module):
    """FastWAM-side adapter wrapping ``ltx_core.LTXModel(model_type=VideoOnly)``.

    Exposes the attribute surface FastWAM scaffolding (MoT, FSDP wrap policy,
    ``preprocess_action_dit_backbone.py``) expects:
        ``hidden_dim``, ``num_heads``, ``attn_head_dim``, ``num_layers``,
        ``blocks`` (alias of ``self._inner.transformer_blocks``).

    Construction options:
        ``LTXVideoDiT(config=..., use_gradient_checkpointing=...)`` — full
          LTX safetensors config dict (the one inside ``__metadata__["config"]``).
        ``LTXVideoDiT.from_safetensors(path, ...)`` — parse metadata + build.
    """

    def __init__(
        self,
        *,
        config: _Optional[_Dict[str, _Any]] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if not _LTX_CORE_OK:
            raise ImportError(
                "ltx-core unavailable; pip install -e third_party/ltx-2/packages/ltx-core. "
                f"Original error: {_LTX_CORE_ERR!r}"
            )
        if config is None:
            raise ValueError(
                "LTXVideoDiT requires `config=<dict>`. Use "
                "LTXVideoDiT.from_safetensors(path) to read it from a checkpoint."
            )

        self._config = config
        # Use the official VideoOnly configurator. It validates non-default keys
        # and assembles caption_projection appropriately for 19B vs 22B models.
        self._inner: torch.nn.Module = _LTXVideoOnlyConfigurator.from_config(config)

        # FastWAM-conventional attribute exposure.
        tcfg = config.get("transformer", {})
        self.hidden_dim: int = self._inner.inner_dim
        self.num_heads: int = tcfg.get("num_attention_heads", 32)
        self.attn_head_dim: int = tcfg.get("attention_head_dim", 128)
        self.num_layers: int = tcfg.get("num_layers", 48)
        self.in_dim: int = tcfg.get("in_channels", 128)
        self.out_dim: int = tcfg.get("out_channels", 128)
        # Alias so MoT / preprocess scripts can iterate `model.blocks`.
        self.blocks = self._inner.transformer_blocks

        self.use_gradient_checkpointing = use_gradient_checkpointing
        if use_gradient_checkpointing:
            self._inner.set_gradient_checkpointing(True)

    @classmethod
    def from_safetensors(
        cls,
        path: str,
        use_gradient_checkpointing: bool = False,
    ) -> "LTXVideoDiT":
        cfg = load_ltx_config_from_safetensors(path)
        return cls(config=cfg, use_gradient_checkpointing=use_gradient_checkpointing)

    # -- weight loading --------------------------------------------------------
    def load_pretrained_state_dict(
        self,
        state_dict: _Dict[str, _Any],
        strict: bool = False,
    ) -> _Tuple[list, list]:
        """Load an LTX safetensors state dict with comfy prefix stripping +
        audio key filtering. Returns ``(missing_keys, unexpected_keys)``."""
        sd = _strip_comfy_prefix(state_dict)
        sd = ltx_video_dit_filter_audio(sd)
        result = self._inner.load_state_dict(sd, strict=strict)
        # torch's return type changed across versions; normalize to lists.
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        return missing, unexpected

    def load_pretrained_safetensors(
        self,
        path: str,
        strict: bool = False,
        device: str = "cpu",
    ) -> _Tuple[list, list]:
        sd = load_ltx_safetensors_state_dict(path, device=device)
        return self.load_pretrained_state_dict(sd, strict=strict)

    # -- forward ---------------------------------------------------------------
    # ====================================================================
    # Task 8: layer-level API for MoT joint forward
    # ====================================================================
    # FastWAM's MoT machinery wants to iterate through the video DiT's blocks
    # manually, intercepting each block's self-attention to perform joint
    # attention with the action expert. To support this we expose:
    #   - prepare(latent_5d, sigma, context, ...) -> (TransformerArgs, meta)
    #   - postprocess(TransformerArgs, meta) -> 5D latent
    #   - build_video_to_video_mask(...) -> (T_v, T_v) bool mask
    # The blocks themselves remain ltx_core.BasicAVTransformerBlock instances;
    # MoT calls their attn1.to_q/to_k/to_v/q_norm/k_norm/to_out attributes
    # directly when running joint attention.

    def prepare(
        self,
        latent_5d,
        sigma,
        context,
        context_mask=None,
        video_self_attention_mask=None,
        perturbations=None,
    ):
        """Convert 5D pixel-latent + conditioning into a TransformerArgs ready
        for block-by-block forward.

        Args:
            latent_5d: (B, C=in_channels=128, F, H, W) tensor from the VAE.
            sigma: (B,) diffusion noise level.
            context: (B, T_text, inner_dim=4096) text embeddings from
                LTXTextEncoder (already projected to inner_dim).
            context_mask: (B, T_text) binary mask, 1 for valid tokens.
            video_self_attention_mask: optional (B, T, T) self-attention mask.
                ``None`` means full attention; pass FastWAM-style masks here.
            perturbations: BatchedPerturbationConfig; ``None`` defaults to empty.

        Returns:
            (video_args, meta) where meta carries token-grid info for downstream
            mask construction.
        """
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig
        from ltx_core.model.transformer.modality import Modality

        B, C, F, H, W = latent_5d.shape
        T = F * H * W
        device = latent_5d.device

        # (B, C, F, H, W) -> (B, T=F*H*W, C) in time-major then h-major then w-major
        x_tokens = latent_5d.flatten(2).transpose(1, 2).contiguous()

        # Build positions (B, 3, T) where dim-1 indexes (time, height, width).
        t_idx = torch.arange(F, device=device)
        h_idx = torch.arange(H, device=device)
        w_idx = torch.arange(W, device=device)
        tt, hh, ww = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
        positions = torch.stack(
            [tt.flatten(), hh.flatten(), ww.flatten()], dim=0
        ).to(dtype=torch.float32 if self._inner.double_precision_rope else x_tokens.dtype)
        # LTX with use_middle_indices_grid=True expects (B, 3, T, 2) where the
        # last dim is (start, end) of each token's grid extent. For point
        # tokens we set start==end so the midpoint equals the position itself.
        positions = positions.unsqueeze(0).expand(B, -1, -1).contiguous()       # (B, 3, T)
        positions = positions.unsqueeze(-1).expand(-1, -1, -1, 2).contiguous()  # (B, 3, T, 2)

        # Per-token timesteps: broadcast batch-level sigma to all T tokens so we
        # match LTX's per-token timestep_embedding code path with no behavior
        # change for batch-shared t.
        if sigma.ndim == 0:
            sigma = sigma.expand(B)
        timesteps = sigma.view(B, 1).expand(B, T).contiguous()  # (B, T)

        modality = Modality(
            latent=x_tokens,
            sigma=sigma,
            timesteps=timesteps,
            positions=positions,
            context=context,
            enabled=True,
            context_mask=context_mask,
            attention_mask=video_self_attention_mask,
        )

        video_args = self._inner.video_args_preprocessor.prepare(modality, cross_modality=None)

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(B)

        meta = {
            "B": B,
            "T": T,
            "F": F,
            "H": H,
            "W": W,
            "tokens_per_frame": H * W,
            "num_frames": F,
            "perturbations": perturbations,
        }
        return video_args, meta

    def run_blocks(self, video_args, perturbations=None):
        """Convenience: run all transformer blocks sequentially with audio=None.

        Equivalent to LTXModel._process_transformer_blocks but exposed so that
        MoT can choose between this 'native' loop (when joint attention is
        disabled) and its own joint-attention loop.
        """
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(video_args.x.shape[0])

        for block in self.blocks:
            video_args, _ = block(video=video_args, audio=None, perturbations=perturbations)
        return video_args

    def postprocess(self, video_args, meta):
        """Apply final scale_shift + norm_out + proj_out, then reshape back to 5D.

        Returns a (B, C_out=128, F, H, W) tensor matching the original
        ``latent_5d.shape`` (with C swapped to out_channels).
        """
        x = self._inner._process_output(
            self._inner.scale_shift_table,
            self._inner.norm_out,
            self._inner.proj_out,
            video_args.x,
            video_args.embedded_timestep,
        )
        # x shape: (B, T, out_channels)
        B = meta["B"]
        F, H, W = meta["F"], meta["H"], meta["W"]
        # (B, T, C) -> (B, C, F, H, W)
        x_5d = x.transpose(1, 2).reshape(B, -1, F, H, W).contiguous()
        return x_5d

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device,
        mode: str = "bidirectional",
    ):
        """Port of wan22 WanVideoDiT.build_video_to_video_mask.

        Returns a (video_seq_len, video_seq_len) bool tensor where ``True``
        means "query at row i can attend to key at column j". FastWAM uses this
        as the video->video block of the larger MoT attention mask.

        Modes:
          - "bidirectional": full attention (all True)
          - "per_frame_causal": frame i can attend to frames <= i
          - "first_frame_causal": first frame cannot see future frames; later
              frames see everything (FastWAM RoboTwin default)
        """
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {mode}")

    def forward(
        self,
        video,  # ltx_core.model.transformer.modality.Modality
        perturbations=None,
    ):
        """Run one transformer pass, forcing audio=None (VideoOnly).

        ``video`` must already be a ``ltx_core.Modality``. Higher-level wrappers
        (FastWAM ``forward(x, context, t, ...)``) will be added in Task 9 once
        MoT integration is in place.
        """
        return self._inner(video=video, audio=None, perturbations=perturbations)


__all__ = [
    *globals().get("__all__", []),
    "LTXVideoDiT",
    "ltx_video_dit_filter_audio",
    "load_ltx_config_from_safetensors",
    "load_ltx_safetensors_state_dict",
]
