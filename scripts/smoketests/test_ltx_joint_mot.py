"""Task 11 smoke: end-to-end joint MoT forward on small synthetic input."""
import torch
from fastwam.models.ltx.ltx_video_dit import LTXVideoDiT, load_ltx_config_from_safetensors
from fastwam.models.ltx.action_dit import LTXAlignedActionDiT
from fastwam.models.ltx.mot import MoT

torch.manual_seed(0)
cfg = load_ltx_config_from_safetensors("checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors")

# Override num_layers to 2 for fast smoke (real run uses 48).
cfg_small = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
cfg_small["transformer"]["num_layers"] = 2

print("[1] build VideoDiT (random init, 2 layers) ...")
dit_v = LTXVideoDiT(config=cfg_small)
n_v = sum(p.numel() for p in dit_v.parameters())
print(f"   params: {n_v/1e9:.2f}B (2-layer smoke)")

print("[2] build ActionDiT (random init, 2 layers) ...")
dit_a = LTXAlignedActionDiT(
    action_dim=10, hidden_dim=1024, num_heads=32, attn_head_dim=128,
    num_layers=2, text_dim=4096, eps=1e-6,
    # Disable cross-attn AdaLN on action: it scales context (dim=4096) by params
    # at hidden_dim=1024, which is a dimensional mismatch. Plain cross-attn is fine.
    cross_attention_adaln=False,
)
n_a = sum(p.numel() for p in dit_a.parameters())
print(f"   params: {n_a/1e9:.3f}B")

print("[3] build MoT (joint single-API)...")
mot = MoT(mixtures={"video": dit_v, "action": dit_a}, mot_checkpoint_mixed_attn=False)
print(f"   MoT num_layers={mot.num_layers}, num_heads={mot.num_heads}, attn_head_dim={mot.attn_head_dim}")

device = "cuda"
dtype = torch.bfloat16

# Initialize uninitialized empty Parameters (ltx-core uses torch.empty in some places
# expecting load_state_dict to fill them). Without pretrained weights, fill with small
# noise to keep numerics bounded for the smoke test.
def _init_empties(model):
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or p.std().item() == 0.0:
            torch.nn.init.normal_(p, std=0.02)
_init_empties(dit_v)
_init_empties(dit_a)

dit_v.to(device=device, dtype=dtype)
dit_a.to(device=device, dtype=dtype)

print("[4] build inputs (small spatial grid)...")
B, F_v, H, W = 1, 3, 4, 4
T_a = 8
latent = torch.randn(B, 128, F_v, H, W, dtype=dtype, device=device) * 0.5
sigma_v = torch.tensor([0.5], dtype=dtype, device=device)
sigma_a = torch.tensor([0.7], dtype=dtype, device=device)
context = torch.randn(B, 32, 4096, dtype=dtype, device=device) * 0.1
ctx_mask = torch.ones(B, 32, dtype=torch.long, device=device)
action = torch.randn(B, T_a, 10, dtype=dtype, device=device) * 0.5

print("[5] video pre + action pre...")
video_args, video_meta = dit_v.prepare(latent, sigma_v, context, context_mask=ctx_mask)
video_pre = {
    "tokens": video_args.x,
    "freqs": video_args.positional_embeddings,
    "t_mod": video_args.timesteps,
    "context": video_args.context,
    "context_mask": video_args.context_mask,
    "embedded_timestep": video_args.embedded_timestep,
    "prompt_timestep": video_args.prompt_timestep,
}
action_pre = dit_a.pre_dit(action, sigma_a, context, context_mask=ctx_mask)
print(f"   video tokens: {tuple(video_pre['tokens'].shape)}")
print(f"   action tokens: {tuple(action_pre['tokens'].shape)}")

print("[6] build FastWAM joint attention mask...")
Tv = video_pre["tokens"].shape[1]
Ta = action_pre["tokens"].shape[1]
total = Tv + Ta
mask = torch.zeros((total, total), dtype=torch.bool, device=device)
mask[:Tv, :Tv] = dit_v.build_video_to_video_mask(Tv, H * W, device, mode="first_frame_causal")
mask[Tv:, Tv:] = True
mask[Tv:, : H * W] = True
print(f"   mask shape: {tuple(mask.shape)}  true: {int(mask.sum().item())}/{total*total}")

print("[7] mot.forward (single layer joint, 2-layer total)...")
with torch.no_grad():
    out = mot(
        embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
        attention_mask=mask,
        freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
        context_all={
            "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
            "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
        },
        t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        prompt_timestep_all={"video": video_pre["prompt_timestep"], "action": action_pre["prompt_timestep"]},
    )
print(f"   out video: {tuple(out['video'].shape)} (expected ({B},{Tv},4096))")
print(f"   out action: {tuple(out['action'].shape)} (expected ({B},{Ta},1024))")
assert out["video"].shape == (B, Tv, 4096)
assert out["action"].shape == (B, Ta, 1024)
print(f"   video stats mean={out['video'].float().mean().item():.4f} std={out['video'].float().std().item():.4f}")
print(f"   action stats mean={out['action'].float().mean().item():.4f} std={out['action'].float().std().item():.4f}")
assert torch.isfinite(out["video"]).all() and torch.isfinite(out["action"]).all()
print("JOINT MoT SMOKE OK")
