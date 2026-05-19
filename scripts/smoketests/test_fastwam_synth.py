"""Task 14: synthetic full-stack smoke for FastWAM(LTX).

Bypasses the 22B-checkpoint loader and the 43GB safetensors file; instead
builds a tiny 2-layer LTX video DiT + 2-layer LTXAlignedActionDiT + stub VAE,
then runs FastWAM.training_loss() once on synthetic data.
"""
import torch
import torch.nn as nn
from fastwam.models.ltx.fastwam import FastWAM
from fastwam.models.ltx.ltx_video_dit import LTXVideoDiT, load_ltx_config_from_safetensors
from fastwam.models.ltx.action_dit import LTXAlignedActionDiT
from fastwam.models.ltx.mot import MoT

device, dtype = "cuda", torch.bfloat16
torch.manual_seed(0)

# 1) Tiny LTX video DiT (2 layers from real metadata, random init).
cfg = load_ltx_config_from_safetensors(
    "checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors"
)
cfg_small = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
cfg_small["transformer"]["num_layers"] = 2

print("[1] build LTXVideoDiT (2 layers, random init)...")
video = LTXVideoDiT(config=cfg_small)
# Replace torch.empty params (ltx-core uninitialized) with small noise.
for p in video.parameters():
    if torch.isnan(p).any() or p.std().item() == 0.0:
        nn.init.normal_(p, std=0.02)
video = video.to(device=device, dtype=dtype)
print(f"   video hidden={video.hidden_dim} layers={video.num_layers} num_heads={video.num_heads}")

# 2) Tiny ActionDiT.
print("[2] build LTXAlignedActionDiT (2 layers)...")
action = LTXAlignedActionDiT(
    action_dim=10, hidden_dim=1024, num_heads=32, attn_head_dim=128,
    num_layers=2, text_dim=4096, eps=1e-6, cross_attention_adaln=False,
)
for p in action.parameters():
    if torch.isnan(p).any() or p.std().item() == 0.0:
        nn.init.normal_(p, std=0.02)
action = action.to(device=device, dtype=dtype)

# 3) MoT
mot = MoT(mixtures={"video": video, "action": action}, mot_checkpoint_mixed_attn=False)

# 4) Stub VAE: encode (B, 3, F, H, W) -> (B, 128, F', H', W') with strides 32/8.
class StubVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_dim = 128
        self.upsampling_factor = 32
        self.temporal_downsample_factor = 8
    @property
    def model(self):
        return self
    @torch.no_grad()
    def encode(self, video, device=None, **kw):
        if isinstance(video, list):
            video = torch.stack(video, dim=0)
        if device is not None:
            video = video.to(device)
        B, C, F, H, W = video.shape
        Fp = (F - 1) // 8 + 1 if F > 1 else 1
        Hp, Wp = H // 32, W // 32
        return torch.randn(B, 128, Fp, max(Hp,1), max(Wp,1), device=video.device, dtype=video.dtype) * 0.1
    @torch.no_grad()
    def decode(self, latent, device=None, **kw):
        return torch.zeros(latent.shape[0], 3, 33, latent.shape[-2]*32, latent.shape[-1]*32,
                           device=latent.device, dtype=latent.dtype)

vae = StubVAE().to(device=device, dtype=dtype)

# 5) Build FastWAM
print("[3] build FastWAM...")
model = FastWAM(
    video_expert=video,
    action_expert=action,
    mot=mot,
    vae=vae,
    text_encoder=None,
    text_dim=4096,
    proprio_dim=None,
    device=device,
    torch_dtype=dtype,
)
print(f"   model.dit (mot) has {sum(p.numel() for p in mot.parameters())/1e9:.2f}B params")

# 6) Synthetic sample matching FastWAM build_inputs contract.
B, T = 1, 9                          # (T-1) % 8 == 0
H = W = 64                           # multiples of 32
T_action = 16                         # divisible by (T-1)=8
sample = {
    "video": torch.randn(B, 3, T, H, W, dtype=dtype, device=device) * 0.5,
    "action": torch.randn(B, T_action, 10, dtype=dtype, device=device) * 0.5,
    "context": torch.randn(B, 32, 4096, dtype=dtype, device=device) * 0.1,
    "context_mask": torch.ones(B, 32, dtype=torch.bool, device=device),
}

print("[4] FastWAM.training_loss(sample)...")
loss_total, loss_dict = model.training_loss(sample)
print(f"   loss_total={loss_total.item():.4f}")
print(f"   {loss_dict}")
assert torch.isfinite(loss_total).item(), "loss not finite"
print("FASTWAM SYNTHETIC SMOKE OK")
