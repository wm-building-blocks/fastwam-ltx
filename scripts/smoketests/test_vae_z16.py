"""Standalone test: instantiate WanVideoVAE(z_dim=16) and verify shape."""
import torch
from fastwam.models.wan22.wan_video_vae import WanVideoVAE

vae = WanVideoVAE(z_dim=16).to(device="cuda", dtype=torch.bfloat16)
assert vae.z_dim == 16
assert vae.upsampling_factor == 8
assert vae.temporal_downsample_factor == 4

# Encode a synthetic 9-frame 384x320 video
H, W, T = 320, 384, 9
video = torch.randn(1, 3, T, H, W, device="cuda", dtype=torch.bfloat16) * 0.5
z = vae.encode(video, device="cuda", tiled=False)
print("encoded z shape:", tuple(z.shape))
# Expected: [1, 16, ceil((9-1)/4)+1=3, 320/8=40, 384/8=48]
assert tuple(z.shape) == (1, 16, 3, 40, 48), f"Got {tuple(z.shape)}"

recon = vae.decode(z, device="cuda", tiled=False)
print("decoded recon shape:", tuple(recon.shape))
assert tuple(recon.shape) == (1, 3, T, H, W), f"Got {tuple(recon.shape)}"
print("Wan2.1-VAE (z_dim=16) load+roundtrip OK.")
