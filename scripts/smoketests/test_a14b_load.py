"""Construct FastWAM with A14B high-noise expert and run 1 forward step on random data."""
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fastwam.utils.config_resolvers import register_default_resolvers
register_default_resolvers()

repo_root = Path(__file__).resolve().parents[2]
with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=robotwin_uncond_3cam_384_1e-4"])

model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
print(f"Model on {model.device}, dtype={model.torch_dtype}")
print(f"Video expert params: {sum(p.numel() for p in model.video_expert.parameters())/1e9:.2f} B")
print(f"Action expert params: {sum(p.numel() for p in model.action_expert.parameters())/1e9:.2f} B")

B = 1
sample = {
    "video": torch.randn(B, 3, 33, 320, 384, device="cuda", dtype=torch.bfloat16) * 0.3,
    "action": torch.randn(B, 32, 14, device="cuda", dtype=torch.bfloat16) * 0.1,
    "proprio": torch.randn(B, 33, 14, device="cuda", dtype=torch.bfloat16) * 0.1,
    "context": torch.randn(B, 128, 4096, device="cuda", dtype=torch.bfloat16) * 0.1,
    "context_mask": torch.ones(B, 128, device="cuda", dtype=torch.bool),
}

model.eval()
with torch.no_grad():
    loss, loss_dict = model.training_loss(sample)
print(f"Loss: {loss.item():.4f} dict: {loss_dict}")
print("A14B high-noise expert forward OK.")
