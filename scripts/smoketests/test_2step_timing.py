"""Construct FastWAM with A14B and run 2 forward steps to measure timing."""
import torch
import time
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fastwam.utils.config_resolvers import register_default_resolvers
register_default_resolvers()

repo_root = Path(__file__).resolve().parents[2]
with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=robotwin_uncond_3cam_384_1e-4"])

print("[*] Building model...")
t0 = time.time()
model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
t_build = time.time() - t0
print(f"[√] Model built in {t_build:.2f}s")
print(f"    Device: {model.device}, dtype: {model.torch_dtype}")
print(f"    Video expert: {sum(p.numel() for p in model.video_expert.parameters())/1e9:.2f} B")
print(f"    Action expert: {sum(p.numel() for p in model.action_expert.parameters())/1e9:.2f} B")

B = 1
sample = {
    "video": torch.randn(B, 3, 33, 320, 384, device="cuda", dtype=torch.bfloat16) * 0.3,
    "action": torch.randn(B, 32, 14, device="cuda", dtype=torch.bfloat16) * 0.1,
    "proprio": torch.randn(B, 33, 14, device="cuda", dtype=torch.bfloat16) * 0.1,
    "context": torch.randn(B, 128, 4096, device="cuda", dtype=torch.bfloat16) * 0.1,
    "context_mask": torch.ones(B, 128, device="cuda", dtype=torch.bool),
}

model.eval()
step_times = []

for step_idx in range(2):
    print(f"\n[*] Step {step_idx + 1}/2...")
    torch.cuda.synchronize()
    t_step_start = time.time()

    with torch.no_grad():
        loss, loss_dict = model.training_loss(sample)

    torch.cuda.synchronize()
    t_step = time.time() - t_step_start
    step_times.append(t_step)

    print(f"[√] Step {step_idx + 1} completed in {t_step:.2f}s")
    print(f"    Loss: {loss.item():.4f}")
    print(f"    Loss dict: {loss_dict}")

print("\n" + "="*60)
print("TIMING SUMMARY")
print("="*60)
print(f"Step 1 (cold, includes compilation): {step_times[0]:.2f}s")
print(f"Step 2 (warm):                       {step_times[1]:.2f}s")
print(f"Average (excluding compilation):     {sum(step_times[1:])/max(len(step_times)-1, 1):.2f}s")
print("="*60)
