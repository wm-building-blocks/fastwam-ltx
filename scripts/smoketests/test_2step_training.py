"""Run 2 training steps with gradients and optimizer to measure real training time."""
import torch
import time
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.optim import AdamW

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

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)
print(f"[√] Optimizer created")

B = 1
sample = {
    "video": torch.randn(B, 3, 33, 320, 384, device="cuda", dtype=torch.bfloat16) * 0.3,
    "action": torch.randn(B, 32, 14, device="cuda", dtype=torch.bfloat16) * 0.1,
    "proprio": torch.randn(B, 33, 14, device="cuda", dtype=torch.bfloat16) * 0.1,
    "context": torch.randn(B, 128, 4096, device="cuda", dtype=torch.bfloat16) * 0.1,
    "context_mask": torch.ones(B, 128, device="cuda", dtype=torch.bool),
}

model.train()
step_times = []
fwd_times = []
bwd_times = []
optim_times = []

for step_idx in range(2):
    print(f"\n[*] Step {step_idx + 1}/2...")

    # Forward pass
    torch.cuda.synchronize()
    t_fwd_start = time.time()
    loss, loss_dict = model.training_loss(sample)
    torch.cuda.synchronize()
    t_fwd = time.time() - t_fwd_start
    fwd_times.append(t_fwd)
    print(f"    Forward: {t_fwd:.2f}s, Loss: {loss.item():.4f}")

    # Backward pass
    torch.cuda.synchronize()
    t_bwd_start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.time() - t_bwd_start
    bwd_times.append(t_bwd)
    print(f"    Backward: {t_bwd:.2f}s")

    # Optimizer step
    torch.cuda.synchronize()
    t_optim_start = time.time()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t_optim = time.time() - t_optim_start
    optim_times.append(t_optim)
    print(f"    Optimizer: {t_optim:.2f}s")

    t_step = t_fwd + t_bwd + t_optim
    step_times.append(t_step)
    print(f"    Total: {t_step:.2f}s")

print("\n" + "="*60)
print("TRAINING TIMING SUMMARY")
print("="*60)
for i in range(len(step_times)):
    print(f"\nStep {i+1}:")
    print(f"  Forward:    {fwd_times[i]:6.2f}s")
    print(f"  Backward:   {bwd_times[i]:6.2f}s")
    print(f"  Optimizer:  {optim_times[i]:6.2f}s")
    print(f"  Total:      {step_times[i]:6.2f}s")

print(f"\nAverage per step: {sum(step_times)/len(step_times):.2f}s")
print("="*60)
