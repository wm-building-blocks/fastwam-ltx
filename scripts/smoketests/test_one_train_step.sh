#!/usr/bin/env bash
# Run exactly 1 training step under DeepSpeed Zero-2 on RoboTwin (smoke test).
set -euo pipefail
cd "$(dirname "$0")/../.."
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
NPROC="${1:-4}"
bash scripts/train_zero2.sh "$NPROC" \
  task=robotwin_uncond_3cam_384_1e-4 \
  num_epochs=1 \
  max_steps=1 \
  log_every=1 \
  save_every=1 \
  eval_every=999999 \
  wandb.enabled=false \
  data.train.pretrained_norm_stats=null \
  data.val.pretrained_norm_stats=null \
  output_dir=./runs/_smoke/$(date +%s)
