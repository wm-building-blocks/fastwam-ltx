# FastWAM_LTX 训练设置总结(用于诊断训练速度)

## TL;DR

- **当前速度**:~9.5s/step(0.10 step/s)
- **GPU util**:~50%(部分卡 100%,部分 0%,可能 NCCL straggler)
- **显存**:50-71 GB / 80 GB(rank 间不均)
- **怀疑**:通信瓶颈,但未 profile 确认

## 硬件

| | |
|---|---|
| 主机 | 单机,7 × NVIDIA H100 80GB HBM3(GPU 1-7,GPU 0 跑 openpi 不参训) |
| GPU 互联 | NVLink/NVSwitch(未显式验证) |
| CPU | 128 vCPU |
| 主机 RAM | 2.0 TiB,**当前 used 1.9 TiB**(buff/cache 67 GiB,available 41 GiB) |

## 软件

| | |
|---|---|
| PyTorch | 2.7.1 + CUDA 12.8 |
| Accelerate | 1.12.0 |
| bitsandbytes | 0.49.2 |
| 启动方式 | `accelerate launch --config_file accelerate_fsdp.yaml` |

## 模型(LTX-2.3 backbone + MoT 双 expert)

| | |
|---|---|
| Video DiT | LTX-2.3 22B base(`ltx-2.3-22b-dev.safetensors`,46GB),48 layers |
| Action DiT | LTX-isomorphic 0.92B,hidden=512,inner=4096(32 head × 128),48 layers |
| 架构 | MoT(Mixture of Transformers)joint attention,video + action expert 双轨 |
| Text encoder | Gemma-3-12B-IT,**训练时不加载**(用离线 cache `text_embeds_cache/robotwin`) |
| VAE | Wan2.1-VAE,**每 step 都跑 encode**(不是离线 latent) |

## 数据(RoboTwin 2.0)

| | |
|---|---|
| Task | robotwin_uncond_3cam_384_1e-4(全任务联合训练) |
| Camera | 3 × (480×640) → resize 到 (240, 320),concat 成 (384, 320) |
| num_frames | 33(action 32 + video 9,`action_video_freq_ratio=4`) |
| Action dim | 14,Proprio dim 14 |
| Norm | z-score from `dataset_stats.json`(全数据集统计) |
| context_len | 128 text tokens |

## FSDP 设置

```yaml
distributed_type: FSDP
mixed_precision: bf16
fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
fsdp_transformer_layer_cls_to_wrap: MoTBlock          # 48 个 wrap unit
fsdp_sharding_strategy: SHARD_GRAD_OP                 # ZeRO-2
fsdp_backward_prefetch: BACKWARD_PRE
fsdp_forward_prefetch: true
fsdp_offload_params: false
fsdp_state_dict_type: SHARDED_STATE_DICT
fsdp_use_orig_params: true
fsdp_sync_module_states: true
fsdp_cpu_ram_efficient_loading: true
fsdp_activation_checkpointing: true                   # external ckpt on every MoTBlock
```

**注意**:`mot_checkpoint_mixed_attn=false`(关掉了 internal ckpt 避免和 external 双重 wrap)。

## 优化器 / scheduler

| | |
|---|---|
| Optimizer | **adamw8bit**(bitsandbytes,fp32 AdamW 会 OOM) |
| LR | 1e-4 |
| LR scheduler | cosine |
| weight_decay | 1e-2 |
| max_grad_norm | 1.0 |
| max_steps | 50000 |

## Batch / 通信

| | |
|---|---|
| Micro batch | 2 per GPU |
| gradient_accumulation_steps | 8 |
| GPU 数 | 7 |
| **Effective batch** | **2 × 8 × 7 = 112** |
| 每 macro step all-gather 次数 | 48 layers × 8 micro = **384** |
| Reduce-scatter | 48(只最后一个 micro) |

## Dataloader

| | |
|---|---|
| num_workers | 16 |
| prefetch_factor | 6 |
| pin_memory + persistent_workers | 是(自动) |

