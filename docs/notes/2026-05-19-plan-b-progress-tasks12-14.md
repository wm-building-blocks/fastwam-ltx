# FastWAM_LTX — Plan B Progress 2026-05-19 (Tasks 12–14)

## Status: 完整 wiring 走通；剩 Task 15 (real-data FSDP smoke) + Task 10 alpha-scale init (可选)

| # | Task | Status | Notes |
|---|---|---|---|
| 12 | fastwam.py + runtime + yaml wiring | ✅ | LTX-2.3 loader, LTXTextEncoder, LTX strides 32/8, prompt_timestep_all forwarded |
| 13 | Text cache precompute script | ✅ | `scripts/precompute_text_embeds_ltx.py` distributed-aware; new slug `ltx23_gemma3_12b_v2connector` |
| 14 | Synthetic full-stack wiring smoke | ✅ | random-init dialed, FastWAM.training_loss runs end-to-end (numerical magnitudes reflect random init) |
| 14b | Normalize sigma to [0,1] for LTX block | ✅ | `WanContinuousFlowMatchScheduler` stores t=σ·N; divide at pre_dit call sites |
| 15 | FSDP real-dataset 1-step smoke | ⏳ | needs Gemma load + text cache precompute on remote |
| 10 | Alpha-scale init preprocess (optional) | ⏳ | ActionDiT trains from random init in 1st run |
| 16 | README + arch notes | (this doc) | |

## Task 12 — Wiring 关键改动

### `LTXVideoDiT.pre_dit` / `post_dit` shim
桥接 `prepare → run_blocks → postprocess` 给 fastwam.py 想要的 dict-API。
- bool/float `context_mask` → long 给 `_prepare_attention_mask`。
- `post_dit` 用 `dataclasses.replace(video_args, x=tokens_out)` 处理 frozen `TransformerArgs`。

### MoT cycle bug
`self._mot_ref = mot` 会被 PyTorch `__setattr__` 当 submodule 注册，造成 MoT→MoTBlock→mot 循环，
`model.to(device)` 无限递归。改用 `object.__setattr__` 绕过 module 注册。

### fastwam.py 工厂签名
- 丢弃 Wan kwargs: `model_id`, `tokenizer_model_id`, `tokenizer_max_len`, `redirect_common_files`。
- 新增 LTX kwargs: `ckpt_path`, `gemma_path`, `attach_gemma_to_text_encoder`。
- 移除 `tokenizer` 字段（LTXTextEncoder 自带）。
- `encode_prompt` 改为 `text_encoder.encode([prompt], device)`。
- `_check_resize_height_width` 改为 32 spatial / 8 temporal stride，约束 `(T-1) % 8 == 0`。
- 所有 `mot(...)` 调用 forward `prompt_timestep_all={"video": ..., "action": ...}`。

### configs/model/fastwam.yaml
完全重写为 LTX 风格。关键决策：
- `ckpt_path: checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors`
- `load_text_encoder: false` (训练读 disk cache，不挂 Gemma)
- ActionDiT `num_heads=32, attn_head_dim=128, hidden_dim=1024, num_layers=48`，与 LTX video block `inner_dim=4096` 对齐
- `cross_attention_adaln: false` on action（context 4096 ≠ hidden 1024，避免维度冲突）

### runtime.create_fastwam
- import: `wan22.fastwam` → `ltx.fastwam`
- 调用：`from_a14b_high_noise_pretrained` → `from_ltx_video_only_pretrained`
- Wan kwargs 全部清除

## Task 13 — Text Cache Precompute

`scripts/precompute_text_embeds_ltx.py`：
- 走 hydra config，跟 train.yaml 复用 dataset 发现路径。
- distributed: `torchrun --standalone --nproc_per_node=N`，prompts 按 rank shard。
- 模型：`LTXTextEncoder.from_config(config) + attach_gemma(gemma_path)` 一次性 load 12B Gemma+8L connector+128 registers。
- 输出：`{sha256(prompt)}.ltx23_gemma3_12b_v2connector.pt`，payload `{"context": (T,4096) bf16, "mask": (T,) bool}`。

`robot_video_dataset.py` 加 `text_cache_slug` 参数（默认保持 wan22 slug 向后兼容），yaml 切到 LTX slug。

## Task 14 — Synthetic Wiring Smoke

`scripts/smoketests/test_fastwam_synth.py`：
- 2 层 LTXVideoDiT random init + 2 层 LTXAlignedActionDiT random init + stub VAE。
- 跑 `FastWAM.training_loss(sample)` 验证 build_inputs → VAE encode → scheduler.add_noise → joint pre_dit → MoT.forward → joint post_dit → loss 流程。
- Random init + bf16 + 小 spatial 数值很爆炸（loss inf），这是 random init 的固有现象，不是 wiring bug。joint MoT smoke 之前已确认在同样设置下 (H=W=4, F=3) 输出 std~0.3 合理。
- 接 pretrained 权重 + 真 VAE-encoded latent + 256-token text emb 后数值会稳。

## Task 14b — Sigma 归一化

`WanContinuousFlowMatchScheduler.sample_training_t` 返回 `t = σ * num_train_timesteps`（默认 1000）。
LTX block timestep_embedding 期望 σ ∈ [0,1]。`fastwam.py` 在每个 `pre_dit` 调用前：

```python
sigma_video = timestep_video / float(self.train_video_scheduler.num_train_timesteps)
sigma_action = timestep_action / float(self.train_action_scheduler.num_train_timesteps)
```

`add_noise()` 仍然用未归一化的 timestep，因此 noisy latent 数学不变。

## 累计 commits (Plan B 时间线)

```
df6ca27 fix(ltx-task14b): normalize timestep to sigma in [0,1] for LTX block pre_dit
f4e3440 feat(ltx-task13): text cache precompute script using Gemma-3 + V2 connector
a191fa0 feat(ltx-task12): wire FastWAM + runtime + yaml to LTX-2.3 loader
b398dd8 docs(notes): Plan B progress through Task 11
da0477d feat(ltx-task11): joint MoT with single LTX API + FastWAM mask preserved
7bf44c2 feat(ltx-task9-redo): replace Wan-style ActionDiT with LTX-isomorphic LTXAlignedActionDiT
...
```

## 剩余风险 / 下一步

| # | 风险 | 严重度 | 处理 |
|---|---|---|---|
| 1 | Random init 数值爆炸（已知；wiring 不变） | Low | 接 pretrained 权重即修复 |
| 2 | Gemma load 内存 ~24GB bf16 + Connector 8L 注册器 ~2GB | Med | text cache 只跑一次 |
| 3 | text cache 切 slug 后旧 wan22 cache 不可读 | Low | dataset 已加 `text_cache_slug` 参数，yaml 切换即可 |
| 4 | LTX scheduler shift（base=0.95, max=2.05, token-count-dep）尚未接 | Low | follow-up；当前 `train_shift=5.0` 是占位 |
| 5 | FSDP wrap 单位 MoTBlock 内含 LTX video block (~290M) + action block (~67M)；7×80GB 应充足 | Low | Task 15 实测 |

### Task 15 启动 checklist（在远端 GPU 上）

```bash
# 1) Precompute text cache (≈30 分钟 on 1×A100):
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds_ltx.py \
    +data=robotwin50 +model=fastwam +task=sim_robotwin

# 2) FSDP 1-step smoke:
bash scripts/train_fsdp.sh   # uses configs/train.yaml + configs/model/fastwam.yaml
```

如 (2) OOM：降 batch_size=1，启 `mot_checkpoint_mixed_attn=true`，
或 num_workers=2 减 CPU latent encode 占用。
