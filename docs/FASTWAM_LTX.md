# FastWAM-LTX — 架构与工程优化总览

> 单一参考文档。取代了原先散落的 `PROGRESS.md` / `notes/` / `plans/`。
> 最后更新：2026-05-22。

FastWAM-LTX 是一个机器人 world-action 模型：把 **LTX-2.3 的视频 DiT** 与一个
**ActionDiT** 通过 **MoT(Mixture-of-Transformers)联合自注意力**拼接,联合训练
视频生成 + 动作预测。代码 fork 自 FastWAM_14B(原先用 Wan2.2 backbone),已整体
替换为 LTX-2.3 backbone。

代码位置:`src/fastwam/models/ltx/{fastwam,ltx_video_dit,ltx_video_vae,ltx_text_encoder,action_dit,mot}.py`

---

# 第 1 部分:架构

## 1.1 总体结构

```
sample ──► VAE.encode(video) ──► video latent ─┐
                                                ├─► MoT (48 层联合) ──► video pred + action pred ──► loss
proprio/action ──► action tokens ──────────────┘
text(离线缓存) ──► context (T,4096) ───────────┘
```

- **总参数 ~14.7B**:视频 DiT ~13.8B + ActionDiT ~0.92B,**全量微调**(LoRA/冻结
  不可接受 —— 研究目标是让整个视频 DiT 适配 action 条件)。
- VAE、text connector 是冻结组件。

## 1.2 视频 DiT(LTXVideoDiT)

- 基座:`LTX-2.3-22b-dev.safetensors`(43GB),**只用 video 部分**,audio 子模块在
  构造时直接省略(`LTXModel` 的 `is_audio_enabled()` guard,无需 monkey-patch)。
- 视频 DiT 本体 1457 个权重 key,加载 **0 missing / 0 unexpected**。
- `hidden_dim = 4096`,`num_layers = 48`,`heads = 32 × head_dim 128 = inner 4096`。
- AdaLN-Zero 调制(`scale_shift_table` + `adaln_single`),**非** Wan 的 6-shift modulation。
- `apply_gated_attention = True`,`cross_attention_adaln = True`(LTX-2.3-22b-dev 的非默认 flag)。
- RoPE:SPLIT 类型,`theta=10000`,`middle=True`,`double=False`。
- 配置不在独立 `config.json` 里 —— 嵌在 safetensors 的 `__metadata__["config"]` JSON 中。
- 层级 API:`prepare / run_blocks / postprocess`,与 `LTXModel.forward` **逐 bit 一致**(diff=0)。

## 1.3 VAE(LTXVideoVAE)

- ltx-core 的 VideoEncoder / VideoDecoder,各 86 个 key。
- **空间 stride 32,时间 stride 8**(注意:不是早期文档误写的 16/4)。
- 128 个 latent channel。约束:`(num_frames - 1) % 8 == 0`,H/W 被 32 整除。
- 例:`384×320×33 → latent (128, 5, 12, 10)`。

## 1.4 文本编码器(LTXTextEncoder)

- Gemma-3-12B-it(`qat-q4_0-unquantized`,24GB)+ FeatureExtractorV2 + 8 层 Connector
  (含 128 个可学习 register),133 个 key。
- Gemma 输出 3840 维 → FeatureExtractorV2 投影到 **4096**(DiT cross-attn 的 inner dim)。
- 输出 `(T, 4096)`。
- **训练时不挂 Gemma** —— 文本 embedding 离线预计算落盘(见 §2.2)。

## 1.5 ActionDiT(LTXAlignedActionDiT)

LTX-isomorphic 设计,内部完全走 LTX `BasicAVTransformerBlock` 结构。

| 项 | 值 |
|---|---|
| `hidden_dim` | **512**(residual stream;早期为 1024,已缩小,见 §2.7) |
| `num_heads × attn_head_dim` | 32 × 128 = **inner 4096**(必须与视频 block 一致,供 MoT 联合注意力 concat) |
| `num_layers` | 48(必须与视频 DiT 一致) |
| `text_dim` | 4096 |
| `cross_attention_adaln` | **false** —— action hidden 512 ≠ context 4096,用 plain cross-attn |
| 参数量 | ~0.92B |

- **Option F**:文本 context 在 `pre_dit` 里一次性从 4096 重投影到 hidden(512),
  于是每个 block 的 text cross-attn(attn2)的 `to_k/to_v` 从 4096×4096 缩到
  4096×512 —— 这是 3.24B→0.92B 缩小的主要来源。`attn1` 的 inner dim 仍保持 4096。
- Action token 用 LTX 3D RoPE 的退化形式 `(frame_idx, 0, 0)`,数学上等价 1D temporal rope,
  形式上保留 3D 以匹配 video block 的联合注意力。
- **alpha-scale init**:用视频 DiT 预训练权重通过单维线性插值 + magnitude 缩放
  (`alpha = sqrt(d_src/d_target)`)初始化 ActionDiT。
  payload:`checkpoints/preprocessed/ltx_action_dit_backbone.pt`,
  脚本:`scripts/preprocess_action_dit_backbone_ltx.py`。
  `action_encoder` / `head` 保持随机初始化(skip-prefix)。
  > 注:早期 100 步对比里 alpha-init 的 warmup loss 反而略高于 random-init(跨模态
  > 迁移:视频空间统计量对 proprioceptive action 不完全对齐)。判断 init 差异会在
  > 30k 步中后期收敛掉,仍按与 Wan2.2 一致的策略采用 alpha-init。

## 1.6 MoT 联合(mot.py)

- 每个 `MoTBlock` 内含一个 `video_block` + 一个 `action_block`,共 48 层。
- **联合自注意力**:每层把 video + action 的 Q/K/V 在 inner_dim 4096 上 concat,
  一次 `scaled_dot_product_attention`,再 split 回两路。
- **FastWAM mask**:video↔video first-frame-causal、video→action False、
  action↔action True、action→first-frame-video True。
- 单 API helper(`_build_expert_attention_io_ltx` + `_apply_expert_post_block_ltx`),
  video/action 共用,无 if-else 分支。
- **参数别名**:`mot.blocks[i].{video,action}_block` 与
  `mot.mixtures.<expert>.blocks[i]` 与(video)`mot.mixtures.video._inner.transformer_blocks[i]`
  是**同一批 nn.Module**。state_dict 会沿所有路径展开 —— checkpoint 保存依赖
  `_drop_mixtures_blocks_alias` hook 丢弃别名(见 §2.5)。
- 推理:`prefill_video_cache`(48 层 video-only,缓存每层 K/V)+
  `forward_action_with_video_cache`(action Q 对 `cat([cached_video_kv, action_kv])`)。

## 1.7 调度器(flow-matching)

- **视频:`type=ltx2`** —— LTX-2 的 stretched shifted logit-normal sampler。
  `mu = lerp(0.95, 2.05, seq_len, [1024, 4096])`,即 shift 随 token 数线性变化。
  与 Lightricks 参考 trainer 一致。
- **动作:`type=wan`,`train_shift=5.0`** —— LTX 的 logit-normal shift 是为视频
  token 量设计的,robotwin action 只有 ~8-32 token,远低于 `min_tokens=1024`,
  套用 LTX schedule 等价于固定 shift。保留 Wan 占位,需单独标定后才换。
- LTX block 的 timestep embedding 期望 σ ∈ [0,1];`fastwam.py` 在每个 `pre_dit`
  调用前把 `t = σ·num_train_timesteps` 归一化回 σ(`add_noise` 仍用未归一化的 t,
  噪声数学不变)。

## 1.8 关键参数设置(`configs/`)

任务配置 `robotwin_uncond_3cam_384_1e-4.yaml`:

| 项 | 值 |
|---|---|
| `batch_size`(每卡 micro) | 2 |
| `gradient_accumulation_steps` | 8 |
| GPU 数 | 7 → **有效 batch 112** |
| `learning_rate` / scheduler | 1e-4 / cosine |
| `max_steps` | 30000(≈122 epoch over 27.5k RoboTwin2.0 demos) |
| `weight_decay` | 1e-2 |
| `save_every / eval_every / log_every` | 2500 / 500 / 10 |
| `num_workers` | 8 |
| `mot_checkpoint_mixed_attn` | false(见 §2 —— 与 FSDP 外层 checkpoint 不叠加) |

## 1.9 磁盘布局

```
~/fang/FastWAM_LTX/
├── checkpoints/
│   ├── Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors      (43GB)
│   ├── google/gemma-3-12b-it-qat-q4_0-unquantized          (24GB)
│   └── preprocessed/ltx_action_dit_backbone.pt             (ActionDiT alpha-init payload)
├── data/
│   ├── robotwin2.0/                                        (本地 NVMe)
│   └── text_embeds_cache  -> 大盘 (~1.8TB, 921,032 个 .pt)
└── runs/                  -> 大盘 (checkpoint 累积 ~200-300GB)
```

---

## 1.10 训练时 eval(在线评测)

每 `eval_every` 步触发一次 eval(`trainer.py:evaluate`),从 `val_dataset` 随机取一条 sample,产出**多块独立信号**。

### 1.10.1 各指标的计算路径

| 指标 | 含义 | 计算 |
|---|---|---|
| `val_loss` | flow-matching MSE on val sample | `model(sample)` 走 `FastWAM.training_loss` —— **跟训练 loss 同公式同输入**,只是 sample 来自 val。包含 `loss_video + lambda_action * loss_action` |
| `action_l1` / `action_l2` | action expert **自主预测**跟真值的物理 L1 / L2 差 | 独立调 `model.infer_action(...)`(**不喂 gt**)从纯噪声 denoise 出 `pred_action`,反归一化后跟真值 action 比 |
| `infer_psnr` / `infer_ssim` | 模型**端到端 rollout** 跟真值视频的相似度 | 见 §1.10.2 |
| `psnr_decode_vs_gt` / `ssim_decode_vs_gt` | VAE 自身重建质量 | 把 gt video 过 VAE encode→decode,跟原视频比 |
| eval mp4(`eval/step_NNNNNN_rank_*.mp4`) | rollout 视觉化 | 见 §1.10.2 |

### 1.10.2 mp4 / `infer_psnr` 渲染方式(2026-05-23 改动)

**改动前** —— eval mp4 用**真值 action** 当条件喂 video expert 渲染。这是 world-model 评测:"给定真实的未来动作,模型能不能正确渲染出该有的画面"。这种做法解耦 video 质量和 action 质量,`infer_psnr` 严格意义上只衡量 video expert 在 action 给定条件下的渲染保真度,不反映 policy 质量。

**改动后(当前默认)** —— eval mp4 用**模型自己预测的 action** 当条件渲染:

```
1) pred_action = model.infer_action(prompt, input_image, proprio)   # 不喂 gt
2) pred_video  = model.infer_joint(action=pred_action, ...)         # 用 pred 渲染
3) save pred_video as mp4
4) infer_psnr/ssim = compare(pred_video, gt_video)
```

mp4 反映的是 **"模型自主部署时的端到端 rollout"** —— action 错了视频会跟着错,PSNR 会掉。**`infer_psnr` 从此是一个更严格的端到端指标**,把 action 质量和 video 质量耦合衡量。

> 历史动机:50ep overfit 实验里,gt-action 渲染的 mp4 显示了完美 reach+grasp,但这部分功劳全在 video expert("被 gt action 牵着走");闭环 RoboTwin eval 实际 0/10。
> 用 pred action 渲染后,mp4 可视化才真正反映模型当下的 policy 行为。

### 1.10.3 注意事项

- **`val_loss` 不受影响**(走 `training_loss` 路径,跟 mp4 渲染完全独立)
- **`action_l1` / `action_l2` 不受影响**(本来就是 pred vs gt)
- **eval 成本多一次 `infer_action` 调用**(因为 `infer_joint` 内部默认 `test_action_with_infer_action=True` 还会再算一次 pred action 做自洽检查 —— 约 +5-10s/eval,可忽略;后续可加 `test_action_with_infer_action=False` 透传消除)
- 对比改前改后的 `infer_psnr` **不可直接比较**:语义不同,不是 regression

### 1.10.4 val_dataset 配置(overfit 特定)

`val_set_proportion=0` 时,`base_lerobot_dataset` 让 train 和 val **共用全部 episode**(不切 split)。所以 overfit 实验里 eval sample 就是从训练集里采,`val_loss` / `action_l1` 反映的是**训练集表现**,不是泛化。

---

# 第 2 部分:工程优化

## 2.0 训练速度现状

| 阶段 | 单步时间(稳态中位数) |
|---|---|
| 最初(FULL_SHARD + 未生效的 compile) | ~15.5s |
| SHARD_GRAD_OP | ~9-10s |
| **当前(SHARD_GRAD_OP + dataloader 调优)** | **~7s**(50 步实测,均值 8.2s,6-13s 抖动) |

约 **−55%**。在「必须全量微调 14.7B + 3 相机」的约束下,~7s 基本是合理地板。
(独立 LTX-2 的「~6s」是 LoRA / 单相机 / batch-1 的工作负载,**不是同一量级的对比**。)

profiler 实测一步分解(SHARD_GRAD_OP,干净 trace):
all-gather 3.8s + 真实计算 ~1.5s + GPU 空闲气泡 ~3.5s。

## 2.1 FSDP 分片策略 —— `SHARD_GRAD_OP`(最大的速度优化)

`scripts/accelerate_configs/accelerate_fsdp.yaml`:
- `fsdp_sharding_strategy: SHARD_GRAD_OP`(ZeRO-2 式)—— 前向后**不 reshard 参数**,
  砍掉反向那半 all-gather。这是 15.5s→~10s 的主要来源。
  代价是峰值显存升高(参数在 fwd→bwd 窗口内驻留),实测稳态 53-76GB / 80GB,余量够。
- `fsdp_forward_prefetch: true` —— 下一层 all-gather 与当前计算重叠。
- `fsdp_backward_prefetch: BACKWARD_PRE`。
- `fsdp_transformer_layer_cls_to_wrap: MoTBlock`,`fsdp_activation_checkpointing: true`。
- `fsdp_use_orig_params: true`,`fsdp_state_dict_type: SHARDED_STATE_DICT`。
- `mot_checkpoint_mixed_attn` 保持 **false** —— FSDP 已对每个 MoTBlock 做外层
  activation checkpoint,内层再叠一层会重复重算 + OOM。

> 实测无效的尝试:cuDNN attention 后端、`NCCL_PROTO=Simple,LL128`、单独的
> `forward_prefetch` —— 对步时无可测量改善。

## 2.2 文本 embedding 离线预计算

训练**不在循环里跑 Gemma**。`scripts/precompute_text_embeds_ltx.py`(distributed,
按 rank shard)离线把 921,032 条 prompt 编码落盘:
`{sha256(prompt)}.ltx23_gemma3_12b_v2connector.pt`,payload `{context:(T,4096) bf16, mask:(T,) bool}`。
数据集 `robot_video_dataset.py` 有 `text_cache_slug` 参数,yaml 切到 LTX slug。
> 注:视频 VAE 编码目前仍**每步在线跑**(profiler 显示 VAE 卷积仅 ~90ms,不是瓶颈;
> 离线预计算 video latent 是个保持训练目标不变的低风险待办,收益有限)。

## 2.3 torch.compile MoT blocks(`compile_mot`)

`trainer._maybe_compile_mot()`:对 48 个 `MoTBlock` **就地** `nn.Module.compile()`
(不是 `torch.compile(block)` —— 后者会换成 `OptimizedModule`,破坏 `unwrap_model`、
FSDP 的 `MoTBlock` 类型匹配、给 state_dict key 加 `_orig_mod.` 前缀)。

LTX-2 模式的 Dynamo/Inductor 旋钮(关键 —— 否则编译反复 recompile、不融合):
- `dynamic=True` —— 视频 token 数每 batch 会变,不按 shape 重编译。
- `torch._dynamo.config.cache_size_limit = 256` —— 防缓存抖动。
- `inline_inbuilt_nn_modules` / `allow_unspec_int_on_nn_module`。

效果:profiler 实测 `vectorized_elementwise` kernel **136k → 8k**(融合降 16 倍),
recompile 只在 warmup 期发生 48 次(每 block 一次)后稳定。

**结论:`compile_mot` 不开。** 两点实测:
1. SHARD_GRAD_OP 之后步时不再受 CPU 派发限制 —— 融合掉 kernel 对步时收益 **≈0**。
2. FSDP + `torch.compile(dynamic=True)` 的**首步编译极慢**:实测一次 50 步 smoke
   test 花了 **~22 分钟**才跑出第一个 step(09:33 启动→09:55 到 step 1),期间
   主线程单线程 100% CPU 卡在 Dynamo 符号 shape 推理。这正是 LTX-2 自己的 trainer
   警告过的 "FSDP + torch.compile may hang on the first training iteration"。

收益≈0 + 启动多付 ~20min,ROI 为负。默认 `compile_mot: false`(`configs/train.yaml`),
保持关闭;代码路径保留,留作后续 torch 版本改善后复评。

## 2.4 优化器:`adamw` / `adamw8bit`

`optimizer_type`(`configs/train.yaml`)。`adamw8bit`(bitsandbytes)把优化器状态
显存减半 —— SHARD_GRAD_OP 下参数驻留吃显存,裸 `adamw` 实测峰值 79.5/80GB(几乎贴顶,
eval 步易 OOM),所以长跑用 `adamw8bit`。

## 2.5 Checkpoint:adamw8bit + FSDP 的保存/恢复

`accelerator.save_state` 会把 8-bit 优化器状态送进 `FSDP.optim_state_dict`,其
`_convert_all_state_info` 假设优化器状态 tensor 与参数 numel 一致 —— bitsandbytes
的块量化状态(uint8 + absmax + qmap)违反这点,直接崩。

`trainer.py` 的解决方案(`_manual_optim_ckpt`,仅 adamw8bit+FSDP 路径生效,
adamw 路径完全不变):
- **模型**:用 accelerate 自己的 `save_fsdp_model` / `load_fsdp_model`(DCP 分片)。
- **优化器**:每个 rank 各自 `torch.save(optimizer.state_dict())` 成
  `optimizer_rank{N}.pt`,绕过 `FSDP.optim_state_dict`。恢复时校验 world_size 一致。
- **scheduler**:rank 0 单文件。
- **加载用 non-strict** `load_state_dict` —— MoT 的别名模块路径是 FSDP 占位符、
  无法被写入;non-strict 容忍其缺失,共享 nn.Module 引用让别名自动同步。
- `mot.py` 的 `_drop_mixtures_blocks_alias` save hook 正则覆盖**全部三条**别名路径
  (`mixtures.<e>.blocks.*` 和 `mixtures.video._inner.transformer_blocks.*`)。

### 2.5.1 Host RAM OOM during save —— 2026-05-24 事故 & 三层防御

**事故**:step 7500 save 触发**全局 host OOM**,rank 0 被内核 kill,**所有 checkpoint 丢失**
(无法 resume,要从 step 0 重训)。

**根因栈**:
1. 长跑后 page cache 累计到 **841GB / 2TB**(75GB 数据集 + 几次 state save 的 dirty page +
   conda env 等),空闲只剩 17GB。
2. step 7500 save 写新 state(~140GB:DCP 模型分片 + `optimizer_rank*.pt` × 7) → 先进 page
   cache → 与内核 reclaim 赛跑。
3. 写速 > 回收速 → 内存峰值突破 2TB → OOM killer 选 anon-rss 最大的 rank 0(49.6GB)kill。
4. **更糟**:`save_checkpoint` 旧逻辑是 **rotate-before-save**(先删旧 state,再写新的),
   注释里承认 "crash mid-write 会丢上一个 slot"。step_5000 在 save 前已被删,step_7500 中途
   被 kill → 双输,state/ 目录归零。

**修复(已合入 `trainer.py` + `configs/task/robotwin_uncond_3cam_384_1e-4.yaml`)**:

| 防御层 | 实现 | 失效后果 |
|---|---|---|
| **L1 — 改 rotate 顺序** | `save_checkpoint` 改成 `drop_caches → save → os.sync → rotate`。`_rotate_state_dir` / `_rotate_weights_dir` 的 `cutoff` 从 `keep_last_n - 1` 改成 `keep_last_n`(rotation 现在跑在 save 之后,新 state 已经在 entries 里)。 | crash 时旧 ckpt 永远在,**永远能 resume** |
| **L2 — save 前 drop page cache** | 新增 `_drop_page_cache_for_save`:`sync && sudo -n tee /proc/sys/vm/drop_caches <<< 1`。只丢 clean cache,无数据风险。需 passwordless sudo(已确认 admin 用户可)。 | OOM 概率从 "几小时必发" 降到接近 0(腾出 800GB headroom 给 140GB write spike) |
| **L3 — `state_keep_last_n: 2`** | yaml 默认值从 1 改成 2。稳态额外 +140GB 磁盘(968GB 空闲,无压力)。 | 即使 L1+L2 都失败,上上次 state 还在 |

**短时双份 state 期间**:save 中新+旧并存,峰值 ~280GB(state_keep_last_n=2 × 140GB) +
长期锚点 30k/40k/50k(共 420GB)= ~700GB,968GB 空闲,够。

**未修的两件**(不在 P0 范围):
- openpi 进程 host RAM 占用 40GB —— 不动(owner 约束)。
- 离线预计算 video latent(§2.2)—— 收益小、工作量大,暂留。

## 2.6 注意力后端

`src/fastwam/models/ltx/helpers/attention_backend.py` —— `FASTWAM_ATTENTION_BACKEND`
环境变量选 SDPA 后端(cudnn/flash/efficient/math/auto)。在 torch.compile 区域内
自动回退为 `nullcontext`,让 Inductor 自己选 kernel(cuDNN 的输出 stride 与
inductor meta kernel 不一致会触发 `assert_size_stride`;flash 不支持 attn_mask)。

## 2.7 ActionDiT 缩小(3.24B → 0.92B)

`hidden_dim` 1024 → 512 + Option F 文本 context 重投影(见 §1.5)。
对应 `checkpoints/preprocessed/ltx_action_dit_backbone.pt` 需用新脚本重新生成。

## 2.8 历史关键修复

| 问题 | 修复 |
|---|---|
| torchcodec 缺 FFmpeg → 退回 pyav,~3min/step | `conda install -n fastwam_ltx ffmpeg=7`,步时回到正常区间 |
| ActionDiT block `scale_shift_table` 用 `torch.empty` 未初始化 → `loss=nan` | `action_dit.py` 构造后 `nn.init.normal_(std=0.02)` |
| eval 后 `save_state` 触发 NCCL OOM(allocator 碎片) | `evaluate()` 末尾 `del` 中间张量 + `torch.cuda.empty_cache()` |
| 非有限 grad_norm 毒化权重 | sync 边界 all-gather `isfinite` flag,任一 rank 非有限则全体 skip `optimizer.step()` |
| `self._mot_ref = mot` 被注册成 submodule → `model.to()` 无限递归 | 改用 `object.__setattr__` 绕过 module 注册 |

## 2.9 Profiler

`trainer._maybe_build_profiler()` —— 环境变量 `FASTWAM_TRAINER_PROFILE=1` 开启,
`FASTWAM_TRAINER_PROFILE_FREQ` 控制 trace 间隔,`FASTWAM_TRAINER_PROFILE_RANKS`
选 rank。chrome trace 写到 `<output_dir>/profiler/`。

## 2.10 dataloader 抖动

根因:数据集每次 `__getitem__` 从 .mp4 **实时解码 3 相机视频帧**(CPU 重活、耗时
不稳),每个优化器步要 16 个新样本,worker 跟不上时主进程空等。

缓解(已生效):加大 `num_workers` + `prefetch_factor`。实测 `num_workers=16
prefetch_factor=6` 把稳态步时中位数从 **10s 降到 7s**(均值 9.3s→8.2s),6s 那一档
步数明显变多。抖动上限仍是 ~13s(comm 侧偶发尖刺,未消除),但分布整体下移。
长跑推荐 `num_workers=16 prefetch_factor=6`。

剩余抖动是通信侧的,继续深挖 ROI 很低。彻底解法是离线预计算 video latent(见 §2.2)。

## 2.11 启动命令

```bash
cd ~/fang/FastWAM_LTX
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastwam_ltx
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash scripts/train_fsdp.sh 7 \
    task=robotwin_uncond_3cam_384_1e-4 \
    optimizer_type=adamw8bit num_workers=16 prefetch_factor=6
# compile_mot 保持关闭(见 §2.3:收益≈0 且首步编译多付 ~20min)
# wandb:加 wandb.enabled=true,WANDB_API_KEY 须在 shell 里 export(不写进 ~/.netrc)
```

重新生成 ActionDiT alpha-init payload:
```bash
python scripts/preprocess_action_dit_backbone_ltx.py \
    --model-config configs/model/fastwam.yaml \
    --output checkpoints/preprocessed/ltx_action_dit_backbone.pt \
    --device cpu --dtype float32
```
