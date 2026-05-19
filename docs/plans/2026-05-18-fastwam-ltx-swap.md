# FastWAM_LTX: 用 LTX-2 (video-only, 跳过 audio) 替换 Wan2.2-A14B 主干

> 起草: 2026-05-18
> 参照: `2026-05-13-fastwam-a14b-swap.md` (主干切换思路)、`2026-05-14-fastwam-a14b-train-optim.md` (训练优化)、`2026-05-15-fastwam-a14b-fsdp-via-motblock.md` (MoT/FSDP)
> 目标新仓库: `/home/admin/fang/FastWAM_LTX/` (与 `FastWAM_14B/` 并列)

---

## 1. Context

`FastWAM_14B` 当前主干是 Wan2.2-T2V-A14B 的 high-noise expert。我们要做一个并列实验仓库 `FastWAM_LTX`, 把视频主干换成 **LTX-2** (Lightricks 22B 多模态视频模型, https://github.com/Lightricks/LTX-2 / `Lightricks/LTX-2.3`)。

**用户硬约束**:
- **不加载 audio 模块的参数** (LTX-2 把 audio 5B 和 video 14B 在同一个 safetensors 文件里, 但前缀分离: `audio_*` / `video_to_audio_attn` / `audio_to_video_attn` 等)
- 与 `FastWAM_14B` 并列, 不动 `FastWAM_14B`
- 沿用 FastWAM 的 **MoT + ActionDiT** 训练框架, 只换视频主干

**Intended outcome**: 在 RoboTwin / LIBERO 上能用 LTX-2 video-only DiT 作为 FastWAM 视频专家完成 1 步前向 + 1 步训练; 不下载 / 不实例化 / 不优化 audio 相关 5B 参数。

---

## 2. LTX-2 与 Wan2.2-A14B 关键差异 (Phase 1 调研结果)

| 维度 | Wan2.2-A14B (FastWAM_14B 当前) | LTX-2.3 (目标) | 影响 |
|---|---|---|---|
| 仓库 | `Wan-AI/Wan2.2-T2V-A14B` (HF) | `Lightricks/LTX-2.3` (HF, 46GB 单文件) | 下载/registry 改 |
| 架构 | 单流 DiT, Wan 风格 | **双流 DiT** (BasicAVTransformerBlock), Self-Attn → Text X-Attn → **Audio<->Video X-Attn** → FFN | 块结构不同, 需新 block 类 |
| 维度 | 5120 hidden / 40 layers / 40 heads / head_dim=128 | **4096 hidden / 48 layers / 32 heads / head_dim=128** | 维度变小但层数变多; MoT pair 关系需重算 |
| In/out channels | 16 (Wan2.1-VAE z_dim) | **128** (LTX VAE z_dim) | VAE conditioning 全改 |
| VAE | Wan2.1-VAE (z_dim=16, stride 4×8×8) | **LTX video VAE** (z_dim=128, 估 8×32×32 stride) | 完全新 VAE 类 |
| 文本编码器 | UMT5-XXL (4096 dim, T5 tokenizer) | **Gemma-3-12B-it** (`google/gemma-3-12b-it-qat-q4_0-unquantized`) | text_dim 可能不同, tokenizer 完全不同 |
| State dict | DiffSynth 转 diffusers 风格 | **Lightricks 自定义** (非 diffusers, 非 transformers) | 写新转换器 |
| Scheduler | FlowMatch shift=12.0 sigma_floor=0.875 | **Flow matching, per-token timesteps**, shift 未公开 | 新调度器或参数 |
| Audio | 无 | **5B 参数, 同文件不同前缀** (`audio_*`, `video_to_audio_attn`, `audio_to_video_attn`) | 必须按前缀过滤 |
| License | Apache-style | **LTX-2 Community License** (商业受限, <$10M revenue 免费) | 标 license, 不发布 |

**关键观察**:
1. **LTX 主干层级 48 ≠ Wan A14B 40**: ActionDiT 必须随之改成 48 层, MoT 配对、`preprocess_action_dit_backbone.py` 的 alpha-scale 重算。
2. **LTX 块内部有 `audio_to_video_attn`**: 即便 `model_type=VideoOnly`, video block 仍然引用 audio stream KV。**要么删这一层, 要么把它 stub 成 no-op**。
3. **LTX 用 per-token timestep + 双流 RoPE**: 现有 `WanContinuousFlowMatchScheduler.sample_training_t` (一个 batch 共享 t) 不直接适配, 需要决定是退化到单 t 还是真正用 per-token。**初版用单 t 退化** (简单, 与 FastWAM 习惯一致)。
4. **License**: 私有训练 OK; 不要把权重 push 到任何公开仓库。

---

## 3. 设计

### 3.1 仓库结构: 复制 FastWAM_14B → FastWAM_LTX, 然后改

```
/home/admin/fang/FastWAM_LTX/
├── src/fastwam/
│   ├── models/
│   │   ├── wan22/             # 保留 (临时), 让 import 不崩, 真训练只走 ltx
│   │   └── ltx/               # 新增 — LTX-2 backbone 适配层
│   │       ├── __init__.py
│   │       ├── ltx_video_dit.py        # video-only LTX DiT 类 (我们自己的封装)
│   │       ├── ltx_video_vae.py        # LTX VAE 适配 (encode/decode 暴露 z_dim/upsampling_factor)
│   │       ├── ltx_text_encoder.py     # Gemma-3-12B 文本编码器 + tokenizer 包装
│   │       ├── action_dit.py           # 拷自 wan22, 改成 48 层
│   │       ├── mot.py                  # 拷自 wan22, MoTBlock 与 LTX block 配对
│   │       ├── fastwam.py / fastwam_idm.py / fastwam_joint.py  # 入口拷贝改名
│   │       └── helpers/
│   │           ├── loader.py           # load_ltx2_video_only_components
│   │           ├── state_dict_converters.py  # ltx_video_dit_filter_audio
│   │           └── ltx_modeltype.py    # 复用上游 LTXModelType 枚举
│   ├── runtime.py             # create_fastwam → 路由到 ltx.fastwam
│   └── trainer.py             # 不动
├── configs/
│   ├── model/fastwam.yaml          # 替换成 LTX 维度
│   └── task/                       # 沿用 robotwin/libero
├── third_party/
│   └── ltx-core/              # 把 Lightricks LTX-2 仓库 vendor 进来 (git submodule 或 pip install -e)
├── scripts/                   # 沿用 + 改 preprocess_action_dit_backbone (48 层)
└── docs/plans/2026-05-18-fastwam-ltx-swap.md   # 本文
```

**`models/wan22/` 保留原因**: 减少初始化时的 import 抖动; 真正训练只走 `models/ltx/`。后续若稳定, 删 wan22 子树。

### 3.2 LTX 上游集成方式

不重写 LTX-2 transformer 内部数学。直接把 LTX 仓库作为依赖, 我们只写**适配层 + audio 过滤**:

```bash
# Option A (推荐): git submodule
cd FastWAM_LTX/third_party/
git submodule add https://github.com/Lightricks/LTX-2.git ltx-2
# 然后 pip install -e third_party/ltx-2/packages/ltx-core (开发态)

# Option B: 直接 pip install (待 LTX 上 PyPI; 当前不行)
```

我们的 `ltx_video_dit.py` 做以下事:
1. 从 `ltx_core.model.transformer.model import LTXModel, LTXModelType`
2. 实例化 `LTXModel(model_type=LTXModelType.VideoOnly, **config)`
3. 在 `forward()` 包装中, 屏蔽 `audio` 入参 (传 None)
4. 暴露与 FastWAM_14B `WanVideoDiT` 相似的属性 (`hidden_dim`, `num_heads`, `attn_head_dim`, `num_layers`, `.blocks` ModuleList) 让 MoT / ActionDiT / 预处理脚本能共用

### 3.3 Audio 参数过滤 (核心需求)

**下载阶段**: 没法跳过 — 单 safetensors 46GB 里 audio + video 混着。**全文件下载**。

**加载阶段**: 用 state_dict converter (`ltx_video_dit_filter_audio`) 过滤掉所有 audio key 后再 `load_state_dict(strict=False)`:

```python
AUDIO_KEY_PREFIXES = (
    "audio_patchify_proj",
    "audio_adaln_single",
    "audio_scale_shift_table",
    "audio_norm_out",
    "audio_proj_out",
    "audio_args_preprocessor",
)
AUDIO_BLOCK_PATTERNS = (
    "audio_attn1", "audio_attn2", "audio_ff",
    "audio_scale_shift_table", "audio_prompt_scale_shift_table",
    "video_to_audio_attn",
)
# 注意: video block 里的 "audio_to_video_attn" 也要过滤掉, 否则 video stream 也会引用 audio KV

def ltx_video_dit_filter_audio(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in AUDIO_KEY_PREFIXES):
            continue
        if any(p in k for p in AUDIO_BLOCK_PATTERNS):
            continue
        if "audio_to_video_attn" in k:
            continue
        out[k] = v
    return out
```

**实例化阶段**: 用 `LTXModelType.VideoOnly` 入参; 上游若仍构造 audio 子模块, 我们 `del model.audio_*` 或不 register 这些 submodule (具体取决于 LTXModel 的实现, Task 3 探明)。

**`audio_to_video_attn` 处理**: 这层在 video block 内。两个选项:
- A: monkey-patch BasicAVTransformerBlock.forward, 让它走纯 video 路径 (跳过 audio cross-attn)
- B: 把 audio KV 全 stub 成 `torch.zeros(0, ...)` 配合 attention 的 0 长度 fallback
- 决策: 先试 A (修改 forward), 若 LTX 上游有 `model_type=VideoOnly` 已经内置短路则不用动。**Task 4 Step 1 实测**。

### 3.4 VAE 处理

LTX video VAE 的 z_dim=128 与 Wan2.1-VAE (z_dim=16) 完全不同。但 FastWAM 的所有 VAE 调用点都通过 `vae.encode(...)` / `vae.decode(...)` / `vae.z_dim` / `vae.upsampling_factor` / `vae.temporal_downsample_factor` 抽象。

**适配层 `ltx_video_vae.py`**:
```python
class LTXVideoVAE(nn.Module):
    def __init__(self, ltx_vae: nn.Module):
        super().__init__()
        self.inner = ltx_vae
        self.z_dim = 128
        self.upsampling_factor = 32         # 估; Task 3 实测
        self.temporal_downsample_factor = 8 # 估; Task 3 实测

    def encode(self, video, device, tiled=False, **_):
        # 把 ltx_core 的 encode 接口包装成 FastWAM 习惯的签名
        ...

    def decode(self, z, device, tiled=False, **_):
        ...
```

**VAE 卷积步长 % 32 检查**: FastWAM_14B 已经把 `_check_resize_height_width` 参数化成 `spatial_stride=vae.upsampling_factor`, 所以这里只要 LTXVideoVAE 暴露 `upsampling_factor=32` 就自动生效。

### 3.5 文本编码器 Gemma-3-12B

LTX-2 用 `google/gemma-3-12b-it-qat-q4_0-unquantized`。我们写 `ltx_text_encoder.py`:

```python
class LTXTextEncoder(nn.Module):
    def __init__(self, model_id: str, device: str, torch_dtype):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.hidden_size = self.model.config.hidden_size  # 3840 for Gemma-3-12B
```

**重大影响**: Gemma-3-12B 的 hidden_size **3840 (不是 4096)**。FastWAM 整条链 (`text_dim=4096` 在 video_dit_config / action_dit_config) 都假设 4096。两个选项:
- A: 在 `LTXTextEncoder` 末尾加一层 `nn.Linear(3840, 4096)` proj (随机初始化, 跟着训练学)
- B: 把 `text_dim=3840` 一路改下去 (action_dit 也跟着改, 但 LTX 上游 cross-attn 接的就是 3840, 所以 video 这边天然对)

决策: **B 路径** — text_dim 从配置传, 各处去掉硬编码 4096。

**文本 cache**: 现有 `data/text_embeds_cache/robotwin/*.t5_len128.wan22t2va14b.pt` 完全不能复用; 新 slug `gemma3_12b_len*`。这部分先不重做 cache, 训练时 on-the-fly CPU encode (Gemma 12B 在 CPU 慢但走得通), 后续看是否需要预计算。**Task 6 暂缓**。

### 3.6 ActionDiT 与 MoT 改造

LTX 48 层 vs A14B 40 层 → ActionDiT 也要 48 层。

- `action_dit.py` 拷自 wan22, 维度 num_layers=48, num_heads=32 (match LTX), attn_head_dim=128。hidden_dim 保留 1024。
- `mot.py` 拷自 wan22, MoTBlock 对接 LTX 的 BasicAVTransformerBlock 与 ActionDiT block。**注意**: LTX block 的 forward 签名与 Wan DiTBlock 不同, MoT 现有 helper (`_split_modulation`, `_build_expert_attention_io`, `_apply_expert_post_block`) 是按 Wan DiTBlock 的 `modulation` 参数 / `norm1` / `self_attn` / `gate` / `cross_attn` / `norm2` / `norm3` / `ffn` 结构写的。LTX BasicAVTransformerBlock 用 `adaln_single` (AdaLN-Zero) + `scale_shift_table`, 命名和切法不一样。
- **MoT helper 不能直接复用** — 需要新写一组 helper 对应 LTX block 的 (Self-Attn → Text X-Attn → FFN) 三段, mixed-attention 在 self-attn 段做 (joint video+action attention)。
- **此处是最大风险点**, 因为 MoT 的核心机制是把 video / action 两个 DiT block 的 Q/K/V 在 self-attn 处 cat 起来跑联合 attention, 然后 split。LTX block 内部需要拆出 Q/K/V 让 MoT 拼接 — 可能需要把 LTX 上游的 attention 拆成 (qkv_proj → mixed_attn → o_proj) 三段, 这意味着不能直接调 `block.forward()`。

**降低风险方案**: 初版**关闭 MoT 的 mixed-attention** 用 (即 video 和 action 各自独立跑 attention, 不在 self-attn 处 mix)。这是 ActionDiT-only 路径, 牺牲 FastWAM 的"video latent 引导 action"特性。Task 7 后再决定要不要补 mixed-attn。

### 3.7 Scheduler

LTX 用 flow matching + per-token timesteps。第一版用 batch 级单 t (复用 `WanContinuousFlowMatchScheduler`), shift 用 LTX README/源码里的值 (Task 3 探明; 找不到就 shift=1.0 作 fallback, 标 TODO)。

### 3.8 FSDP / Optimizer

LTX-2 22B - 5B audio = ~17B video 参数。比 A14B+ActionDiT (~15B) 略大。沿用 FastWAM_14B 的 FSDP + MoTBlock wrap 策略:
- `fsdp_transformer_layer_cls_to_wrap: MoTBlock` 仍然适用 (我们的 MoTBlock 持有 LTX video block + ActionDiT block 一对)
- `fsdp_activation_checkpointing: true` (外部 CheckpointWrapper, 同 14B 经验)
- 7 卡 GPU sharded 显存预估: 17B fp32 m/v / 7 ≈ 30 GB/rank, GPU 80GB 应能容纳 (与 14B 经验类似)

### 3.9 不在本计划范围内
- 推理路径 (FastWAM 推理用 prefill_video_cache + forward_action_with_video_cache 双段) — 初版只验训练, 推理后续单独做
- LTX 多分辨率 / 多帧数 → robotwin/libero 固定到 384×320×33 (RoboTwin) / 224×224×33 (LIBERO) 即可
- LTX 的 distilled 变体 — 用 dev 全精度
- audio 模块的"轻量利用" (如冻结 audio 用于 zero-shot 推理) — 完全不用
- 推理 / sampling 质量评估 — 初版只看 loss 不 NaN, 不验生成视觉效果

---

## 4. 任务清单

每个 task 完成后直接 commit (默认行为), 不再每次确认。

### Task 0: 仓库初始化 (FastWAM_LTX)

**新建**: `/home/admin/fang/FastWAM_LTX/` (rsync from FastWAM_14B 或 git clone)

- [ ] Step 1: 在远端 `~/fang/` 下复制
  ```bash
  cp -r /home/admin/fang/FastWAM_14B /home/admin/fang/FastWAM_LTX
  cd /home/admin/fang/FastWAM_LTX
  rm -rf .git checkpoints data runs
  git init && git add -A && git commit -m "init: fork of FastWAM_14B as FastWAM_LTX baseline"
  ```
- [ ] Step 2: 把 LTX-2 作为 submodule 加进来
  ```bash
  mkdir -p third_party && cd third_party
  git submodule add https://github.com/Lightricks/LTX-2.git ltx-2
  cd .. && git commit -m "deps: vendor Lightricks/LTX-2 as submodule"
  ```
- [ ] Step 3: 在 fastwam conda env 里 `pip install -e third_party/ltx-2/packages/ltx-core`, 验证 `python -c "from ltx_core.model.transformer.model import LTXModel, LTXModelType; print(LTXModelType.VideoOnly)"` 不报错。
- [ ] Step 4: 把 FastWAM_LTX 加进 `/home/admin/fang/` 的 worktree 习惯里 — 后续所有 edit 都 ssh 到远端做 (沿用 14B 的 workflow)。

### Task 1: LTX-2 静态调研 (代码级)

读 `third_party/ltx-2/packages/ltx-core/src/ltx_core/model/transformer/model.py` 与 `transformer.py`, 填以下空白:

- [ ] `LTXModel.__init__` 实际 kwargs 列表 (in_channels, out_channels, num_layers, num_heads, attention_head_dim, cross_attention_dim, caption_channels, ...) — 我们的 `LTX_DIT_CONFIG` 字典要按此构造
- [ ] `LTXModelType.VideoOnly` 是否真的不实例化 `audio_*` submodule, 还是只在 forward 时短路 (前者最干净, 后者要我们手动 del)
- [ ] `BasicAVTransformerBlock.forward` 完整签名 — 看是否能从外部传 `audio=None` 跳过 audio cross-attn
- [ ] 找 video VAE 真实 stride: `packages/ltx-core/src/ltx_core/model/video_vae/` 的 config.json 或类定义中 `spatial_compression_ratio` / `temporal_compression_ratio` (估 32 / 8, 待验)
- [ ] 找文本编码器 hidden size: `caption_channels` 或 `cross_attention_dim`  (估 3840 for Gemma-3-12B, 待验)
- [ ] 找 flow matching shift / sigma 默认值: 看 `ltx-pipelines` 里 scheduler 实例化处

**输出**: 在 plan 末尾追加一节 "Task 1 探明的常量", 把以上变量逐个写入。无 commit (纯调研)。

### Task 2: 新建 `src/fastwam/models/ltx/` 目录骨架

- [ ] `mkdir -p src/fastwam/models/ltx/helpers && touch __init__.py`
- [ ] 从 `models/wan22/` 拷过来 + 改名:
  - `wan22/fastwam.py` → `ltx/fastwam.py`
  - `wan22/fastwam_idm.py` → `ltx/fastwam_idm.py`
  - `wan22/fastwam_joint.py` → `ltx/fastwam_joint.py`
  - `wan22/action_dit.py` → `ltx/action_dit.py` (后续改 48 层)
  - `wan22/mot.py` → `ltx/mot.py`
- [ ] 把这些文件里 `from .wan_video_dit import ...` 全换成 `from .ltx_video_dit import ...` (类名也跟着改 `WanVideoDiT` → `LTXVideoDiT`)
- [ ] 把 `ltx/fastwam.py` 的 `from_a14b_high_noise_pretrained` 改名为 `from_ltx_video_only_pretrained`
- [ ] **Step 4 验证**: `python -c "from fastwam.models.ltx.fastwam import FastWAM; print('import ok')"` 当然会报错 (LTXVideoDiT 还没写); 仅要求 syntax error 为 0。
- [ ] **Commit**: `feat(ltx): scaffold src/fastwam/models/ltx/ package by copying wan22 layout`

### Task 3: 实现 `ltx_video_dit.py` (video-only 适配 + audio 过滤)

**File**: `src/fastwam/models/ltx/ltx_video_dit.py`

- [ ] Step 1: 写 `LTXVideoDiT(nn.Module)`, 内部持有 `LTXModel(model_type=VideoOnly, **cfg)`
- [ ] Step 2: 暴露 FastWAM 期望的属性: `self.hidden_dim`, `self.num_heads`, `self.attn_head_dim`, `self.num_layers`, `self.blocks` (指向 inner 的 transformer blocks ModuleList)
- [ ] Step 3: forward 包装, 永远 `audio=None`
- [ ] Step 4: 写 audio 过滤的 state_dict converter (§3.3 的代码), 放 `helpers/state_dict_converters.py`
- [ ] Step 5: 处理 `audio_to_video_attn` (video block 内的 audio 引用) — 要么 monkey-patch block.forward 跳过, 要么把这层置零并冻结。**先试 monkey-patch**:
  ```python
  def _video_only_block_forward(block, hidden_states, **kwargs):
      # 调用原 forward 但绕过 audio<->video x-attn
      ...
  for blk in self.blocks:
      blk.forward = types.MethodType(_video_only_block_forward, blk)
  ```
  若 LTXModelType.VideoOnly 已经内置短路, 跳过这步。
- [ ] Step 6: 离线测试 — 下载 LTX-2.3 dev checkpoint 头 100MB 看 key 列表:
  ```bash
  huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors --local-dir checkpoints/Lightricks/LTX-2.3
  python -c "
  from safetensors import safe_open
  with safe_open('checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors', 'pt') as f:
      keys = list(f.keys())
      audio = [k for k in keys if 'audio' in k]
      print(f'total={len(keys)}, audio={len(audio)}')
      print('sample audio keys:', audio[:5])
      print('sample video keys:', [k for k in keys if 'audio' not in k][:5])
  "
  ```
- [ ] Step 7: 实例化测试 (不加载 weights):
  ```bash
  python -c "
  from fastwam.models.ltx.ltx_video_dit import LTXVideoDiT
  m = LTXVideoDiT(num_layers=48, num_heads=32, attention_head_dim=128, in_channels=128, ...)
  print(f'params={sum(p.numel() for p in m.parameters())/1e9:.2f}B')
  print(f'hidden={m.hidden_dim}, layers={m.num_layers}')
  # 期望 ~14-17B (去除 audio 后)
  "
  ```
- [ ] Step 8: 加载真实 weights, 验证 audio 过滤后 `missing_keys` / `unexpected_keys` 都是 "audio_*" 前缀或为空:
  ```python
  filtered = ltx_video_dit_filter_audio(full_state_dict)
  missing, unexpected = model.load_state_dict(filtered, strict=False)
  assert all('audio' in k for k in unexpected), f"unexpected non-audio keys: {[k for k in unexpected if 'audio' not in k]}"
  ```
- [ ] **Commit**: `feat(ltx): LTXVideoDiT adapter with audio param filtering (load video-only weights)`

### Task 4: 实现 VAE 与 Text Encoder 适配

- [ ] Step 1: `src/fastwam/models/ltx/ltx_video_vae.py` — wrap ltx-core 的 video VAE, 暴露 `z_dim=128`, `upsampling_factor` (Task 1 实测填入), `temporal_downsample_factor` (Task 1 实测填入), encode/decode 签名对齐 FastWAM 习惯
- [ ] Step 2: `src/fastwam/models/ltx/ltx_text_encoder.py` — 包装 `transformers.AutoModel.from_pretrained("google/gemma-3-12b-it-...")`, 暴露 `.hidden_size`, `__call__(text_list) -> (embed, mask)` 与 FastWAM 调用习惯对齐
- [ ] Step 3: 写 `ltx_video_vae` shape smoke (33 frames × 384×320 → expect latent shape with z=128)
- [ ] Step 4: 写 `ltx_text_encoder` smoke (encode 一个 prompt 看 shape)
- [ ] **Commit**: `feat(ltx): VAE + Gemma-3-12B text encoder adapters`

### Task 5: 实现 `loader.py` — `load_ltx2_video_only_components`

**File**: `src/fastwam/models/ltx/helpers/loader.py`

- [ ] Step 1: 拷 `wan22/helpers/loader.py` 大框架, 替换 registry 内容:
  ```python
  LTX_MODEL_REGISTRY = [
      {
          "model_hash": "<填 LTX-2.3 dev safetensors 的 md5>",
          "model_name": "ltx_video_dit",
          "model_class": LTXVideoDiT,
          "model_class_kwargs": {... },  # Task 1 调研得来
          "state_dict_converter": ltx_video_dit_filter_audio,
      },
      {
          "model_name": "ltx_video_vae",
          "model_class": LTXVideoVAE,
          # ltx vae 走 ltx-core 自己的 loader, 这里只 register class
      },
      {
          "model_name": "ltx_text_encoder",
          "model_class": LTXTextEncoder,
          "model_class_kwargs": {"model_id": "google/gemma-3-12b-it-qat-q4_0-unquantized"},
      },
  ]
  ```
- [ ] Step 2: 写 `load_ltx2_video_only_components(device, torch_dtype, model_id="Lightricks/LTX-2.3", dit_config, ...)`, 返回 `LTXLoadedComponents(dit, vae, text_encoder, tokenizer, dit_path, vae_path, ...)`
- [ ] Step 3: 在 `dit` 加载阶段, 注入 `state_dict_converter=ltx_video_dit_filter_audio`
- [ ] **Commit**: `feat(ltx): load_ltx2_video_only_components with audio-filtering registry`

### Task 6: 修改 `runtime.py` + `configs/model/fastwam.yaml`

- [ ] Step 1: 在 `src/fastwam/runtime.py` 的 `create_fastwam` / `create_fastwam_idm` / `create_fastwam_joint` 里, 把 `from fastwam.models.wan22.fastwam import FastWAM` 换成 `from fastwam.models.ltx.fastwam import FastWAM`
- [ ] Step 2: 改 `configs/model/fastwam.yaml`:
  ```yaml
  _target_: fastwam.runtime.create_fastwam
  model_id: Lightricks/LTX-2.3
  load_text_encoder: false  # 与 14B 一致, text 走 dataset cache
  video_dit_config:
    in_channels: 128
    out_channels: 128
    num_layers: 48
    num_heads: 32
    attention_head_dim: 128
    hidden_dim: 4096
    caption_channels: 3840         # Gemma-3-12B hidden (待 Task 1 确认)
    # 其他 LTX 特有字段
  action_dit_config:
    hidden_dim: 1024
    num_layers: 48                 # match LTX
    num_heads: 32                  # match LTX
    attn_head_dim: 128
    text_dim: 3840                 # match Gemma
    ffn_dim: 4096
  video_scheduler:
    train_shift: 1.0               # 占位; Task 1 探明后改
    infer_shift: 1.0
    num_train_timesteps: 1000
    sigma_floor: 0.0               # LTX 单 expert, 不做 σ 截断
  action_scheduler:
    train_shift: 5.0
    sigma_floor: 0.0
  ```
- [ ] Step 3: 同样改 `fastwam_idm.yaml` / `fastwam_joint.yaml`
- [ ] Step 4: 验证 hydra parse:
  ```bash
  python -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/model/fastwam.yaml'))"
  ```
- [ ] **Commit**: `config(model): switch to LTX-2.3 video-only backbone defaults`

### Task 7: 改 MoT helper 适配 LTX block 结构

**File**: `src/fastwam/models/ltx/mot.py`

LTX block 用 AdaLN-Zero (`scale_shift_table`) 而不是 Wan 的 modulation 6-shift, attention 是 `audio_attn1` (self) + `audio_attn2` (text cross) + `audio_ff` 三段。video stream 是 `attn1` (self) + text cross + ff。

- [ ] Step 1: 在 `mot.py` 里**新写**: `_split_scale_shift(block, t)` (从 `block.scale_shift_table + t.unsqueeze(...)` 取 modulation params) — 替代 wan 的 `_split_modulation`
- [ ] Step 2: 新写: `_build_expert_attention_io_ltx(block, x, freqs, t_mod, use_gc)`, 走 `block.attn1.{q,k,v}` 路径 (LTX 用 `to_q/to_k/to_v` 命名) — 这里需要看 Task 1 调研结果决定具体改造
- [ ] Step 3: 新写: `_apply_expert_post_block_ltx(block, x, attn_out, ...)`, 处理 attn 输出 → text x-attn → ff 的剩余流程
- [ ] Step 4: MoTBlock._joint / _video_prefill / _action_with_kv 调用新 helper
- [ ] **风险缓解**: 若 LTX block 内部结构难以拆解 (qkv 与 mixed_attn 之间没法切口), **降级方案**: MoT 退化为 "video block 独立跑 → 把中间 hidden 当作 cross-attn key 给 action block 用", 不在 self-attn 处 mix。这是 ablation, 单独立一个开关 `mot.mixed_attention_mode: "joint" | "ablation"`, 第一版默认 ablation, 跑通后再做 joint。
- [ ] **Commit**: `feat(ltx-mot): rewire MoTBlock for LTX BasicAVTransformerBlock structure (ablation mode default)`

### Task 8: ActionDiT preprocess (48 层)

- [ ] Step 1: 修改 `scripts/preprocess_action_dit_backbone.py` — 主入口换成 `load_ltx2_video_only_components`, 输出 path 改成 `checkpoints/ActionDiT_linear_interp_LTX_alphascale_1024hdim.pt`
- [ ] Step 2: 运行 (估 5-10 分钟):
  ```bash
  python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output checkpoints/ActionDiT_linear_interp_LTX_alphascale_1024hdim.pt
  ```
- [ ] Step 3: 验证 `meta.num_layers=48, num_heads=32, attn_head_dim=128, text_dim=3840` (或实测值)
- [ ] **Commit**: `feat(preprocess): regenerate ActionDiT backbone for LTX (48 layers, 32 heads)`

### Task 9: 1-step forward smoke (随机 latent, 不走 dataloader)

**File**: `scripts/smoketests/test_ltx_load.py`

- [ ] Step 1: 写脚本 — 实例化 FastWAM (LTX 路径), 喂随机 video latent + random text embedding (shape: 1, len, 3840), 调 `model.training_loss(sample)`
- [ ] Step 2: 跑, **要求**: 不 NaN, 不 OOM (用 7 卡 FSDP, 单卡 80GB 应该够)
- [ ] **若 OOM**: 走 `fsdp_offload_params: true` 中间档, 或 batch_size=1
- [ ] **若 NaN**: 检查 audio 过滤是否把某些必需 video 层一起干掉了; grep `model.state_dict()` 里是否有 video 层完全空
- [ ] **Commit**: `test: smoke test LTX video-only 1 forward step on synthetic data`

### Task 10: 1-step FSDP train smoke (走真实 dataloader)

- [ ] Step 1: 沿用 `scripts/train_fsdp.sh`, max_steps=1, 不 save:
  ```bash
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash scripts/train_fsdp.sh 7 \
    task=robotwin_uncond_3cam_384_1e-4 max_steps=1 wandb.enabled=false \
    output_dir=./runs/_smoke/ltx_$(date +%s)
  ```
- [ ] Step 2: 期望: 1 step loss 输出, 无 NaN, 无 OOM, 退出码 0
- [ ] Step 3: 记录 step time, 与 FastWAM_14B FSDP (15 s/step) 对比
- [ ] **Commit**: `test: 1-step FSDP smoke for LTX backbone on RoboTwin (records step time)`

### Task 11: README + 文档

- [ ] Step 1: 写 `FastWAM_LTX/README.md` 顶部说明:
  - 这是 FastWAM_14B 的 LTX-2 实验分支
  - LTX-2 license 限制 (LTX-2 Community License), 不要把权重提交到任何公开仓库
  - audio 模块未加载, 由 `ltx_video_dit_filter_audio` 在 state_dict 阶段过滤
  - 文本编码器是 Gemma-3-12B, 不是 UMT5-XXL
- [ ] Step 2: 在本 plan 末尾追加 "Task 1 探明的常量" 真实数值
- [ ] **Commit**: `docs: FastWAM_LTX README + LTX-2 architecture notes`

---

## 5. Critical Files

**新建** (全部在 `FastWAM_LTX/` 下):
- `src/fastwam/models/ltx/ltx_video_dit.py` — LTX video DiT 适配 + audio 过滤包装
- `src/fastwam/models/ltx/ltx_video_vae.py` — VAE 适配 (暴露 z_dim/upsampling_factor)
- `src/fastwam/models/ltx/ltx_text_encoder.py` — Gemma-3-12B 包装
- `src/fastwam/models/ltx/helpers/loader.py` — `load_ltx2_video_only_components` + registry
- `src/fastwam/models/ltx/helpers/state_dict_converters.py` — `ltx_video_dit_filter_audio`
- `src/fastwam/models/ltx/{mot,action_dit,fastwam,fastwam_idm,fastwam_joint}.py` — 从 wan22 拷贝改造
- `scripts/smoketests/test_ltx_load.py`
- `third_party/ltx-2/` (submodule)

**修改**:
- `src/fastwam/runtime.py` — 改 import 路径到 `models/ltx/`
- `configs/model/fastwam{,_idm,_joint}.yaml` — LTX 维度 + Gemma text_dim
- `scripts/preprocess_action_dit_backbone.py` — 入口换 LTX loader
- `configs/task/*.yaml` — 不动 (batch / grad_accum 沿用 14B; 若 OOM 再改)

**不动** (架构保留):
- `src/fastwam/models/wan22/` 整树保留 (作为参照, 后续 stable 再删)
- `src/fastwam/trainer.py`
- `src/fastwam/datasets/` (LIBERO / RoboTwin dataset 都用 dataset 内部计算的 text cache; cache slug 自然按 model_id 区分)

---

## 6. End-to-end Verification

1. **import**: `python -c "from fastwam.models.ltx.fastwam import FastWAM; print('ok')"`
2. **audio 过滤正确**: load 后 `unexpected_keys` 应全是 audio 前缀; `missing_keys` 应为空 (或仅是 ActionDiT 新增层)
3. **forward 通**: synthetic input, 1 forward, 无 NaN (Task 9)
4. **FSDP train 1-step**: 真 dataloader + RoboTwin, max_steps=1, 退出 0, loss 非 NaN (Task 10)
5. **ActionDiT preprocess**: 48 层输出 (Task 8)
6. **License**: README 标注, 权重不进 git

---

## 7. 风险与已知未决

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| 1 | LTX 上游 API 不稳定 (Lightricks 仓库还在快速变化) | High | 用 git submodule 锁版本; 在 README pin commit hash |
| 2 | LTX 没有正式 diffusers 集成, ltx-core 包是不是真的可以单独 pip install -e | High | Task 0 Step 3 实测; 若不行, 退路是手抄 `LTXModel` 关键代码到 `models/ltx/ltx_upstream.py` |
| 3 | `audio_to_video_attn` (video block 内引用 audio KV) 没法干净跳过 | High | Task 3 Step 5; 实在不行就跑 model 时全程把 audio_kv 喂 `torch.zeros(B, 1, hidden)` (1-token 假 audio), 让 forward 跑通 |
| 4 | LTX block 与 Wan DiTBlock 内部结构差异太大, MoT helper 重写工作量爆炸 | High | Task 7 降级方案: ablation 模式 (无 mixed-attention), 先跑通再优化 |
| 5 | LTX 17B + ActionDiT 1.5B 总参数比 14B+0.6B 大 ~30%, 7 卡 FSDP 显存可能不够 | Medium | 沿用 `mot_checkpoint_mixed_attn=false` + 外部 FSDP ckpt; 若仍 OOM 切 `fsdp_offload_params: true` |
| 6 | Gemma-3-12B 单独占 ~24GB VRAM, 与 DiT 抢显存 | Medium | text encoder 强制 CPU encode (与 14B 一致, dataset workers 干这事); 不上 GPU |
| 7 | 双流 RoPE + per-token timestep 在 batch 单 t 调度下数学不等价 | Low | 第一版接受, 训练能跑就行; 后续若收敛慢再补 |
| 8 | LTX license 限制商业 / 不允许竞品 | Low | 私有实验, 标 license, 不发布 |

---

## 8. Open Questions (Task 1 调研需要回答的)

1. `LTXModel.__init__` 完整 kwargs 签名 (尤其 `caption_channels`, `attention_head_dim` 之类)
2. `LTXModelType.VideoOnly` 是 forward-time 短路, 还是构造时就不 instantiate audio submodule
3. `BasicAVTransformerBlock.forward` 接受 `audio=None` 后行为
4. LTX video VAE 的 `spatial_compression_ratio` / `temporal_compression_ratio` 真实值
5. Gemma-3-12B 真实 hidden_size (3840? 还是其它)
6. LTX flow matching 默认 shift / sigma 配置
7. ltx-core 包能否单独 `pip install -e` 而不依赖其他 LTX 子包

---

## 9. Fallback / Out of Scope

- **Out of scope**: 推理质量评估、distilled 变体、推理多步 sampling
- **Fallback**: 若 LTX MoT mixed-attention 集成不可行, 接受 ActionDiT 与 LTX video 用 cross-attention 串联 (而非 self-attn 处 mix) — 牺牲 FastWAM 的 MoT 原始论点, 但功能上仍是 "video 主干 + action 头" 的两段 DiT
- **Fallback²**: 若 Gemma-3-12B 集成成本过高, 暂时**让 dataset 直接喂 random 4096-d 文本向量** (debug 用), 把 LTX 跑通后再补 text encoder

---

## 10. 与 FastWAM_14B 的代码同步策略

- FastWAM_LTX 是 fork, **不再回流到 FastWAM_14B**
- 重要 trainer / dataset 改动若双向都需要, 手动 cherry-pick
- `third_party/ltx-2` 是 submodule, 升级 LTX 时单独 `git submodule update --remote`

---

# 📑 Phase 1 调研结果 (2026-05-19 by Claude, post code-read)

## A. 修正与新发现（**覆盖原 §2 / §3 中的猜测**）

### A.1 VideoOnly 不需要担心 audio 引用
原 §3.3 担心 `audio_to_video_attn` 在 video block 内引用 audio KV。**结论：不存在**。

代码事实（`ltx_core/model/transformer/transformer.py` `BasicAVTransformerBlock.__init__`）：
- `self.attn1` (self-attn), `self.attn2` (text cross-attn), `self.ff` 仅在 `video is not None` 时构造；
- `self.audio_attn1/2/ff` 仅在 `audio is not None` 时构造；
- `self.audio_to_video_attn` / `self.video_to_audio_attn` 仅在 **同时 video 和 audio** 时构造；
- `forward` 用 `run_vx`/`run_ax`/`run_a2v`/`run_v2a` 标志守卫。VideoOnly 时 `audio=None`, `run_ax/run_a2v/run_v2a` 均为 False，跳过所有 audio 路径。

`ltx_core/model/transformer/model.py` `LTXModel.__init__` 也用 `model_type.is_audio_enabled()` 守卫 `_init_audio()` / `_init_audio_video()`，VideoOnly 时**根本不实例化任何 `audio_*` submodule**。

**影响**：
- `helpers/state_dict_converters.py` 的 audio 过滤仍然必要（checkpoint 文件里物理上有 audio 张量），但实例化端干净。
- 原 plan §3.3 的 monkey-patch / `audio_to_video_attn` 处理 → **删除**。

### A.2 LTX DiT cross_attention_dim 是 4096, 不是 3840
原 §3.5 / §3.6 写 text_dim=3840 错误。

实际 (safetensors `__metadata__.config` `transformer` 段)：
```
caption_channels: 3840      # Gemma hidden_size (输入侧)
cross_attention_dim: 4096   # DiT 内部 cross-attn KV 维度
```

中间存在两件事将 3840 投到 4096：
1. **V2 FeatureExtractor** (`ltx_core/text_encoders/gemma/encoders/encoder_configurator.py`)：
   - 输入：Gemma 全部 49 层 hidden states (含 embedding 层) 沿 channel cat → shape `(B, T, 49*3840)`
   - 单个 `nn.Linear(49*3840, 4096, bias=True)` 投到 4096
   - 配置 key: `caption_proj_before_connector=True`, `text_encoder_norm_type="per_token_rms"`
2. **Connector** — 一个**独立的 8-layer transformer** (`connector_num_layers=8`, `connector_num_attention_heads=32`, `connector_attention_head_dim=128`, 共 128 个 learnable register tokens, `connector_apply_gated_attention=true`, `connector_norm_output=true`)，把 (text_emb, 128 registers) 一起跑过 8 层后输出 4096-d 序列，再给 DiT 当 KV。

**影响**：
- ActionDiT 的 `text_dim` 保持 **4096**（与 FastWAM_14B 当前一致，无需改），plan §3.6 的 "改成 3840" 删除。
- 文本编码栈不是简单的 `AutoModel.from_pretrained(gemma).last_hidden_state`，而是
  ```
  prompt -> Gemma3ForConditionalGeneration -> 49-layer hidden states
        -> FeatureExtractorV2 (per-token RMS + Linear → 4096)
        -> Connector (8-layer Transformer + 128 learnable regs)
        -> DiT cross-attn KV (4096-d)
  ```
- 最简实现：**直接复用 ltx-core 的 `GemmaTextEncoder` + `EmbeddingsProcessor` 整个栈**，我们包一层即可。手抄不划算。

### A.3 LTXModel 完整 kwargs（针对 LTX-2.3-22b-dev）
```python
# 来自 safetensors __metadata__.config["transformer"]
LTX_DIT_CONFIG = dict(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=32,
    attention_head_dim=128,
    in_channels=128,
    out_channels=128,
    num_layers=48,
    cross_attention_dim=4096,        # DiT 内部 KV 维度（不是 Gemma 3840）
    norm_eps=1e-6,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    rope_type=LTXRopeType.SPLIT,
    double_precision_rope=False,
    apply_gated_attention=True,      # 非默认; checkpoint 有 to_gate_logits 权重
    cross_attention_adaln=True,      # 非默认; block 多 prompt_scale_shift_table
    # caption_projection 由 EmbeddingsProcessorConfigurator 在外部构造再传入
)
# Audio 字段全部省略（VideoOnly 不用）
```
新发现的 kwargs（plan §4 Task 1 已经"想搞清楚"的，这里直接定值）：
- `apply_gated_attention=True` → Attention 类多 `to_gate_logits` Linear，每个 attention 输出乘 `2*sigmoid(gate)`
- `cross_attention_adaln=True` → 每 block 多 `prompt_scale_shift_table (2, dim)`，模型 root 多 `prompt_adaln_single` MLP
- `causal_temporal_positioning=True` → 影响时间 RoPE 计算（保留 LTX 上游行为）
- `connector_*` 一组参数 → 全部走 LTX 上游的 connector 模块

### A.4 LTX VAE 配置（从 safetensors metadata 提取，已确定）
```python
VAE_CONFIG = dict(
    _class_name="CausalVideoAutoencoder",
    dims=3, in_channels=3, out_channels=3,
    latent_channels=128,
    patch_size=4,
    causal_decoder=False,
    encoder_blocks=[
        ("res_x",              {"num_layers": 4}),
        ("compress_space_res", {"multiplier": 2}),   # ÷2 H,W
        ("res_x",              {"num_layers": 6}),
        ("compress_time_res",  {"multiplier": 2}),   # ÷2 T
        ("res_x",              {"num_layers": 4}),
        ("compress_all_res",   {"multiplier": 2}),   # ÷2 T,H,W
        ("res_x",              {"num_layers": 2}),
        ("compress_all_res",   {"multiplier": 1}),   # no-op (但仍有 res blocks)
        ("res_x",              {"num_layers": 2}),
    ],
)
```
**实际压缩比**：
- 空间：patch_size 4 × compress_space_res 2 × compress_all_res 2 = **16**（不是 plan 原估的 32）
- 时间：compress_time_res 2 × compress_all_res 2 = **4**（plan 原估 8，错）

**影响**：
- `LTXVideoVAE.upsampling_factor = 16`，`temporal_downsample_factor = 4`
- 现 FastWAM_14B `_check_resize_height_width(spatial_stride=...)` 直接读 `vae.upsampling_factor`，传 16 即可
- RoboTwin 384×320×33: latent 形状预估 (B, 128, 9, 24, 20)（时间 33→9 含 causal）

### A.5 Scheduler — LTX2Scheduler
位置：`ltx_core/components/schedulers.py::LTX2Scheduler`
```python
LTX2Scheduler.execute(
    steps,
    latent=...,
    max_shift=2.05,    # 默认
    base_shift=0.95,   # 默认
    stretch=True,
    terminal=0.1,
    default_number_of_tokens=4096,
)
# sigma_shift = base_shift + (tokens - 1024) * (max_shift - base_shift) / (4096 - 1024)
```
- **token-count-dependent shift**: 训练用 RoboTwin latent tokens ≈ 9*24*20 = 4320 ≈ MAX_SHIFT_ANCHOR → shift ≈ 2.05
- 训练用单 t 退化（与 FastWAM 一致）: 直接采样一个 sigma per batch，跳过 stretch 逻辑。
- 第一版用 LTX2Scheduler 的 sigma 公式包一层，使其符合 FastWAM `ContinuousFlowMatchScheduler` 接口。

### A.6 LTX-2.3 没有独立 config.json
HF repo `Lightricks/LTX-2.3` 只放 safetensors + LICENSE + README + .gitattributes（无 `config.json` / `model_index.json`）。

**配置全部嵌在 safetensors 的 `__metadata__["config"]`（JSON 字符串）里**。读取方式：
```python
import safetensors, json
with safetensors.safe_open(path, framework="pt") as f:
    cfg = json.loads(f.metadata()["config"])
# cfg["transformer"], cfg["vae"], (audio 段也在), 等
```
LTX 上游用 `SafetensorsModelStateDictLoader` 自动读这个。我们 `loader.py` 沿用此方式即可，**不要把 LTX_DIT_CONFIG 硬编码到 yaml**。

### A.7 ModuleList 名称差异
LTX `LTXModel.transformer_blocks` ≠ Wan 的 `model.blocks`。所有引用 `m.blocks` 的代码（preprocess_action_dit_backbone, MoT 配对, FSDP wrap 名单）都要改成 `m.transformer_blocks`。建议在 `LTXVideoDiT` 包装类暴露一个 `self.blocks = self._inner.transformer_blocks` alias，把改动收敛在一点。

### A.8 LTX block 内部命名（MoT 改造的关键）
Block 结构（`BasicAVTransformerBlock`）：
```
hidden = video_in
# 1. self-attn
shift_msa, scale_msa, gate_msa = adaln_slice(scale_shift_table[0:3], timesteps)
norm = rms_norm(hidden) * (1+scale_msa) + shift_msa
hidden = hidden + self.attn1(norm, pe=rope) * gate_msa
# 2. text cross-attn (with optional adaln if cross_attention_adaln=True)
hidden = hidden + _apply_text_cross_attention(hidden, context, self.attn2, ...)
# (av cross-attn 路径 in VideoOnly 直接跳过)
# 3. ff
shift_mlp, scale_mlp, gate_mlp = adaln_slice(scale_shift_table[3:6], timesteps)
scaled = rms_norm(hidden) * (1+scale_mlp) + shift_mlp
hidden = hidden + self.ff(scaled) * gate_mlp
```
关键命名：
- `self.attn1` 自注意力（含 RoPE，`to_q/to_k/to_v/to_out`，RMSNorm q/k）
- `self.attn2` 文本交叉注意力（context_dim=4096）
- `self.ff` FFN
- `self.scale_shift_table (6, dim)` AdaLN 参数（前 3 给 self-attn，后 3 给 ffn）
- `cross_attention_adaln=True` 多一块 `prompt_scale_shift_table (2, dim)`

→ **MoT 拆分点**：在 `block.attn1` 处可以 hook `to_q/to_k/to_v` 拆 QKV 做 joint attention，与 Wan 类似可行性高。所以 plan §3.6 的 "降级到 ablation 模式" 不是强制必要，但作为初版仍可保留以快速跑通。

---

## B. 决策与方案修订

### B.1 起点仓库（用户问题）
**决定：从 `FastWAM_14B` 复制**，理由：
- 14B 已有 `caption_projection` 在 DiT 内（与 LTX EmbeddingsProcessor 概念对齐）
- 14B 已有 14B 级 FSDP wrap / activation ckpt 经验
- 14B 已有 MoT helper 拆 attn 的脚手架（虽要改成 LTX 风格但结构相近）
- 14B preprocess_action_dit_backbone.py 已是大 DiT 适配版
- 从原始 `FastWAM` 起需要重做 14B 已经做过的工作（A14B-scale FSDP, big-DiT preprocess），无收益

### B.2 LTX 上游集成方式
**复用 `ltx-core` 整个 stack**，而不是只用 `LTXModel`：
- `GemmaTextEncoder` + `EmbeddingsProcessor` （Gemma → V2 FeatureExtractor → 4096-d）
- `Connector` 模块（8-layer transformer with 128 learnable registers）作为 LTXModel 的 caption_projection（在 `transformer/model.py::_init_video` 里 `self.caption_projection = caption_projection`）
- `VideoEncoder/VideoDecoder` 直接走 `VideoEncoderConfigurator` / `VideoDecoderConfigurator`
- 我们写的适配层只做：**形状/接口对齐 + audio 过滤 + 配置注入**

→ 由 `pip install -e third_party/ltx-2/packages/ltx-core` 装一次。`ltx-pipelines` / `ltx-trainer` 不装（避免无关依赖）。

### B.3 配置来源
**抛弃 plan 中 hardcoded yaml 维度的做法**，改为：
1. yaml 只指定 `model_id: Lightricks/LTX-2.3` + 几个非 LTX 配置（FastWAM 自己的 action_dit / scheduler 选项）
2. DiT 维度、VAE 配置全部从 safetensors `__metadata__.config` 读出后注入

### B.4 训练时 text encoding 路径
两条路：
- **路 1（一致性高，复杂）**：dataset 端不做文本 cache，trainer 内部跑 Gemma → EmbeddingsProcessor → 4096 序列。优点：行为与 LTX 原版完全一致；缺点：Gemma-12B + 8-layer connector 每步都跑，慢且占显存。
- **路 2（FastWAM 习惯）**：dataset 端预计算 EmbeddingsProcessor 输出（4096-d 序列）并 cache，trainer 直接读。Cache 文件 slug: `*.ltx23.gemma3_12b.connector_v2.pt`。

**决定**：第一版**路 2**。Gemma + connector 输出对 prompt 是固定的，缓存安全。等基线跑通再考虑路 1（如需 prompt augmentation）。

### B.5 MoT 改造（plan §3.6 修订）
LTX block 的 `attn1` 是标准 Attention 类（含 `to_q/to_k/to_v`），可拆 QKV → MoT joint attention 在数学上可行。
但 LTX block 的 AdaLN 形态（`scale_shift_table` + `adaln_single`）与 Wan 的 6-shift `modulation` 不同。

**两阶段路径**（取代原 §3.6 单段方案）：
- **阶段 1**（Task 7 第一版）：MoT ablation — video block 与 action block 各自跑 self-attn，跨模态信息只走 cross-attn 路径（action block 把 video latent 当作 cross-attn KV）。**先跑通**。
- **阶段 2**（如有需要，单独 follow-up plan）：实现 LTX 风格 joint self-attention，hook `block.attn1.{to_q,to_k,to_v}` 拼接 video+action token，过 `attention_function`，再 split + `to_out`。

### B.6 仓库与 LTX submodule 关系
**子模块迁移流程**：
1. 当前阶段 LTX-2 已经 clone 在远端 `/tmp/ltx-research/LTX-2/`（用于本次调研，全功能 working tree）
2. 创建 `FastWAM_LTX/` 时把这个 clone 直接 `mv` 进 `FastWAM_LTX/third_party/ltx-2/`，**不重新下载**（省 100MB+几分钟时间，并保证 commit hash 与本调研一致）
3. 在 FastWAM_LTX 的 git 中以**普通 submodule**形式登记：
   ```bash
   cd FastWAM_LTX
   git submodule add https://github.com/Lightricks/LTX-2.git third_party/ltx-2
   # 上面命令会复用已有目录（如果 commit hash 对得上）
   ```
   若 submodule add 因目录已存在而失败，备用方式：写 `.gitmodules` 手动 + `git add third_party/ltx-2`。

### B.7 不在范围（与原 plan 一致，重申）
- LTX 推理路径（FastWAM 推理双段）— 单独 follow-up
- LTX distilled / upscaler / IC-LoRA — 不用
- audio 5B 任何形式利用 — 完全不用
- 文本编码生成质量评估 — 不验

---

## C. 修订后的 Task 列表（覆盖原 §4）

> 旧 Task 0-11 保留语义，但具体动作以本节为准。每个 task 完成 → commit。

### Task 0：FastWAM_LTX 仓库初始化（**已 ready 执行**）
```bash
# 远端 admin 上
cd /home/admin/fang/
cp -r FastWAM_14B FastWAM_LTX
cd FastWAM_LTX
rm -rf .git checkpoints data runs wandb outputs
# 保留 third_party/ 若 14B 已经有；否则 mkdir
mkdir -p third_party
mv /tmp/ltx-research/LTX-2 third_party/ltx-2
git init && git add -A && git commit -m "init: fork of FastWAM_14B + vendor LTX-2 source @ master"
# 后续 submodule 化（可选; 见 §B.6 备用方式）
```

### Task 1：在 fastwam env 里安装 ltx-core
```bash
source ~/anaconda*/bin/activate fastwam   # 或 miniconda 路径
cd /home/admin/fang/FastWAM_LTX
pip install -e third_party/ltx-2/packages/ltx-core
python -c "from ltx_core.model.transformer.model import LTXModel, LTXModelType; print(LTXModelType.VideoOnly)"
```
**验证点**：import 不崩；不要装 ltx-pipelines / ltx-trainer（避开无关依赖）。

### Task 2：下载 LTX-2.3 dev 主权重 + Gemma
```bash
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors \
    --local-dir checkpoints/Lightricks/LTX-2.3
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir checkpoints/google/gemma-3-12b-it-qat-q4_0-unquantized
```
~46GB + ~25GB; 走后台 + 监控。验证 `python -c "import safetensors; ..."` 能读出 metadata.config（应与 §A.3 / §A.4 一致）。

### Task 3：创建 `src/fastwam/models/ltx/` 骨架
- 复制 `src/fastwam/models/wan22/` → `src/fastwam/models/ltx/`
- 改 import: `from .wan_video_dit import WanVideoDiT` → `from .ltx_video_dit import LTXVideoDiT`
- `fastwam.py` 入口函数 `from_a14b_high_noise_pretrained` → `from_ltx_video_only_pretrained`
- 不删 `models/wan22/` 子树（参照保留）
- **commit**: `feat(ltx): scaffold models/ltx package by copying wan22 layout`

### Task 4：`ltx_video_dit.py` — LTX video-only 适配器
- 持有 `LTXModel(model_type=LTXModelType.VideoOnly, ...)` 内部实例
- 暴露：`self.blocks = self._inner.transformer_blocks`（alias，给 MoT / preprocess 用），`self.hidden_dim`, `self.num_heads`, `self.attn_head_dim`, `self.num_layers`
- forward 永远 `audio=None` 调 `_inner(video, audio=None, perturbations=...)`
- **不**做 monkey-patch（A.1 解释）
- 写 `helpers/state_dict_converters.py::ltx_video_dit_filter_audio` 过滤 `audio_*` / `*_audio_*` / `audio_to_video_attn` / `video_to_audio_attn` / `audio_caption_projection` / `audio_args_preprocessor` / `audio_norm_out` / `audio_proj_out` / `audio_scale_shift_table` / `audio_patchify_proj` 等
- 写 `helpers/config_loader.py::load_ltx_config_from_safetensors(path) -> dict`（读 `__metadata__["config"]`）
- 写 `helpers/state_dict_renamer.py` — 把 checkpoint 的 LTXV_MODEL_COMFY_RENAMING_MAP 复用（ltx-core 已经 provide），把 comfy 风格 key 名映射到 LTXModel 标准 key
- 验证：`load_state_dict(strict=False)` 的 `unexpected_keys` 应**全部**带 audio 标志；`missing_keys` 应为空或仅 ActionDiT 自有 key
- **commit**: `feat(ltx): LTXVideoDiT adapter + audio key filter + safetensors metadata loader`

### Task 5：`ltx_video_vae.py`
- wrap LTX `VideoEncoder` / `VideoDecoder`（通过 `VideoEncoderConfigurator.from_config(cfg)` 构造）
- 暴露 `z_dim=128`, `upsampling_factor=16`（§A.4）, `temporal_downsample_factor=4`
- `encode(video, device, tiled=False, **_)` / `decode(z, device, tiled=False, **_)` 对齐 FastWAM 接口
- VAE 的 state dict key 用 ltx-core 提供的 `VAE_ENCODER_COMFY_KEYS_FILTER` / `VAE_DECODER_COMFY_KEYS_FILTER` 过滤同一份 safetensors
- 同样**注意**：VAE 权重也在那个 46GB 文件里（不是单独文件）
- 验证：33 帧 × 384×320 RGB → latent shape `(B, 128, 9, 24, 20)`（实测确认）
- **commit**: `feat(ltx): LTXVideoVAE adapter (uses ltx-core VideoEncoder/Decoder)`

### Task 6：`ltx_text_encoder.py` — Gemma + EmbeddingsProcessor
- 直接复用 ltx-core 的 `GemmaTextEncoderConfigurator` 和 `EmbeddingsProcessorConfigurator`
- 不要自己写 `AutoModel.from_pretrained(gemma)`（缺 V2 FeatureExtractor + connector）
- 暴露 `__call__(prompts: list[str]) -> (text_emb: (B, T, 4096), mask: (B, T))`
- 验证：`'a robot picking up a cube'` → shape `(1, ~256, 4096)` (含 128 learnable registers)
- **commit**: `feat(ltx): Gemma + EmbeddingsProcessor + Connector text encoder adapter`

### Task 7：`helpers/loader.py::load_ltx2_video_only_components`
- 从 `Lightricks/LTX-2.3` 一份 safetensors 同时载入 DiT / VAE encoder / VAE decoder（共享文件，用不同的 key filter）
- text encoder 独立从 `google/gemma-3-12b-it-...` 载入
- 返回 `LTXLoadedComponents(dit, vae, text_encoder, embeddings_processor, scheduler_factory, ...)`
- 注入 `state_dict_converter=ltx_video_dit_filter_audio` 给 DiT
- 配置完全从 safetensors metadata 派生（不读 yaml hardcoded 维度）
- **commit**: `feat(ltx): load_ltx2_video_only_components — single-checkpoint shared load`

### Task 8：`runtime.py` + `configs/model/fastwam{,_idm,_joint}.yaml`
yaml 极简化：
```yaml
_target_: fastwam.runtime.create_fastwam
model_id: Lightricks/LTX-2.3
dit_filename: ltx-2.3-22b-dev.safetensors
text_encoder_id: google/gemma-3-12b-it-qat-q4_0-unquantized
load_text_encoder: false      # dataset cache
action_dit_config:
  hidden_dim: 1024
  num_layers: 48              # match LTX
  num_heads: 32
  attn_head_dim: 128
  text_dim: 4096              # 与 DiT cross-attn 对齐 (§A.2)
  ffn_dim: 4096
video_scheduler:
  type: ltx2                  # 走 LTX2Scheduler
  base_shift: 0.95
  max_shift: 2.05
action_scheduler:
  train_shift: 5.0
  sigma_floor: 0.0
```
- runtime 入口 import 从 `fastwam.models.wan22.fastwam` → `fastwam.models.ltx.fastwam`
- **commit**: `config(model): switch backbone to LTX-2.3 video-only (metadata-driven)`

### Task 9：MoT helper 改造（**第一版只做 ablation 模式**）
- 新 `_split_ada_values_ltx(block, t)`：从 `block.scale_shift_table` + `t.unsqueeze(...)` slice 0:3 / 3:6
- 新 `_run_video_self_attn_ltx(block, x, pe, t)`：调 `block.attn1(rms_norm(x)*(1+scale)+shift, pe=pe) * gate_msa`
- 新 `_run_video_text_cross_attn_ltx(block, x, context, t)`：调 `block._apply_text_cross_attention(...)`
- 新 `_run_video_ff_ltx(block, x, t)`：rms_norm + scale_shift + `block.ff` + gate
- MoTBlock 第一版 mixed_attention_mode="ablation"（独立 self-attn）
- joint 模式留 TODO（hook `attn1.to_q/k/v` 拼接 + 调 `attention_function`）
- **commit**: `feat(ltx-mot): MoTBlock wired to LTX block structure (ablation-mode joint attn)`

### Task 10：ActionDiT preprocess（48 层）
- 改 `scripts/preprocess_action_dit_backbone.py` 入口 → `load_ltx2_video_only_components`
- 输出 `checkpoints/ActionDiT_linear_interp_LTX_alphascale_1024hdim.pt`
- 校验 meta：`num_layers=48, num_heads=32, attn_head_dim=128, text_dim=4096`
- **commit**: `feat(preprocess): ActionDiT backbone for LTX (48 layers, text_dim=4096)`

### Task 11：1-step synthetic smoke
- `scripts/smoketests/test_ltx_load.py`：随机 latent (B=1, 128, 9, 24, 20) + 随机 text_emb (B=1, 256, 4096)，调 `model.training_loss(...)`
- 验证：无 NaN，单卡 80GB 内
- **commit**: `test: LTX video-only synthetic 1-forward smoke`

### Task 12：1-step FSDP train smoke（真 dataloader）
- `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash scripts/train_fsdp.sh 7 task=robotwin_uncond_3cam_384_1e-4 max_steps=1 wandb.enabled=false output_dir=./runs/_smoke/ltx_$(date +%s)`
- 验证：退出 0，loss 非 NaN，记录 step time（与 14B 15s/step 对比）
- **commit**: `test: LTX FSDP 1-step train smoke on RoboTwin`

### Task 13：README + 文档
- LTX-2 Community License 注明，权重不进 git
- audio 过滤说明
- text encoder 不是 UMT5（指向 Gemma + EmbeddingsProcessor + Connector）
- 把 §A 调研结果写进 docs/notes/2026-05-19-ltx-arch-notes.md
- **commit**: `docs: FastWAM_LTX README + LTX architecture notes`

---

## D. 新发现的风险（增补到原 §7）

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| 9 | `apply_gated_attention=True` 让 Attention 多 `to_gate_logits`；如果 FastWAM 现有任何 attention 替换层（如 flash-attn）假设标准 attention 结构，会兼容性出错 | Med | 直接调 LTX 上游 Attention.forward 不替换 |
| 10 | EmbeddingsProcessor 输出长度可变（取决于 prompt + 128 registers），FastWAM dataset 文本 cache 切长机制需对齐（不能简单按 token 数截断） | Med | 第一版固定 prompt_max_tokens + 不裁 registers |
| 11 | `cross_attention_adaln=True` 增加 prompt_scale_shift_table 等参数；MoT helper 必须正确切片 `[6:9]` 而非只用 `[0:6]` | Med | A.8 已说明，Task 9 注意 |
| 12 | LTX 的 RoPE (LTXRopeType.SPLIT) + 3D positional embedding `[20, 2048, 2048]` 与 FastWAM 单 t 调度匹配尚未数学验证 | Med | 接受单 t 退化（plan §3.7），收敛性不行再改 |
| 13 | `causal_temporal_positioning=True` 意味着时间 RoPE 按帧 idx 而非中心化；与 FastWAM "use_middle_indices_grid" 概念冲突？ | Low | `use_middle_indices_grid=True` 由 LTX 上游内部处理；保留默认即可 |
| 14 | 46GB 单文件下载在内网带宽紧张时可能 4-8h | Low | Task 2 后台下载 + checksum 校验 |

---

## E. Open Questions 残留（Task 1 之外仍未确定）

1. ✅ LTXModel.__init__ kwargs — **已确定**（§A.3）
2. ✅ VideoOnly 是否构造 audio submodule — **不构造**（§A.1）
3. ✅ BasicAVTransformerBlock.forward audio=None — **行为正确**（§A.1）
4. ✅ VAE 真实压缩比 — **空间 16, 时间 4**（§A.4）
5. ✅ Gemma hidden_size — **3840**, 但 DiT 输入 4096（§A.2）
6. ✅ 默认 shift / sigma — **base 0.95 / max 2.05**（§A.5）
7. ✅ ltx-core 能否独立 pip install -e — 待 Task 1 实测，但 pyproject.toml 是 workspace 子包，预期 OK
8. ⚠️ FSDP wrap 用 `transformer_blocks` 还是 `MoTBlock`？ — 沿用 MoTBlock（与 14B 一致）
9. ⚠️ `latent` shape 中 token 顺序约定（patchify 后 (T, H, W) 顺序 vs (H, W, T)） — Task 5 smoke 时测

---

> 调研更新结束。下一步：执行 Task 0（建仓 + LTX-2 mv 进 third_party）。
