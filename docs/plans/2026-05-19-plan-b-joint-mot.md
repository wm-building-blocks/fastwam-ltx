# Plan B: 在 LTX 基础上保留 FastWAM 真 joint self-attention 的详细实施方案

> 起草：2026-05-19
> 目标：在 LTX-2.3 video DiT 与 ActionDiT 之间实现 **真正的 MoT joint self-attention**（不走 ablation 降级路径），保留 FastWAM 论文最核心的"video token 在 self-attn 内部直接看到 action token"特性。
> 前置：Tasks 1-7 已完成，LTX 三件套适配器（DiT/VAE/Text Encoder）均已在 LTX-2.3 真权重上 0/0 验证通过。

---

## 1. 关键技术约束（从代码调研得出）

### 1.1 inner_dim 是 joint attention 的拼接维度，不是 hidden_dim
Wan 现状已经如此：
- Wan22 video: hidden=5120, num_heads=40, attn_head_dim=128 → **inner_dim = 40×128 = 5120**
- Wan22 action: hidden=1024, num_heads=40, attn_head_dim=128 → **inner_dim = 40×128 = 5120**
- `block.self_attn.q = nn.Linear(hidden, inner_dim)`，Q/K/V 形状 (B, T, **inner_dim**) ←--- 在这里 cat

LTX 对照：
- LTX video: hidden=inner=4096（恰好相等，因为 head_dim=128 × num_heads=32 = 4096）
- 我们的 action 必须 num_heads=32, attn_head_dim=128 → **inner_dim = 4096**
- 选 action hidden_dim=1024（FastWAM 既有约定，不变）→ `attn1.to_q = nn.Linear(1024, 4096)`

→ Q/K/V 均 (B, T, 4096) 可以直接 cat。**joint attention 维度不变**。

### 1.2 LTX block API ≠ Wan DiTBlock API
| 操作 | Wan DiTBlock | LTX BasicAVTransformerBlock |
|---|---|---|
| AdaLN params 来源 | `block.modulation (1,6,H)` + `t_mod (B,1,6H)` | `block.scale_shift_table (6,H)` + `timestep (B,T,6H)` |
| AdaLN shift/scale 数学 | `modulate(norm1(x), shift, scale) = norm1(x)*(1+scale)+shift` | `rms_norm(x)*(1+scale)+shift`（同语义） |
| pre-attn norm | `nn.LayerNorm(eps,no-elementwise)` (`block.norm1`) | `rms_norm(x, eps)` 函数 |
| Q/K/V 投影 | `block.self_attn.{q,k,v}` Linear (no bias?) | `block.attn1.{to_q,to_k,to_v}` Linear (**bias=True**) |
| Q/K norm | `block.self_attn.{norm_q,norm_k}` (RMSNorm on inner_dim) | `block.attn1.{q_norm,k_norm}` (RMSNorm on inner_dim) — 同语义 |
| RoPE 应用 | `rope_apply(q, freqs, num_heads)` (Wan 自家 3D RoPE) | `apply_rotary_emb(q, pe, rope_type=SPLIT)` (LTX SPLIT RoPE) |
| Per-head gate | 无 | `block.attn1.to_gate_logits(x) → 2σ(gate) * out`（apply_gated_attention=True） |
| Attention 后输出 proj | `block.self_attn.o` | `block.attn1.to_out` Sequential(Linear, Identity) |
| 残差 + gate | `block.gate(x, gate_msa, attn_out) = x + gate_msa * attn_out` | `x + attn_out * gate_msa`（同语义） |
| 文本 cross-attn | `block.cross_attn(norm3(x), context, mask)` | `block._apply_text_cross_attention(x, context, block.attn2, ...)` (内嵌 AdaLN if cross_attention_adaln=True) |
| FF | `block.ffn` (Linear+GELU+Linear) | `block.ff` (FeedForward 模块) |
| FF norm | `block.norm2` LayerNorm | `rms_norm(x)` |

→ MoT helpers 全部要按 LTX 命名重写；**不能复用 wan22/mot.py 现有 helper**。

### 1.3 LTX timestep 形态：(B, T, 6×inner_dim)
LTX `adaln_single` 模块（在 LTXModel root 上）输出 `(B, T_tok_or_1, 6×inner_dim)`。块内 `get_ada_values` 把它 reshape 成 `(B, T, 6, inner_dim)` 再按 indices 切片。

FastWAM 用单 t per batch → 输入 `timestep` 形状是 `(B, 1, 6×inner_dim)`，自动广播。

→ **video 与 action 各自有独立的 AdaLN MLP 和 scale_shift_table**（dim 不同：video 4096, action 1024）。

### 1.4 ActionDiT 必须重新设计，结构必须对齐 LTX block
现有 wan22 `DiTBlock` 不能直接用，因为：
- 它用 Wan-style modulation/norm1/cross_attn 命名
- 它的 `self_attn.q/k/v` 是 bias=False，LTX 是 bias=True
- 它的 RoPE 用 Wan 的 `rope_apply` 而非 LTX 的 `apply_rotary_emb`

新设计：**`LTXAlignedActionBlock`**，与 LTX `BasicAVTransformerBlock` **同构** 但参数维度变小（hidden=1024, inner=4096），并复用 LTX 上游的 `Attention`/`FeedForward` 模块（避免重新实现 RoPE/gated-attn）。

```python
LTXAlignedActionBlock(
    attn1 = ltx_core.Attention(query_dim=1024, context_dim=None, heads=32, dim_head=128,
                                rope_type=SPLIT, apply_gated_attention=True),
    attn2 = ltx_core.Attention(query_dim=1024, context_dim=4096, heads=32, dim_head=128,
                                rope_type=SPLIT, apply_gated_attention=True),
    ff = ltx_core.FeedForward(1024, dim_out=1024),
    scale_shift_table = nn.Parameter(torch.empty(6, 1024)),  # cross_attention_adaln=True → coeff=3? 详见 §4.2
    prompt_scale_shift_table = nn.Parameter(torch.empty(2, 1024)),
)
```

**注意**：LTX block 的 `scale_shift_table` 实际是 `(adaln_embedding_coefficient(cross_attention_adaln=True)=3, dim) × 2 = (6, dim)` — 详见 §4.2。

### 1.5 RoPE 必须**按流分别**应用，再 cat
- video token：3D RoPE（time × H × W），用 `block.video.positional_embeddings`
- action token：1D RoPE（时间维度）→ 需要构造 1D `pe` 张量给 `apply_rotary_emb`
- 拼接 cat([v_q_rope, a_q_rope], dim=1) 后再 attention — RoPE 已经分别应用，attention 看的是"已旋转"的 Q/K

可行性：`apply_rotary_emb(q, pe, rope_type=SPLIT)` 只看 q 的 (B, T, H*D) shape 和 pe 的形状，不要求 q 来自哪个 stream。

### 1.6 Joint attention 的 mask 设计
**默认全 mixed-attn**（每个 video token 与每个 action token 互相可见）。但需要支持：
- video 自身的 self-attention mask（如 LTX 的 first-frame-causal）
- action 自身的 causal mask（Wan ActionDiT 已有 `action_group_causal_mask_mode`）
- video → action 方向是否设限？默认全开。
- action → video 方向是否设限？默认全开（这就是 MoT 论点：video 主干"看见" action）。

mask 形状：(B, T_video+T_action, T_video+T_action) 或更紧凑的 head-broadcast form。沿用 Wan MoT 已经设计好的 `attention_mask` 拼装逻辑（见 `mot.py::MoTBlock._joint`）即可。

---

## 2. 模块层级图

```
LTXJointMoTBlock                                  # MoT 配对单元，FSDP wrap 边界
├── video_block: BasicAVTransformerBlock          # LTX 上游块（4096-dim）
│   ├── attn1, attn2, ff, scale_shift_table, prompt_scale_shift_table
│   └── (audio_* 全部缺失，因为 VideoOnly)
└── action_block: LTXAlignedActionBlock           # 我们新写的块（1024-dim）
    ├── attn1 (1024→4096→1024), attn2, ff
    ├── scale_shift_table (6, 1024), prompt_scale_shift_table (2, 1024)
    └── 同 LTX block 结构

LTXVideoDiT                                       # 已有 (Task 4)
└── _inner: LTXModel
    ├── patchify_proj, adaln_single, scale_shift_table, norm_out, proj_out
    └── transformer_blocks: ModuleList[48] of video-side blocks

LTXAlignedActionDiT                               # 新写 (Task 10 替换)
├── action_encoder: Linear(action_dim, 1024)
├── text_embedding: (no-op or identity, since text already 4096-d via Connector)
├── adaln_single_action: AdaLayerNormSingle(1024, embedding_coefficient=3)
├── prompt_adaln_single_action: AdaLayerNormSingle(1024, embedding_coefficient=2)
├── scale_shift_table_action: (2, 1024) (final norm before head)
├── norm_out: nn.LayerNorm(1024, elementwise_affine=False)
├── proj_out: Linear(1024, action_dim) (head)
└── blocks: ModuleList[48] of LTXAlignedActionBlock  ← 被 MoT 接管

MoT (Wan22 同名)
├── mixtures = {"video": LTXVideoDiT, "action": LTXAlignedActionDiT}
├── motblocks: ModuleList[48] of LTXJointMoTBlock
└── mot.forward 编排
```

每个 `LTXJointMoTBlock` 持有一个 video block 与一个 action block 的指针。**FSDP wrap 边界仍是 MoTBlock**（保留 14B 经验）。

---

## 3. 数据流（每层 forward）

输入：
- `video_args: TransformerArgs` (LTX 标准结构，含 x, timesteps, positional_embeddings, context, etc.)
- `action_args: ActionTransformerArgs` (我们定义，**形似 LTX TransformerArgs** 但 dim=1024)

伪代码（这就是 §4 要实现的 `LTXJointMoTBlock.forward`）：

```python
def forward(self, video_args, action_args, perturbations=None):
    v_blk = self.video_block
    a_blk = self.action_block

    # ===== Step 1: 每流 AdaLN 取 shift/scale/gate =====
    v_shift_msa, v_scale_msa, v_gate_msa = v_blk.get_ada_values(
        v_blk.scale_shift_table, video_args.x.shape[0], video_args.timesteps, slice(0, 3))
    a_shift_msa, a_scale_msa, a_gate_msa = a_blk.get_ada_values(
        a_blk.scale_shift_table, action_args.x.shape[0], action_args.timesteps, slice(0, 3))

    # ===== Step 2: 每流 normalize → Q/K/V projection → q_norm/k_norm =====
    v_norm = rms_norm(video_args.x) * (1 + v_scale_msa) + v_shift_msa
    a_norm = rms_norm(action_args.x) * (1 + a_scale_msa) + a_shift_msa

    v_q = v_blk.attn1.to_q(v_norm); v_k = v_blk.attn1.to_k(v_norm); v_v = v_blk.attn1.to_v(v_norm)
    a_q = a_blk.attn1.to_q(a_norm); a_k = a_blk.attn1.to_k(a_norm); a_v = a_blk.attn1.to_v(a_norm)

    v_q = v_blk.attn1.q_norm(v_q); v_k = v_blk.attn1.k_norm(v_k)
    a_q = a_blk.attn1.q_norm(a_q); a_k = a_blk.attn1.k_norm(a_k)

    # ===== Step 3: 每流单独应用 RoPE =====
    v_q = apply_rotary_emb(v_q, video_args.positional_embeddings, v_blk.attn1.rope_type)
    v_k = apply_rotary_emb(v_k, video_args.positional_embeddings, v_blk.attn1.rope_type)
    a_q = apply_rotary_emb(a_q, action_args.positional_embeddings, a_blk.attn1.rope_type)
    a_k = apply_rotary_emb(a_k, action_args.positional_embeddings, a_blk.attn1.rope_type)

    # ===== Step 4: cat across streams =====
    joint_q = torch.cat([v_q, a_q], dim=1)   # (B, T_v + T_a, 4096)
    joint_k = torch.cat([v_k, a_k], dim=1)
    joint_v = torch.cat([v_v, a_v], dim=1)

    # ===== Step 5: build joint mask =====
    joint_mask = build_joint_attention_mask(
        video_mask=video_args.self_attention_mask,
        action_mask=action_args.self_attention_mask,
        T_v=v_q.shape[1], T_a=a_q.shape[1],
        device=joint_q.device, dtype=joint_q.dtype,
    )

    # ===== Step 6: joint attention =====
    attention_function = v_blk.attn1.attention_function  # 与 video 端共用 (DEFAULT)
    joint_out = attention_function(joint_q, joint_k, joint_v, v_blk.attn1.heads, joint_mask)

    # ===== Step 7: split + 各自 per-head gate + to_out =====
    v_out = joint_out[:, :v_q.shape[1]]
    a_out = joint_out[:, v_q.shape[1]:]

    if v_blk.attn1.to_gate_logits is not None:
        v_gate = 2.0 * torch.sigmoid(v_blk.attn1.to_gate_logits(v_norm))
        v_out = v_out.view(B, T_v, H, D) * v_gate.unsqueeze(-1)
        v_out = v_out.view(B, T_v, H*D)
    v_out = v_blk.attn1.to_out(v_out)
    # 同理 action

    # ===== Step 8: 残差 + gate_msa =====
    video_args.x = video_args.x + v_out * v_gate_msa
    action_args.x = action_args.x + a_out * a_gate_msa

    # ===== Step 9: 每流独立做 text cross-attention =====
    video_args.x = video_args.x + v_blk._apply_text_cross_attention(
        video_args.x, video_args.context, v_blk.attn2, v_blk.scale_shift_table,
        getattr(v_blk, "prompt_scale_shift_table", None),
        video_args.timesteps, video_args.prompt_timestep, video_args.context_mask,
        cross_attention_adaln=v_blk.cross_attention_adaln,
    )
    action_args.x = action_args.x + a_blk._apply_text_cross_attention(
        action_args.x, action_args.context, a_blk.attn2, ...   # 同 video 协议
    )

    # ===== Step 10: 每流独立做 FF（AdaLN + gate） =====
    v_shift_mlp, v_scale_mlp, v_gate_mlp = v_blk.get_ada_values(..., slice(3, 6))
    v_norm2 = rms_norm(video_args.x) * (1 + v_scale_mlp) + v_shift_mlp
    video_args.x = video_args.x + v_blk.ff(v_norm2) * v_gate_mlp
    # 同理 action

    return video_args, action_args
```

**关键观察**：所有 LTX-side 调用（`apply_rotary_emb`, `Attention.q_norm/k_norm/to_*/to_gate_logits`, `_apply_text_cross_attention`, `get_ada_values`, `FeedForward`）**都直接复用 LTX 上游模块**，我们只写 orchestration。无新数学。

---

## 4. 详细任务清单（替换原 plan Task 8-12）

每个 task 完成后 commit。任务编号继续沿用，便于交叉引用。

### Task 8: 修改 `LTXVideoDiT` 暴露 block-level forward API（不改 LTXModel 上游）

**为什么需要**：现在 `LTXVideoDiT.forward(video_modality)` 是端到端整本 48 层 forward。MoT 要的是**逐层调度**，需要：
- 拆开 LTXModel：`patchify → adaln_single → for each block: ... → norm_out + proj_out`
- 暴露：`prepare_video_args(latent, context, t, ...)` → `TransformerArgs`
- 暴露：`postprocess_video(video_args)` → `(B, C_out, F, H, W)`

实现方式：**继承 + 拆方法**，不修改 ltx-core 源码：
```python
class LTXVideoDiT(nn.Module):
    def prepare_args(self, latent, sigma, context, context_mask):
        # 调 self._inner.video_args_preprocessor.prepare(video_modality, audio=None)
        ...
    
    def postprocess(self, video_args):
        # 复制 LTXModel._process_output
        ...
```

- [ ] Step 1: 新增 `LTXVideoDiT.prepare_args(latent_5d, sigma, context, context_mask) -> TransformerArgs`
- [ ] Step 2: 新增 `LTXVideoDiT.postprocess(video_args) -> Tensor`（应用 final scale_shift + norm_out + proj_out + unpatchify）
- [ ] Step 3: 端到端等价测试：用 LTXModel.forward 一遍 vs prepare→\[手动 for block in blocks: block.forward\]→postprocess，断言数值差 < 1e-4
- [ ] **Commit**: `feat(ltx): LTXVideoDiT prepare_args/postprocess for layer-level scheduling`

**验证**：单输入随机 latent + 随机 context，两条路径数值一致。

### Task 9: 设计并实现 `LTXAlignedActionBlock` + `LTXAlignedActionDiT`

**File**: `src/fastwam/models/ltx/action_dit.py`（重写）

- [ ] Step 1: 写 `LTXAlignedActionBlock`，结构与 BasicAVTransformerBlock 同构，但 video.dim=1024（用 ltx_core 的 `TransformerConfig` 实例 + `BasicAVTransformerBlock` 直接构造，audio=None）
  ```python
  from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
  
  class LTXAlignedActionBlock(BasicAVTransformerBlock):
      def __init__(self, hidden_dim=1024, ffn_dim=4096, ...):
          video_cfg = TransformerConfig(
              dim=hidden_dim, heads=32, d_head=128, context_dim=4096,
              apply_gated_attention=True, cross_attention_adaln=True,
          )
          super().__init__(idx=0, video=video_cfg, audio=None, rope_type=SPLIT, norm_eps=1e-6)
  ```
  → 完全复用 LTX 上游 block 实现，零代码重写。

- [ ] Step 2: 写 `LTXAlignedActionDiT`，结构同 LTXModel video-only 子树（patchify_proj 改 action_encoder，adaln_single 用 hidden=1024）
- [ ] Step 3: 删除现有 `LTXAlignedActionDiT.from_pretrained` 加载 wan-style backbone 的逻辑；改成支持 LTX-derived backbone
- [ ] **Commit**: `feat(ltx-action): LTXAlignedActionDiT block-isomorphic to LTX video block (hidden=1024, inner=4096)`

### Task 10: 写 ActionDiT backbone preprocess（取 LTX video block init action block）

**File**: `scripts/preprocess_action_dit_backbone.py`（重写）

LTX video block 维度 (hidden=inner=4096)；action block (hidden=1024, inner=4096)。哪些权重可以"alpha-scale 复用"，哪些必须重新初始化：

| Submodule | Video shape | Action shape | 复用策略 |
|---|---|---|---|
| `attn1.to_q.weight` | (4096, 4096) | (4096, 1024) | 取 video 的前 1024 列，按 sqrt(1024/4096) 缩放 |
| `attn1.to_k.weight` | (4096, 4096) | (4096, 1024) | 同上 |
| `attn1.to_v.weight` | (4096, 4096) | (4096, 1024) | 同上 |
| `attn1.to_out.0.weight` | (4096, 4096) | (1024, 4096) | 取 video 的前 1024 行 + 缩放 |
| `attn1.q_norm.weight` | (4096,) | (4096,) | 直接复用 |
| `attn1.k_norm.weight` | (4096,) | (4096,) | 直接复用 |
| `attn1.to_gate_logits.weight` | (32, 4096) | (32, 1024) | 取前 1024 列 + 缩放 |
| `attn2.to_q.weight` (cross-attn) | (4096, 4096) | (4096, 1024) | 同 attn1.to_q |
| `attn2.to_k.weight` | (4096, 4096) | (4096, 4096) | text dim 不变；直接复用 |
| `attn2.to_v.weight` | (4096, 4096) | (4096, 4096) | 直接复用 |
| `attn2.to_out.0.weight` | (4096, 4096) | (1024, 4096) | 取前 1024 行 + 缩放 |
| `ff.net.0.proj.weight` | (16384, 4096) | (4096, 1024) | FF dim ratio 不同；行列都裁切+缩放 |
| `ff.net.2.weight` | (4096, 16384) | (1024, 4096) | 同上 |
| `scale_shift_table` | (6, 4096) | (6, 1024) | 取前 1024 + 缩放 |
| `prompt_scale_shift_table` | (2, 4096) | (2, 1024) | 同上 |

这是 alpha-scale interpolation 的经典手法（Wan22 14B 已经验证过）。预处理脚本输出一个 `.pt` 文件作为 ActionDiT 初始化检查点。

- [ ] Step 1: 写 alpha-scale 矩阵裁切函数（复用 wan22 现有 `preprocess_action_dit_backbone.py` 大部分逻辑，仅改 key 名映射）
- [ ] Step 2: 跑预处理：
  ```bash
  python scripts/preprocess_action_dit_backbone.py \
      --ltx-ckpt checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors \
      --action-hidden-dim 1024 --action-ffn-dim 4096 --action-num-layers 48 \
      --output checkpoints/ActionDiT_linear_interp_LTX_alphascale_1024hdim.pt
  ```
- [ ] Step 3: 校验 meta：`num_layers=48, num_heads=32, attn_head_dim=128, text_dim=4096, hidden_dim=1024, ffn_dim=4096`
- [ ] **Commit**: `feat(preprocess): ActionDiT backbone alpha-scale init from LTX-2.3 video block`

### Task 11: 实现 `LTXJointMoTBlock` + `LTXMoT`

**File**: `src/fastwam/models/ltx/mot.py`（重写）

- [ ] Step 1: 写 `LTXJointMoTBlock.forward` 完整版（§3 伪代码的实际实现）
- [ ] Step 2: 写 `build_joint_attention_mask(video_mask, action_mask, T_v, T_a, device, dtype)` — 默认全 mixed，可选 action causal
- [ ] Step 3: 写 `LTXMoT` 顶层（管理 motblocks ModuleList[48]、按层调度、prefill 缓存机制）
- [ ] Step 4: 与 ablation 模式的隔离开关（虽然主线走 joint，但保留 `mot_mode: "joint" | "ablation"` 配置项以便对照实验）
- [ ] Step 5: 单层数值验证：
  - 构造 video latent (B=1, T_v=64, 4096) + action latent (B=1, T_a=33, 1024)
  - 调 `motblock.forward(video_args, action_args)`
  - 验证：输出形状 (1, 64, 4096) + (1, 33, 1024) 不 NaN
- [ ] Step 6: 多层叠加验证：48 层 stacked → 输出 stable，gradient 通到 input
- [ ] Step 7: ablation vs joint 对比：检查 action gradient w/r/t video input 在 joint mode > 0，ablation mode ≈ 0（cross-attn 通路除外）— 证明 joint self-attn 确实让两流交互
- [ ] **Commit**: `feat(ltx-mot): joint self-attention MoTBlock with LTX video + action experts (real MoT mechanism preserved)`

### Task 12: 改造 `FastWAM` 入口 + `runtime.py` + yaml

**Files**: `src/fastwam/models/ltx/fastwam.py`, `runtime.py`, `configs/model/fastwam*.yaml`

- [ ] Step 1: 改 fastwam.py `from_ltx_video_only_pretrained`：
  - 调用新 `load_ltx2_video_only_components` 签名（drop wan-specific kwargs）
  - 用 `LTXAlignedActionDiT.from_pretrained` 替代旧 ActionDiT
  - 用 `LTXJointMoTBlock` 配对
  - 文本走 dataset cache（`load_text_encoder=false`）
- [ ] Step 2: 重写 `_encode_video_latents` / `_decode_latents` 调用，对齐 LTXVideoVAE API（已对齐，应该无需大改）
- [ ] Step 3: 重写 `_encode_prompt` — 移除 Wan UMT5 tokenizer/encoder 逻辑；text 直接读 cache
- [ ] Step 4: 改 configs/model/fastwam.yaml 三件套：
  ```yaml
  _target_: fastwam.runtime.create_fastwam
  ckpt_path: checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors
  gemma_path: checkpoints/google/gemma-3-12b-it-qat-q4_0-unquantized
  load_text_encoder: false
  load_gemma: false  # 训练时不在内存
  mot_mode: "joint"  # 默认走 plan B 真 MoT
  action_dit_config:
    hidden_dim: 1024
    num_heads: 32       # LTX video 同步
    attn_head_dim: 128
    num_layers: 48      # LTX video 同步
    text_dim: 4096      # LTX cross-attn KV dim
    ffn_dim: 4096
    apply_gated_attention: true
    cross_attention_adaln: true
    rope_type: "split"
  video_scheduler:
    type: ltx2
    base_shift: 0.95
    max_shift: 2.05
  action_scheduler:
    train_shift: 5.0
    sigma_floor: 0.0
  ```
- [ ] Step 5: 删除/标 deprecated 旧 wan-specific yaml 字段（model_id, tokenizer_*, redirect_*）
- [ ] **Commit**: `feat(ltx-runtime): wire FastWAM entrypoint + yaml configs for LTX joint MoT`

### Task 13: 数据集文本 cache 预计算脚本

**File**: `scripts/precompute_text_cache_ltx.py`（新增）

- 加载 `LTXTextEncoder.from_config(cfg)` + `attach_gemma(gemma_path)`
- 对每个 prompt：`enc.encode([prompt], device="cuda", max_tokens=128) → (1, 128, 4096)`
- 缓存到 `data/text_embeds_cache/robotwin/{task}_seed{X}.ltx23_gemma3_12b_v2connector.pt`
- 沿用 14B cache 的目录结构和 slug 拼装规则，仅换 slug

- [ ] Step 1: 写脚本
- [ ] Step 2: 对 robotwin 10 task 子集跑一遍验证（~少量 prompts，<5min）
- [ ] Step 3: 对完整 50 task 跑（估 30min）
- [ ] **Commit**: `feat(data): LTX text cache precompute script (Gemma + V2 + Connector → 4096-d)`

### Task 14: 1-step 合成 smoke

**File**: `scripts/smoketests/test_ltx_joint_mot.py`

- [ ] Step 1: 实例化 `FastWAM.from_ltx_video_only_pretrained(...)` 完整
- [ ] Step 2: 喂随机 video latent + 随机 action latent + 随机 text emb (4096-d)
- [ ] Step 3: 一次 `model.training_loss(sample)` forward + backward
- [ ] Step 4: 验证：loss 非 NaN，gradient 流经 video & action backbone，no OOM 单卡 80GB
- [ ] **Commit**: `test: LTX joint MoT 1-step synthetic smoke`

### Task 15: 1-step FSDP 真数据 smoke

- [ ] Step 1: 用 robotwin task 跑 `bash scripts/train_fsdp.sh 7 task=robotwin_uncond_3cam_384_1e-4 max_steps=1 wandb.enabled=false`
- [ ] Step 2: 记录 step time（与 14B 15s/step 对比）
- [ ] Step 3: 验证：退出 0，loss 非 NaN，FSDP wrap MoTBlock 正确
- [ ] **Commit**: `test: LTX joint MoT 1-step FSDP smoke on RoboTwin`

### Task 16: README + arch notes

- [ ] Step 1: 写 FastWAM_LTX/README.md
- [ ] Step 2: docs/notes/2026-05-19-ltx-arch-notes.md
- [ ] Step 3: 更新 plan 末尾："Plan B 完成"小节
- [ ] **Commit**: `docs: FastWAM_LTX README + LTX joint MoT architecture notes`

---

## 5. 关键风险与缓解

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| 1 | LTX `Attention.forward` 内部把 RoPE + attention + gating 串成一个不可拆的 callable，**真 joint attention 要从外部拆 QKV** | **High** | §1.2 已确认所有需要的 sub-step（`to_q/to_k/to_v`, `q_norm/k_norm`, `apply_rotary_emb`, `attention_function`, `to_gate_logits`, `to_out`）都是 attn1 的公开属性，**不需要 monkey-patch** |
| 2 | RoPE shape 不一致：video 是 3D pe，action 是 1D pe — concat 前形状要匹配 | High | `apply_rotary_emb` 接受 `pe: (B, T, head_dim)` 一致 shape；两流分别旋转后再 cat **不要求 pe 一致** |
| 3 | `apply_gated_attention=True` 让 `to_gate_logits` 在 attention 之后但 to_out 之前作用 → 必须在 split 之后**每流单独应用** | High | §3 Step 7 已经按此设计 |
| 4 | Action 上 `prompt_scale_shift_table` 等 cross_attention_adaln=True 的额外参数 — 初始化策略不明 | Med | Task 10 alpha-scale 表已列入；初值用 video 同位置 + 缩放 |
| 5 | FSDP wrap 单位是 MoTBlock，但 MoTBlock 内部 `video_block` 是 LTX 上游模块（不是我们写的）— FSDP wrap policy 要识别它 | Med | 在 fsdp wrap policy 用 `transformer_layer_cls_to_wrap=[LTXJointMoTBlock]`（顶层，不下钻）即可 |
| 6 | Action causal mask × video first-frame-causal mask 在 joint 维度合并复杂 | Med | 第一版用 **全 mixed 无 mask**（最宽松），跑通后再叠加 mask |
| 7 | gradient checkpointing：LTX block 本身有 set_gradient_checkpointing，我们的 MoT 又有外层 wrap → 双重 ckpt 会 OOM | Med | sym Wan 14B 经验：disable LTX 内部 set_gradient_checkpointing，仅在 MoTBlock 外层 CheckpointWrapper 包一次 |
| 8 | LTX block forward 内有 `replace(video, x=vx)` 返回 dataclass —我们绕过这个调用直接操作 args.x | Low | OK，replace 只是 immutable dataclass 风格；我们用 mutable wrapper |
| 9 | 内存：joint attention Q=K=V 都是 (B, T_v+T_a, 4096) bfloat16 ≈ 5400 tokens × 4096 × 2 ≈ 44MB per tensor，× 48 layers no-ckpt ≈ 6GB activation/batch — 可控 | Low | 与 Wan22 14B 同量级 |
| 10 | LTX `attention_function` 是 module-level callable (PyTorch SDPA / xFormers / Flash3) — 我们在 joint mode 复用 video 端的 callable，与 action 端是否一致？ | Low | action block 也用 `AttentionFunction.DEFAULT` → 同一份 callable，安全 |
| 11 | Alpha-scale init 对 LTX video block 可能不工作（Wan 验证过，LTX 没验证过） | Med | Task 10 完成后看 Task 14 smoke 是否 loss 下降；如果 alpha-scale 不行，回退到随机初始化 ActionDiT |

---

## 6. 验证矩阵（每个里程碑必须通过）

| 里程碑 | 验证 | 期望结果 |
|---|---|---|
| Task 8 完成 | LTXVideoDiT prepare→blocks→postprocess vs LTXModel.forward | 数值差 < 1e-4 |
| Task 9 完成 | LTXAlignedActionBlock instantiation + random forward | 输出 (B,T,1024)，无 NaN |
| Task 10 完成 | preprocess.pt meta 与 LTX 一致 | num_layers=48, num_heads=32, attn_head_dim=128, text_dim=4096 |
| Task 11 完成 | 1-layer MoTBlock joint forward | 输出 shape (B, T_v, 4096) + (B, T_a, 1024) 一致，joint mode 下 action grad w/r/t video input 显著 > 0 |
| Task 11 ablation 对照 | 同上但 ablation mode | 输出 shape 一致；action grad w/r/t video input ≈ 0（仅 cross-attn 通路） |
| Task 14 合成 smoke | 单卡 80GB synthetic 1-forward+backward | loss 非 NaN, no OOM, < 30s |
| Task 15 FSDP smoke | 7 卡 1-step train | exit 0, loss 非 NaN, step time 记录（与 14B 15s 对比）|

---

## 7. 时间预算

| Task | 估时 | 主要工作 |
|---|---|---|
| 8 | 0.5 day | 拆 LTXVideoDiT prepare/postprocess + 等价测试 |
| 9 | 0.5 day | LTXAlignedActionBlock/DiT（结构同构，主要是 wiring） |
| 10 | 0.5 day | preprocess.py 重写 + 跑通 |
| 11 | 1.0 day | **核心实现** — joint MoT block + 数值验证 + ablation 对照 |
| 12 | 0.5 day | fastwam.py / runtime.py / yaml |
| 13 | 0.5 day | 文本 cache precompute + 跑 50 task |
| 14 | 0.25 day | 合成 smoke |
| 15 | 0.25 day | FSDP smoke |
| 16 | 0.25 day | 文档 |
| **合计** | **~4 天** | 单人专注 |

---

## 8. Open Questions（开工前确认）

1. **ActionDiT 维度**：保留 `hidden_dim=1024, ffn_dim=4096`？或者跟着 LTX 也用 hidden=4096？
   - 推荐**保留 1024**（FastWAM 历史一致，模型大小可控，alpha-scale 表已设计）
2. **Alpha-scale vs 随机初始化**：Task 10 优先做 alpha-scale；失败回退随机？
   - 推荐 alpha-scale 先做（Wan22 验证有效），失败再说
3. **Action causal mask**：第一版 joint attention **全 mixed 无 mask** 是否 OK？
   - 推荐 **OK**，先跑通；后续按需加 `action_group_causal_mask_mode`
4. **gradient checkpointing 策略**：禁用 LTX 内部 ckpt + 外层 CheckpointWrapper wrap MoTBlock？
   - 推荐 **是**（与 Wan22 14B 一致）
5. **mot_mode 默认值**：`"joint"` 还是 `"ablation"`？yaml 默认怎么设？
   - 推荐 **`"joint"`** 默认（用户已确认 Plan B）；ablation 仅作为对照实验保留
6. **第一版动作维度**：robotwin action_dim=7（默认）？还是按 robotwin_uncond_3cam_384_1e-4.yaml？
   - 跟现有 yaml 走，不动
7. **文本 cache 路径冲突**：写到 `data/text_embeds_cache/robotwin/*.ltx23_gemma3_12b_v2connector.pt` 不会和 14B 的 `*.wan22t2va14b.pt` 冲突 — slug 不同
   - 推荐**同目录、不同 slug**，共享 disk

---

## 9. 不在范围

- 推理路径（prefill_video_cache / forward_action_with_video_cache 双段）— 现有 MoT 已支持 prefill 机制，但需要单独适配 LTX block 的 K/V 形状。**Plan B 不做**，单独后续 follow-up
- 多分辨率 / 多帧数 — robotwin 固定 384×320×33
- LTX distilled / upscaler / IC-LoRA — 不用
- 评测路径（eval scripts）— 跑通训练后单独做
- 论文 ablation runs（joint vs no-joint 对照训练）— 仅留出对比 hook，不做完整训练对比

---

## 10. 与原 plan 的关系

本文档**取代** original plan §4 的 Task 9-12 描述（"MoT helper ablation 模式" 替换为本文档的 Task 8-11 真 joint 实现）。其余章节（§A 调研结果、§B.1-§B.4 决策、§B.6 submodule、§B.7 不在范围）仍然有效。

> 设计稿结束。等用户审阅后开工 Task 8。
