# FastWAM_LTX — Progress 2026-05-19 (Plan B, post Task 11)

## Status: 高风险任务全部通过 — Plan B 核心机制已 smoke

| # | Task | Status | Verification |
|---|---|---|---|
| 1 | Install ltx-core in fastwam_ltx env (isolated from fastwam) | ✅ | imports OK |
| 2 | Download LTX-2.3 (43GB) + Gemma-3-12B (23GB) | ✅ | files in `checkpoints/` |
| 3 | Scaffold `src/fastwam/models/ltx/` (rename pass) | ✅ | imports OK |
| 4 | LTXVideoDiT adapter + audio key filter | ✅ | **0 missing / 0 unexpected** on 13B DiT weights |
| 5 | LTXVideoVAE adapter | ✅ | 86/86 keys per encoder/decoder; 33×384×320 ↔ (5,12,10) round-trip |
| 6 | LTXTextEncoder (Gemma + V2 FeatureExtractor + 8L Connector) | ✅ | 133/133 keys; prompt → (1,128,4096) on GPU bf16 |
| 7 | `load_ltx2_video_only_components` loader | ✅ | 62s CPU bf16 load |
| 8 | LTXVideoDiT layer-level API (`prepare`/`run_blocks`/`postprocess`/`build_video_to_video_mask`) | ✅ | **diff=0 bit-exact** vs LTXModel.forward |
| 9 (redo) | LTXAlignedActionDiT (LTX-isomorphic, hidden=1024, inner=4096) | ✅ | 3.24B params (matches Wan22 ActionDiT order); `pre_dit`/`post_dit` smoke |
| **11** | **Joint MoT with single LTX API + FastWAM mask** | **✅** | **Joint smoke: video (1,48,4096) + action (1,8,1024), no NaN, std ~0.3** |
| 10 | Alpha-scale init preprocess | ⏳ pending | (optional; ActionDiT can start from random init for first run) |
| 12 | fastwam.py + runtime + yaml wiring | ⏳ pending | (mechanical, needs `_encode_prompt` / dataset cache lookup changes) |
| 13 | Text cache precompute script | ⏳ pending | uses LTXTextEncoder.encode() in batch |
| 14 | Synthetic full-stack 1-step smoke | ⏳ pending | needs Task 12 + Task 11 |
| 15 | FSDP 1-step real-dataset smoke | ⏳ pending | needs Tasks 12, 13, 14 |
| 16 | README + arch notes | ⏳ pending | after everything lands |

## 重要决策（已 lock 进代码）

1. **保留 FastWAM 结构** — 两专家 MoT、FastWAM joint mask（video→action False、action→first_frame_video True）、ActionDiT hidden=1024 比例
2. **抛弃 Wan2.2 实现细节** — block 内部全部走 LTX BasicAVTransformerBlock，RoPE 全 LTX SPLIT，AdaLN-Zero scale_shift_table，apply_gated_attention=True
3. **Action expert cross_attention_adaln=False** — 因为 context dim (4096) ≠ action hidden (1024)，避免维度不兼容；这是 action 端唯一与 video 不对称的设计选择
4. **单 API MoT helpers** — `_build_expert_attention_io_ltx` + `_apply_expert_post_block_ltx` 各一份，video/action 共用，无 if-else 分支

## Task 11 Smoke 详情（scripts/smoketests/test_ltx_joint_mot.py）

| 项 | 值 |
|---|---|
| Video DiT | 2 层 random init, 0.76B params, hidden=4096 |
| Action DiT | 2 层 random init, 0.142B params, hidden=1024 |
| Input | B=1, video (3, 4, 4)=48 tokens, action 8 tokens, text (1, 32, 4096) |
| FastWAM Mask | `first_frame_causal` (video↔video), False (video→action), True (action↔action), True (action→first_frame_video). 1984/3136 attention slots active |
| Joint attention | Q/K/V cat at inner_dim=4096, single SDPA call, split back |
| Output | video (1, 48, 4096), action (1, 8, 1024) |
| 数值 | mean ≈ 0, std ≈ 0.3, **no NaN/Inf** |

## 剩余风险与已知未决

| # | 风险 | 严重度 | 处理 |
|---|---|---|---|
| 1 | Task 12 wiring 时 FastWAM 现有 `_encode_prompt` 用 Wan tokenizer/encoder — 要重写为 cache 读取 | Med | Task 12 必做 |
| 2 | Text cache slug 需统一 — robotwin 数据集里需要先用 LTXTextEncoder 重算 | Med | Task 13 |
| 3 | LTX 真实权重加载到 48 层 MoT 后显存预估：13B (video) + 3.2B (action, init random) ≈ 33GB bf16 + activations。7×80GB FSDP 应该够 | Low | Task 15 实测 |
| 4 | LTX 块的 `to_gate_logits` 在 random init 下激活分布很广（gates ∈ (0, 2)）；用预训练权重后会更稳 | Low | 暂不动 |
| 5 | Action token 用 LTX 3D rope (frame_idx, 0, 0) 的退化形式 — 数学上等价于 1D temporal rope，但训练时检查梯度是否合理 | Low | Task 14 看 |
| 6 | Alpha-scale init 没做 — ActionDiT 完全随机；可能收敛慢但不阻塞 | Low | Task 10 是 follow-up |

## 累计 commits

```
da0477d feat(ltx-task11): joint MoT with single LTX API + FastWAM mask preserved
7bf44c2 feat(ltx-task9-redo): replace Wan-style ActionDiT with LTX-isomorphic LTXAlignedActionDiT
8a9a589 feat(ltx-task9): ActionDiT confirmed LTX-compatible (dims contract documented)
2898cfb feat(ltx-task8): LTXVideoDiT layer-level API for MoT joint forward
cc23cae docs(notes): Tasks 1-7 progress + corrections vs Phase 1 plan
010b306 docs(plan-b): detailed design — preserve real MoT joint self-attention with LTX backbone
3675e0a feat(ltx): single-file load_ltx2_video_only_components
2efae57 feat(ltx): LTXTextEncoder wrapping Gemma + V2 FeatureExtractor + 8L Connector
744a4ec feat(ltx): LTXVideoVAE adapter wrapping ltx-core VideoEncoder+VideoDecoder
8fc70d1 feat(ltx): LTXVideoDiT adapter wrapping LTXModel(VideoOnly) + audio key filter
04b4108 feat(ltx): scaffold src/fastwam/models/ltx by copying wan22 + mechanical rename
69bedc1 docs(plan): import LTX-swap plan with Phase 1 research findings
4f46c9d init: fork from FastWAM_14B (code+configs+docs only, no data/runs/ckpts)
```

## 下一步建议

**首选**：直接跳到 Task 12（wire fastwam.py + yaml），然后 Task 13（text cache precompute），再 Task 14（合成 smoke）→ Task 15（FSDP smoke）。

**或**：先做 Task 10 alpha-scale 让 ActionDiT 起点更好，再走 Task 12-15。

**或**：先做 Task 16 文档 + 让用户对架构 review 一遍再继续大改。
