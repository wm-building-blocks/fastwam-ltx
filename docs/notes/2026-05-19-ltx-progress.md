# FastWAM_LTX — Progress 2026-05-19

## Done (Tasks 1-7)

All adapter pieces built and validated end-to-end against real LTX-2.3 weights
(`checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors`, 43 GB) and
Gemma-3-12B-it (`checkpoints/google/gemma-3-12b-it-qat-q4_0-unquantized`, 23 GB).

| # | Task | Status | Verification |
|---|---|---|---|
| 1 | Install ltx-core in `fastwam_ltx` env (cloned from `fastwam`, transformers 5.8.1) | ✅ | All imports pass |
| 2 | Download LTX-2.3 + Gemma | ✅ | ~67 GB on disk |
| 3 | Scaffold `src/fastwam/models/ltx/` from wan22 + mechanical rename | ✅ | `import fastwam.models.ltx.*` syntax-ok |
| 4 | `LTXVideoDiT` adapter + audio key filter + safetensors metadata loader | ✅ | 1457 inner keys; **0 missing / 0 unexpected** on real load |
| 5 | `LTXVideoVAE` adapter via ltx-core VideoEncoder/Decoder | ✅ | 86/86 keys each; smoke `(1,3,33,384,320) → (1,128,5,12,10) → (1,3,33,384,320)` |
| 6 | `LTXTextEncoder` (Gemma + V2 FeatureExtractor + 8L Connector) | ✅ | 133 keys; end-to-end prompt → `(1,128,4096)` |
| 7 | `load_ltx2_video_only_components` unified loader | ✅ | CPU bf16 load of DiT+VAE in 62 s, asserts strict 0/0 |

## Important findings vs original plan (corrections)

1. **VAE compression: spatial 32, temporal 8** (plan §A.4 incorrectly said 16/4).
   - 384×320×33 → latent (5, 12, 10), confirmed by docstring + smoke run.
2. **`apply_gated_attention=True` and `cross_attention_adaln=True`** for LTX-2.3-22b-dev
   (non-default flags in `LTXModel.__init__`).
3. **DiT cross-attention dim = 4096** (Gemma 3840 is the *input* to `FeatureExtractorV2`,
   which projects to 4096 before reaching the DiT). ActionDiT `text_dim` stays 4096.
4. **VideoOnly cleanly omits audio submodules** at construction (`is_audio_enabled()`
   guards in `LTXModel.__init__`; `if audio is not None` guards in
   `BasicAVTransformerBlock.__init__`). No monkey-patch required.
5. **Checkpoint contains 4 namespaces beyond DiT**:
   - `model.diffusion_model.*` (DiT, 4444 keys total, of which 2729 audio + 258
     audio_embeddings_connector + 1457 video DiT)
   - `vae.*` (170 keys: encoder + decoder + per_channel_statistics)
   - `audio_vae.*` (102 keys, discarded)
   - `vocoder.*` (1227 keys, discarded — audio only)
   - `text_embedding_projection.*` (4 keys: video_aggregate_embed.{w,b}, audio version)
6. **No standalone `config.json`** on the HF repo; the config is embedded in
   `__metadata__["config"]` JSON inside the safetensors file.

## Repo state

- `/home/admin/fang/FastWAM_LTX/` — git initialized, 7 commits (one per Task).
- `third_party/ltx-2/` — vendored LTX-2 master (commit `1799988`), in `.gitignore`
  for now; convert to submodule when stabilising.
- conda env: `fastwam_ltx` (cloned from `fastwam`, isolated, transformers 5.8.1
  for Gemma3 support — does **not** pollute the original `fastwam` env).

## Remaining (Tasks 8-13) — needs design check before continuing

| # | Task | Why it needs a pause |
|---|---|---|
| 8 | `runtime.py` + `configs/model/fastwam{,_idm,_joint}.yaml` + `fastwam.py` call-site rewrite | New loader signature drops Wan-specific kwargs (`model_id`, `tokenizer_model_id`, `tokenizer_max_len`, `redirect_common_files`); `LTXLoadedComponents.tokenizer` removed (text encoder owns it). FastWAM `_encode_prompt` / `_encode_video_latents` paths assume Wan APIs — needs decision on how to bridge. |
| 9 | **MoT helper for LTX block structure** | Plan §7 risk #4 (high). LTX block uses AdaLN-Zero (`scale_shift_table` + `adaln_single`) not Wan's 6-shift `modulation`. **First version = ablation mode** (independent self-attn) per plan §B.5 — needs confirmation before starting. |
| 10 | ActionDiT 48-layer preprocess | Depends on Task 9. |
| 11 | Synthetic 1-forward smoke | Depends on Tasks 9-10. |
| 12 | FSDP 1-step real-dataset smoke | Needs: dataset text cache precompute with `LTXTextEncoder`, FSDP wrap policy for `MoTBlock` containing LTX `BasicAVTransformerBlock`, 7 GPUs. |
| 13 | README + arch notes | After 8-12 land. |

## Decision asks before continuing

1. **MoT mode**: confirm first version uses ablation (no joint self-attn between
   video and action tokens; action block uses cross-attn against video latent).
   This is plan §B.5 default but worth confirming.
2. **Text cache schema**: pre-compute `LTXTextEncoder.encode(prompt) → (T, 4096)`
   tensors keyed by prompt hash + `ltx23_gemma3_12b_v2connector` slug — same disk
   layout as `wan22t2va14b` but with new slug. OK to overwrite existing
   `data/text_embeds_cache/robotwin/` location for the LTX cache, or want a
   separate `text_embeds_cache_ltx/` tree?
3. **GPU availability**: Tasks 11/12 want a 7×80GB box. Plan §3.8 anticipates
   17 GB/rank sharded — confirm that's still the layout to target (or hold off on
   FSDP smoke if GPUs are busy with robustness eval).
4. **Plan §A.4 correction**: VAE stride is 32/8 not 16/4 — update Phase 1 notes
   in the plan? (just docs)

## Commits

```
3675e0a feat(ltx): single-file load_ltx2_video_only_components
2efae57 feat(ltx): LTXTextEncoder wrapping Gemma + V2 FeatureExtractor + 8L Connector
744a4ec feat(ltx): LTXVideoVAE adapter wrapping ltx-core VideoEncoder+VideoDecoder
8fc70d1 feat(ltx): LTXVideoDiT adapter wrapping LTXModel(VideoOnly) + audio key filter
04b4108 feat(ltx): scaffold src/fastwam/models/ltx by copying wan22 + mechanical rename
69bedc1 docs(plan): import LTX-swap plan with Phase 1 research findings
4f46c9d init: fork from FastWAM_14B (code+configs+docs only, no data/runs/ckpts)
```
