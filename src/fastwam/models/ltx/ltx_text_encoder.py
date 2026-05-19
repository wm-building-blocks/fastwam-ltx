"""
LTX-2 text encoder adapter (Task 6 / 2026-05-19).

Bridges Gemma-3-12B's all-layer hidden states to the 4096-d cross-attention
context that LTXVideoDiT expects, by wrapping ltx-core's EmbeddingsProcessor
(FeatureExtractorV2 + Embeddings1DConnector + 128 learnable register tokens).

This file replaces the previous Wan/UMT5-based encoder. The original Wan
implementation (previously living in ltx_text_encoder.py) is intentionally
not preserved — it does not apply to the LTX backbone.

Weight provenance (LTX-2.3-22b-dev safetensors → EmbeddingsProcessor):
    text_embedding_projection.video_aggregate_embed.{w,b}
        → feature_extractor.video_aggregate_embed.{w,b}
    model.diffusion_model.video_embeddings_connector.X
        → video_connector.X

Audio side (audio_aggregate_embed, audio_embeddings_connector) is intentionally
dropped: VideoOnly mode never runs audio through the processor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import safetensors
    from ltx_core.text_encoders.gemma import (
        EmbeddingsProcessorConfigurator,
    )
    from ltx_core.text_encoders.gemma.embeddings_processor import (
        EmbeddingsProcessor,
        EmbeddingsProcessorOutput,
    )
    _LTX_TXT_OK = True
    _LTX_TXT_ERR = None
except Exception as _e:  # noqa: BLE001
    _LTX_TXT_OK = False
    _LTX_TXT_ERR = _e


# Constants from LTX-2.3 metadata (see Phase 1 research).
GEMMA_HIDDEN_SIZE = 3840
GEMMA_NUM_HIDDEN_LAYERS = 48
GEMMA_FEATURE_DIM = GEMMA_HIDDEN_SIZE * (GEMMA_NUM_HIDDEN_LAYERS + 1)  # 49 * 3840 = 188160
LTX_DIT_TEXT_DIM = 4096
LTX_CONNECTOR_NUM_REGISTERS = 128


def _build_embeddings_processor_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Build a state dict for ``EmbeddingsProcessor`` (video-only) by remapping
    the two key namespaces in the LTX-2.3 checkpoint.
    """
    sd: Dict[str, torch.Tensor] = {}
    audio_drop = 0
    fe_pref = "text_embedding_projection."
    con_pref = "model.diffusion_model.video_embeddings_connector."
    with safetensors.safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(fe_pref):
                # both video_aggregate_embed.* and audio_aggregate_embed.* → feature_extractor.*
                # We keep audio side here so the FeatureExtractorV2 module has all params
                # populated; the audio branch simply never runs in video-only forwards.
                sd["feature_extractor." + k[len(fe_pref):]] = f.get_tensor(k)
            elif k.startswith(con_pref):
                sd["video_connector." + k[len(con_pref):]] = f.get_tensor(k)
    return sd


class LTXTextEncoder(nn.Module):
    """LTX-2 text encoder adapter.

    Wraps the ``EmbeddingsProcessor`` (FeatureExtractorV2 + Embeddings1DConnector
    + 128 learnable registers). Two intended usage modes:

    1. **From pre-computed Gemma hidden states** (training, expected path):
       ``encoder.process_features(hidden_states, attention_mask)``
       where ``hidden_states`` is the tuple returned by Gemma's
       ``output_hidden_states=True`` forward (length 49 for Gemma-3-12B-it).

    2. **End-to-end** (Task 12 cache precompute will use this):
       ``encoder.encode(prompts, device)`` — runs Gemma internally. Requires
       Gemma to be loaded first via ``attach_gemma(...)``.

    Output shape: ``(B, T, 4096)`` plus a binary ``(B, T)`` attention mask.
    """

    def __init__(
        self,
        embeddings_processor: "EmbeddingsProcessor",
        *,
        dit_text_dim: int = LTX_DIT_TEXT_DIM,
    ) -> None:
        super().__init__()
        if not _LTX_TXT_OK:
            raise ImportError(
                "ltx-core text encoder imports failed; pip install -e "
                "third_party/ltx-2/packages/ltx-core. "
                f"Original error: {_LTX_TXT_ERR!r}"
            )
        self.embeddings_processor = embeddings_processor
        self.dit_text_dim = dit_text_dim
        self.gemma: Optional[nn.Module] = None
        self.tokenizer = None

    # ----- construction ----------------------------------------------------
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LTXTextEncoder":
        """Build the EmbeddingsProcessor from the LTX safetensors metadata
        config. We drop the audio connector so audio weights need not be
        loaded; this matches LTXModelType.VideoOnly on the DiT side."""
        ep = EmbeddingsProcessorConfigurator.from_config(config)
        # Drop audio side: video-only inference.
        ep.audio_connector = None
        # If feature_extractor has audio side, neuter the attribute so weight
        # loading does not produce phantom missing keys. We keep the module
        # itself (it's small) but never call it.
        return cls(embeddings_processor=ep)

    # ----- weight loading -------------------------------------------------
    def load_pretrained_state_dict(
        self,
        ltx_state_dict_remapped: Dict[str, torch.Tensor],
        strict: bool = False,
        drop_audio_unexpected: bool = True,
    ) -> Tuple[list, list]:
        """Load a *pre-remapped* state dict (keys already shaped for
        EmbeddingsProcessor). Use ``load_pretrained_safetensors`` for the
        common path that does the remapping for you."""
        result = self.embeddings_processor.load_state_dict(
            ltx_state_dict_remapped, strict=strict
        )
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        if drop_audio_unexpected:
            unexpected = [k for k in unexpected if not k.startswith("audio_")]
        return missing, unexpected

    def load_pretrained_safetensors(
        self,
        ltx_ckpt_path: str,
        strict: bool = False,
    ) -> Tuple[list, list]:
        sd = _build_embeddings_processor_state_dict(ltx_ckpt_path)
        return self.load_pretrained_state_dict(sd, strict=strict)

    # ----- Gemma plumbing (deferred to cache precompute time) -------------
    def attach_gemma(self, gemma_root: str, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        """Load Gemma-3-12B-it (and tokenizer) lazily. Used by the cache-precompute
        script; training reads pre-computed embeddings from disk and does *not*
        need Gemma in memory."""
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(gemma_root)
        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_root, torch_dtype=torch_dtype
        ).eval()
        self.gemma = gemma

    # ----- forward APIs ---------------------------------------------------
    def process_features(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> "EmbeddingsProcessorOutput":
        """Run FeatureExtractorV2 + video Connector on pre-computed Gemma
        hidden states. ``hidden_states`` must be a tuple of length 49 (one per
        Gemma transformer layer + the embedding output).

        We bypass ``embeddings_processor.process_hidden_states`` because the
        upstream pipeline returns both video and audio features unconditionally
        and then asserts ``audio_connector is not None``. In video-only mode we
        drop the audio features explicitly before the connector stage.
        """
        from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask

        ep = self.embeddings_processor
        if ep.feature_extractor is None:
            raise ValueError("feature_extractor is required for process_features()")
        video_feats, _audio_feats = ep.feature_extractor(
            hidden_states, attention_mask, padding_side
        )
        additive_mask = convert_to_additive_mask(attention_mask, video_feats.dtype)
        video_enc, _audio_enc, binary_mask = ep.create_embeddings(
            video_feats, None, additive_mask
        )
        return EmbeddingsProcessorOutput(video_enc, None, binary_mask)

    @torch.no_grad()
    def encode(
        self,
        prompts: List[str],
        device: torch.device,
        max_tokens: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """End-to-end text encode. Requires :py:meth:`attach_gemma` to have
        been called first. Returns ``(video_encoding, binary_attention_mask)``
        where the encoding shape is ``(B, T, 4096)``."""
        if self.gemma is None or self.tokenizer is None:
            raise RuntimeError(
                "Call attach_gemma(gemma_root) before encode(). Training paths "
                "should normally precompute embeddings to disk; this method "
                "exists mostly for the cache-precompute script."
            )
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
        ).to(device)
        out = self.gemma(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            output_hidden_states=True,
        )
        hidden = out.hidden_states  # tuple of 49 tensors, each (B, T, 3840)
        ep_out = self.process_features(
            hidden_states=hidden,
            attention_mask=tok["attention_mask"],
            padding_side=getattr(self.tokenizer, "padding_side", "left"),
        )
        return ep_out.video_encoding, ep_out.attention_mask


__all__ = [
    "LTXTextEncoder",
    "GEMMA_HIDDEN_SIZE",
    "GEMMA_FEATURE_DIM",
    "LTX_DIT_TEXT_DIM",
    "LTX_CONNECTOR_NUM_REGISTERS",
]
