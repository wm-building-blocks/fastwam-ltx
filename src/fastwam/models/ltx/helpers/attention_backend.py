"""SDPA backend selection for LTX attention.

Ported from LTX-2's ``CudnnAttention`` optimization. The stock LTX path calls
``F.scaled_dot_product_attention`` and lets PyTorch's dispatcher pick a kernel.
On Hopper GPUs (H100/H200) the cuDNN attention backend is typically ~5-15%
faster than the flash/mem-efficient kernels, but the dispatcher does not always
prefer it. This module lets us pin the backend via an env var.

Usage:
    from .helpers.attention_backend import sdpa_backend_ctx
    with sdpa_backend_ctx():
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

The backend is read once at import time from ``FASTWAM_ATTENTION_BACKEND``:

  - ``cudnn``   : prefer cuDNN attention, flash as fallback (Hopper-tuned).
  - ``flash``   : flash attention only.
  - ``efficient`` / ``xformers`` : memory-efficient kernel.
  - ``math``    : pure-math reference kernel (slow; debugging only).
  - ``auto`` / unset : PyTorch's default dispatcher (legacy behavior).

Resolving the choice once at module-init keeps the context cheap and lets
``torch.compile`` trace through the attention call without graph breaks.
"""

import os
from contextlib import nullcontext

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_BACKEND = os.environ.get("FASTWAM_ATTENTION_BACKEND", "auto").strip().lower()


def _resolve_backends():
    """Return the eager-mode SDPA backend list, or None for auto.

    Used only outside torch.compile. Inside a compiled region we deliberately
    do NOT pin a backend (see ``sdpa_backend_ctx``): inductor has its own SDPA
    lowering and picks a kernel whose meta kernel it models correctly. Pinning
    via ``sdpa_kernel`` inside compile is fragile — cuDNN trips an
    ``assert_size_stride`` (transposed output layout), and flash-only aborts
    because flash attention does not support the ``attn_mask`` the MoT joint
    attention always passes.
    """
    if _BACKEND in ("", "auto"):
        return None
    try:
        from torch.nn.attention import SDPBackend
    except ImportError:
        logger.warning(
            "FASTWAM_ATTENTION_BACKEND=%s requested but torch.nn.attention is "
            "unavailable; falling back to the default SDPA dispatcher.",
            _BACKEND,
        )
        return None

    mapping = {
        # cuDNN first, flash as fallback when cuDNN rejects the shape/mask.
        "cudnn": [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "xformers": [SDPBackend.EFFICIENT_ATTENTION],
        "math": [SDPBackend.MATH],
    }
    eager = mapping.get(_BACKEND)
    if eager is None:
        logger.warning(
            "Unknown FASTWAM_ATTENTION_BACKEND=%r; expected one of "
            "cudnn/flash/efficient/xformers/math/auto. Using default dispatcher.",
            _BACKEND,
        )
        return None

    logger.info("LTX attention backend: %s (FASTWAM_ATTENTION_BACKEND)", _BACKEND)
    return eager


_EAGER_BACKENDS = _resolve_backends()


def sdpa_backend_ctx():
    """Context manager selecting the configured SDPA backend.

    Returns a ``nullcontext`` when the backend is ``auto``/unset, or when
    running under torch.compile (let inductor's SDPA lowering pick the kernel),
    so callers can unconditionally wrap their ``scaled_dot_product_attention``
    call.
    """
    if _EAGER_BACKENDS is None:
        return nullcontext()
    try:
        import torch

        if torch.compiler.is_compiling():
            return nullcontext()
    except Exception:
        pass
    from torch.nn.attention import sdpa_kernel

    return sdpa_kernel(_EAGER_BACKENDS)
