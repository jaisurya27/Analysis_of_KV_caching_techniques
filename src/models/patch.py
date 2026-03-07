"""
Model loading and KV-compression patching for Qwen2.5-1.5B-Instruct.

Approach is directly inspired by KVCache-Factory (MIT License):
  https://github.com/Zefan-Cai/KVCache-Factory

KVCache-Factory's monkeypatch.py replaces each attention module's `forward`
at the class level.  We do the same per-instance (so multiple models with
different methods can coexist), storing the constructed budget parameters as
attributes on the model object.

Supported methods
─────────────────
full          – Full-attention baseline, no compression.
h2o           – Heavy-Hitter Oracle (NeurIPS 2023).
streaming_llm – Attention Sink + sliding window (ICLR 2024).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.kv_cache.h2o import H2OCache, patch_model_for_h2o
from src.kv_cache.streaming_llm import StreamingLLMCache

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Model + Tokenizer loading
# ────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    attn_implementation: str = "eager",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Qwen2.5 (or compatible Llama-family) causal-LM and tokenizer.

    Args:
        model_name:           HuggingFace model ID.
        attn_implementation:  "eager" | "sdpa" | "flash_attention_2".
                              H2O **requires "eager"** to capture softmax weights.
                              StreamingLLM and full work with any backend.
        device_map:           "auto" for multi-GPU, "mps" for Apple Silicon,
                              "cpu" for CPU-only.
        torch_dtype:          Weight dtype; bfloat16 recommended.
        load_in_4bit:         BitsAndBytes 4-bit quantisation (CUDA only).

    Returns:
        (model, tokenizer) — model is in eval mode.
    """
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs.pop("torch_dtype", None)
        except ImportError:
            logger.warning("bitsandbytes not installed; using full precision.")

    logger.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


# ────────────────────────────────────────────────────────────────────────────
# Method application  (mirrors KVCache-Factory's replace_mistral + per-layer
# config approach, but stored on the model instance rather than globally)
# ────────────────────────────────────────────────────────────────────────────

def apply_kv_method(
    model: PreTrainedModel,
    method: str,
    *,
    hh_ratio: float = 0.10,
    recent_ratio: float = 0.10,
    sink_size: int = 4,
    max_seq_len: int = 4096,
    cache_ratio: float = 0.20,
) -> PreTrainedModel:
    """Patch *model* in-place for the requested KV-compression method.

    Analogous to KVCache-Factory's `replace_mistral(method)` + per-layer
    config loop.  Parameters are stored on the model so `create_cache()` can
    reconstruct the correct cache object at inference time.

    Args:
        model:        Loaded causal-LM in eval mode.
        method:       "full" | "h2o" | "streaming_llm".
        hh_ratio:     H2O heavy-hitter budget as fraction of max_seq_len.
        recent_ratio: H2O recency window as fraction of max_seq_len.
        sink_size:    StreamingLLM number of sink tokens.
        max_seq_len:  Context length used to compute absolute token counts.
        cache_ratio:  StreamingLLM total budget as fraction of max_seq_len.
    """
    method = method.lower()

    if method == "full":
        logger.info("Method: full attention (no compression)")
        return model

    if method == "h2o":
        hh_size     = max(1, int(max_seq_len * hh_ratio))
        recent_size = max(1, int(max_seq_len * recent_ratio))
        logger.info(
            "Method: H2O  |  hh=%d  recent=%d  budget=%d  (of %d)",
            hh_size, recent_size, hh_size + recent_size, max_seq_len,
        )
        patch_model_for_h2o(model)          # monkey-patch each attention layer
        model._h2o_hh_size     = hh_size
        model._h2o_recent_size = recent_size
        return model

    if method == "streaming_llm":
        total_budget = max(sink_size + 1, int(max_seq_len * cache_ratio))
        window_size  = total_budget - sink_size
        logger.info(
            "Method: StreamingLLM  |  sink=%d  window=%d  budget=%d  (of %d)",
            sink_size, window_size, total_budget, max_seq_len,
        )
        # No model patching needed — SinkCache handles eviction internally.
        model._streaming_sink_size   = sink_size
        model._streaming_window_size = window_size
        return model

    raise ValueError(
        f"Unknown method '{method}'. Choose from: full, h2o, streaming_llm."
    )


def create_cache(model: PreTrainedModel, method: str) -> Optional[object]:
    """Create a fresh cache object for one inference request.

    IMPORTANT: never reuse the same cache object across requests.

    Returns:
        H2OCache, StreamingLLMCache, or None (for full attention).
    """
    method = method.lower()

    if method == "full":
        return None

    if method == "h2o":
        return H2OCache(
            hh_size=model._h2o_hh_size,
            recent_size=model._h2o_recent_size,
        )

    if method == "streaming_llm":
        return StreamingLLMCache(
            sink_size=model._streaming_sink_size,
            window_size=model._streaming_window_size,
        )

    raise ValueError(f"Unknown method '{method}'.")
