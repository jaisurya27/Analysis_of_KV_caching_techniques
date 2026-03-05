"""
Model loading and KV-compression method application for Qwen3.

Supported methods
-----------------
full          – Standard full-attention baseline (no compression).
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

# ---------------------------------------------------------------------------
# Model + Tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen3-8B",
    attn_implementation: str = "eager",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Qwen3 (or compatible) causal-LM and its tokenizer.

    Args:
        model_name:           HuggingFace model ID.
        attn_implementation:  One of "eager", "sdpa", "flash_attention_2".
                              H2O requires "eager" to obtain softmax weights.
        device_map:           Passed to from_pretrained; use "auto" for
                              multi-GPU or "mps" / "cpu" for Apple Silicon.
        torch_dtype:          Model weight dtype.
        load_in_4bit:         Enable BitsAndBytes 4-bit quantization to fit
                              large models on consumer GPUs.

    Returns:
        (model, tokenizer) tuple.
    """
    logger.info("Loading tokenizer for %s …", model_name)
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
            logger.warning(
                "bitsandbytes not installed; falling back to full precision."
            )

    logger.info("Loading model %s …", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Method application
# ---------------------------------------------------------------------------

def apply_kv_method(
    model: PreTrainedModel,
    method: str,
    *,
    hh_ratio: float = 0.1,
    recent_ratio: float = 0.1,
    sink_size: int = 4,
    max_seq_len: int = 4096,
    cache_ratio: float = 0.2,
) -> PreTrainedModel:
    """Patch *model* in-place so it uses the requested KV-compression method.

    Args:
        model:        The loaded causal-LM (must be in eval mode).
        method:       One of "full", "h2o", "streaming_llm".
        hh_ratio:     For H2O — fraction of context kept as heavy hitters.
        recent_ratio: For H2O — fraction of context always kept as recent.
        sink_size:    For StreamingLLM — number of sink tokens.
        max_seq_len:  Reference sequence length used to compute absolute sizes.
        cache_ratio:  Overall budget as a fraction of max_seq_len (used when
                      hh_ratio + recent_ratio == cache_ratio for consistency).

    Returns:
        The same model (possibly patched in-place).
    """
    method = method.lower()

    if method == "full":
        logger.info("Using full-attention baseline (no compression).")
        return model

    if method == "h2o":
        hh_size = max(1, int(max_seq_len * hh_ratio))
        recent_size = max(1, int(max_seq_len * recent_ratio))
        logger.info(
            "Applying H2O: hh_size=%d, recent_size=%d, budget=%d",
            hh_size,
            recent_size,
            hh_size + recent_size,
        )
        patch_model_for_h2o(model)
        # Store construction parameters on the model for create_cache() below.
        model._h2o_hh_size = hh_size
        model._h2o_recent_size = recent_size
        return model

    if method == "streaming_llm":
        total_budget = max(sink_size + 1, int(max_seq_len * cache_ratio))
        window_size = total_budget - sink_size
        logger.info(
            "Applying StreamingLLM: sink_size=%d, window_size=%d, budget=%d",
            sink_size,
            window_size,
            total_budget,
        )
        # No patching needed; the cache is constructed at inference time.
        model._streaming_sink_size = sink_size
        model._streaming_window_size = window_size
        return model

    raise ValueError(
        f"Unknown method '{method}'.  Choose from: full, h2o, streaming_llm."
    )


def create_cache(model: PreTrainedModel, method: str) -> Optional[object]:
    """Instantiate the appropriate KV cache object for the active method.

    Call this once per inference request; do NOT reuse across requests.

    Returns:
        A Cache instance for H2O / StreamingLLM, or None for full attention.
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
