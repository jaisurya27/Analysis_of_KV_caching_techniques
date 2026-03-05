"""
H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
Paper: https://arxiv.org/abs/2306.14048
NeurIPS 2023

Algorithm:
  During generation, a small subset of tokens ("heavy hitters") receives
  disproportionately large accumulated attention scores.  H2O evicts the
  lowest-scoring non-recent tokens whenever the KV cache exceeds a fixed budget
  B = hh_size + recent_size.

  Eviction policy per layer:
    1. Compute cumulative attention score for every cached token by summing
       (and averaging over heads) the softmax attention weights across all
       query positions seen so far.
    2. Split the budget into Heavy Hitters (HH) and Recent tokens.
       - Recent: always keep the last `recent_size` tokens unchanged.
       - HH:     from the remaining candidates keep the top-`hh_size` by score.
    3. Drop all other tokens from that layer's key/value cache.

Implementation notes:
  • Requires eager attention (output_attentions=True) to obtain softmax weights.
    Use  attn_implementation="eager"  when loading the model.
  • patch_model_for_h2o() monkey-patches every attention layer to forward
    attention weights to the H2OCache after each forward call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import Cache, DynamicCache


class H2OCache(DynamicCache):
    """DynamicCache extended with H2O eviction.

    Args:
        hh_size:     Number of heavy-hitter tokens to retain per layer.
        recent_size: Number of most-recent tokens to always retain per layer.
    """

    def __init__(self, hh_size: int, recent_size: int) -> None:
        super().__init__()
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.budget: int = hh_size + recent_size

        # Accumulated attention scores: list indexed by layer_idx.
        # Each element is a tensor of shape [batch, seq_len] or None.
        self._acc_scores: List[Optional[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Public interface used by the patched attention forward
    # ------------------------------------------------------------------

    def record_attn_weights(
        self,
        attn_weights: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Update accumulated attention scores and evict if over budget.

        Args:
            attn_weights: Softmax attention weights of shape
                          [batch, num_heads, q_len, k_len].  These cover the
                          full cached sequence (k_len == current cache size).
            layer_idx:    Index of the decoder layer.
        """
        # Average over heads, then sum over query positions → [batch, k_len]
        layer_scores = attn_weights.detach().float().mean(dim=1).sum(dim=1)

        # Grow the list if this layer hasn't been seen yet.
        while len(self._acc_scores) <= layer_idx:
            self._acc_scores.append(None)

        if self._acc_scores[layer_idx] is None:
            self._acc_scores[layer_idx] = layer_scores
        else:
            prev = self._acc_scores[layer_idx]
            k_len = layer_scores.size(-1)
            prev_len = prev.size(-1)
            if k_len > prev_len:
                # New tokens were appended; pad the accumulated scores.
                pad = torch.zeros(
                    prev.size(0),
                    k_len - prev_len,
                    device=prev.device,
                    dtype=prev.dtype,
                )
                prev = torch.cat([prev, pad], dim=-1)
                self._acc_scores[layer_idx] = prev
            # Add new per-token votes.
            self._acc_scores[layer_idx] = prev + layer_scores

        # Evict if the cache exceeds the budget.
        if layer_idx < len(self.key_cache):
            seq_len = self.key_cache[layer_idx].size(-2)
            if seq_len > self.budget:
                self._evict_layer(layer_idx)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_layer(self, layer_idx: int) -> None:
        """Keep top-hh_size heavy hitters + last recent_size tokens."""
        scores = self._acc_scores[layer_idx]  # [batch, seq_len]
        seq_len = self.key_cache[layer_idx].size(-2)

        # Indices for the "recent" window (always kept).
        recent_start = max(0, seq_len - self.recent_size)
        recent_idx = torch.arange(
            recent_start, seq_len, device=scores.device
        )

        # Candidate heavy-hitter pool: everything before the recent window.
        n_candidates = recent_start
        if n_candidates <= 0:
            return

        hh_scores = scores[:, :n_candidates]  # [batch, n_candidates]

        # Take the top-hh_size candidates (use min in case pool is small).
        k = min(self.hh_size, n_candidates)
        _, top_local = hh_scores.topk(k, dim=-1, sorted=False)

        # Use batch element 0 (evaluation runs with batch_size=1).
        top_global = top_local[0].sort().values  # keep temporal order

        # Merge with recent indices and deduplicate.
        keep = torch.cat([top_global, recent_idx]).unique(sorted=True)

        # Apply selection to key, value caches.
        self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep, :]

        # Prune accumulated scores to match.
        self._acc_scores[layer_idx] = self._acc_scores[layer_idx][:, keep]


# ---------------------------------------------------------------------------
# Model-level patching
# ---------------------------------------------------------------------------

def _make_h2o_forward(original_forward, layer_idx: int):
    """Return a patched attention forward that feeds weights to H2OCache."""

    def h2o_forward(*args, **kwargs):
        # Force output_attentions so we can capture the softmax weights.
        # We'll strip them from the output if the caller didn't request them.
        caller_wants_attn = kwargs.get("output_attentions", False)
        kwargs["output_attentions"] = True

        outputs = original_forward(*args, **kwargs)

        # outputs is a tuple whose structure depends on the attention class:
        #   (attn_output, attn_weights, past_key_value)  — eager impl.
        # attn_weights may be None for flash/sdpa even with output_attentions.
        attn_output = outputs[0]
        attn_weights = outputs[1] if len(outputs) > 1 else None

        # Feed weights to the H2OCache if it is the active cache.
        past_key_value = kwargs.get("past_key_value") or (
            args[3] if len(args) > 3 else None
        )
        if (
            isinstance(past_key_value, H2OCache)
            and attn_weights is not None
        ):
            past_key_value.record_attn_weights(attn_weights, layer_idx)

        if caller_wants_attn:
            return outputs
        # Drop attention weights from output tuple.
        return (attn_output,) + outputs[2:]

    return h2o_forward


def patch_model_for_h2o(model: Any) -> Any:
    """Monkey-patch every attention layer in *model* for H2O compatibility.

    This must be called **before** inference so that each attention forward
    passes its softmax weights to the H2OCache.

    Args:
        model: A HuggingFace causal-LM (e.g. Qwen3ForCausalLM).

    Returns:
        The same model object (patched in-place).
    """
    decoder_layers = model.model.layers  # works for Llama / Qwen3 topology

    for layer_idx, layer in enumerate(decoder_layers):
        attn = layer.self_attn
        original_fwd = attn.forward
        attn.forward = _make_h2o_forward(original_fwd, layer_idx)

    return model
