"""
H2O: Heavy-Hitter Oracle for Efficient Generative Inference
Paper: https://arxiv.org/abs/2306.14048  (NeurIPS 2023)

Algorithmic approach adapted from KVCache-Factory:
  https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py

How it works
────────────
Every token that has ever been in the context accumulates an "importance score"
equal to the sum of softmax attention weights it received across all query
positions so far.  A small number of these tokens become "heavy hitters" (H²)
— they keep attracting disproportionate attention.

KV eviction policy (per layer, per forward call):
  budget = hh_size + recent_size
  if cache_len > budget:
      keep = top-hh_size tokens by cumulative score   (heavy hitters)
           ∪ last recent_size tokens                  (recency window)
      drop everything else

Two-part implementation
───────────────────────
1. H2OKVCluster  — stateful object stored on each attention module.
                   Holds the compressed K/V tensors and the accumulated scores.
                   `update_kv(attn_weights, key_states, value_states)` is the
                   main entry point called from the patched forward.

2. patch_model_for_h2o(model, hh_size, recent_size)
                 — monkey-patches every MistralAttention (or LlamaAttention)
                   layer's `forward` so that it feeds softmax weights to the
                   H2OKVCluster after each forward call.

Requirement: attn_implementation="eager"
  SDPA and Flash-Attention do not return softmax weight tensors even when
  output_attentions=True, so we cannot observe which tokens are heavy hitters.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache


# ────────────────────────────────────────────────────────────────────────────
# H2OKVCluster — per-layer state  (mirrors KVCache-Factory's H2OKVCluster)
# ────────────────────────────────────────────────────────────────────────────

class H2OKVCluster:
    """Maintains a compressed K/V cache for a single attention layer.

    The cluster is initialised empty and grows token-by-token.  Once the
    cache exceeds `hh_size + recent_size` entries the eviction policy fires.

    Args:
        hh_size:     Maximum heavy-hitter slots to keep.
        recent_size: Number of most-recent tokens always kept.
    """

    def __init__(self, hh_size: int, recent_size: int) -> None:
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.budget = hh_size + recent_size

        # Running KV tensors — shape [batch, n_kv_heads, seq, head_dim]
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        # Accumulated attention scores — shape [batch, seq]
        self.acc_scores: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def update_kv(
        self,
        attn_weights: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V, update scores, and evict if over budget.

        Args:
            attn_weights:  Softmax weights [batch, heads, q_len, k_len].
            key_states:    New key tensor [batch, heads, q_len, head_dim].
            value_states:  New value tensor [batch, heads, q_len, head_dim].

        Returns:
            (key_cache, value_cache) after compression — to be used as the
            effective full K/V for the current attention computation.
        """
        # Accumulate scores: average over heads, sum over query positions.
        # Result: [batch, k_len]
        new_scores = attn_weights.detach().float().mean(dim=1).sum(dim=1)

        if self.k_cache is None:
            # First call — just store everything.
            self.k_cache = key_states
            self.v_cache = value_states
            self.acc_scores = new_scores
        else:
            # Append the new tokens to the cache.
            self.k_cache = torch.cat([self.k_cache, key_states[:, :, -1:, :]], dim=2)
            self.v_cache = torch.cat([self.v_cache, value_states[:, :, -1:, :]], dim=2)

            # Extend the score vector for the new position(s).
            pad_len = self.k_cache.shape[2] - self.acc_scores.shape[1]
            if pad_len > 0:
                self.acc_scores = F.pad(self.acc_scores, (0, pad_len))
            self.acc_scores = self.acc_scores + new_scores

            # Evict if we have grown past the budget.
            if self.k_cache.shape[2] > self.budget:
                self._evict()

        return self.k_cache, self.v_cache

    # ------------------------------------------------------------------ #
    #  Eviction                                                            #
    # ------------------------------------------------------------------ #

    def _evict(self) -> None:
        """Keep top-hh_size heavy hitters + last recent_size tokens."""
        seq_len = self.k_cache.shape[2]
        recent_start = max(0, seq_len - self.recent_size)

        # Score candidates are everything *before* the recent window.
        n_candidates = recent_start
        if n_candidates <= 0:
            return

        hh_scores = self.acc_scores[:, :n_candidates]   # [batch, n_cand]
        k = min(self.hh_size, n_candidates)
        _, top_local = hh_scores.topk(k, dim=-1, sorted=False)

        # Batch dim 0 (eval always uses batch_size=1).
        # top_idx ∈ [0, recent_start), recent_idx ∈ [recent_start, seq_len) —
        # non-overlapping, so .unique() is unnecessary and unsafe on MPS.
        top_idx = top_local[0].sort().values
        recent_idx = torch.arange(recent_start, seq_len,
                                  device=self.k_cache.device)
        keep = torch.cat([top_idx, recent_idx])

        self.k_cache = self.k_cache[:, :, keep, :]
        self.v_cache = self.v_cache[:, :, keep, :]
        self.acc_scores = self.acc_scores[:, keep]

    def reset(self) -> None:
        """Clear cached state (call between inference requests)."""
        self.k_cache = None
        self.v_cache = None
        self.acc_scores = None


# ────────────────────────────────────────────────────────────────────────────
# Model patching — injects H2OKVCluster into every attention forward
# ────────────────────────────────────────────────────────────────────────────

def _make_h2o_forward(original_forward, layer_idx: int):
    """Wrap an attention module's forward to feed weights to H2OKVCluster."""

    def h2o_forward(*args, **kwargs):
        # Always request attention weights; restore caller's preference later.
        caller_wants_attn = kwargs.get("output_attentions", False)
        kwargs["output_attentions"] = True

        outputs = original_forward(*args, **kwargs)

        # outputs ≡ (attn_output, attn_weights, past_key_value)  for eager.
        attn_output = outputs[0]
        attn_weights = outputs[1] if len(outputs) > 1 else None

        # Resolve the cache object (keyword arg or positional arg 3).
        # Use `is None` — not `or` — so an empty (falsy) H2OCache is not dropped.
        past_kv = kwargs.get("past_key_value")
        if past_kv is None:
            past_kv = args[3] if len(args) > 3 else None

        if isinstance(past_kv, H2OCache) and attn_weights is not None:
            past_kv.record_attn_weights(attn_weights, layer_idx)

        return outputs if caller_wants_attn else (attn_output, None) + outputs[2:]

    return h2o_forward


class H2OCache(DynamicCache):
    """DynamicCache extended with per-layer H2OKVCluster eviction.

    Stores one H2OKVCluster per decoder layer.  The cluster's `update_kv`
    is called *after* the standard cache update so that heavy-hitter scores
    can be tracked and the cache trimmed to budget.

    Note: the K/V tensors returned by `DynamicCache.update()` are the *full*
    (growing) cache.  After that call completes, `record_attn_weights` trims
    `self.key_cache[layer_idx]` and `self.value_cache[layer_idx]` in-place.
    The patched forward already captured `attn_weights` from the *un-trimmed*
    cache — that is correct behaviour: we score all tokens, *then* evict.
    """

    def __init__(self, hh_size: int, recent_size: int) -> None:
        super().__init__()
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.budget = hh_size + recent_size
        self._acc_scores: List[Optional[torch.Tensor]] = []

    def record_attn_weights(
        self, attn_weights: torch.Tensor, layer_idx: int
    ) -> None:
        """Update accumulated importance scores and evict if over budget."""
        # Average over heads, sum over query positions → [batch, k_len]
        new_scores = attn_weights.detach().float().mean(dim=1).sum(dim=1)

        while len(self._acc_scores) <= layer_idx:
            self._acc_scores.append(None)

        if self._acc_scores[layer_idx] is None:
            self._acc_scores[layer_idx] = new_scores
        else:
            prev = self._acc_scores[layer_idx]
            k_len = new_scores.size(-1)
            prev_len = prev.size(-1)
            if k_len > prev_len:
                prev = F.pad(prev, (0, k_len - prev_len))
            elif k_len < prev_len:
                prev = prev[:, :k_len]
            self._acc_scores[layer_idx] = prev + new_scores

        if layer_idx < len(self.key_cache):
            if self.key_cache[layer_idx].size(-2) > self.budget:
                self._evict_layer(layer_idx)

    def _evict_layer(self, layer_idx: int) -> None:
        seq_len = self.key_cache[layer_idx].size(-2)
        scores = self._acc_scores[layer_idx]           # [batch, seq]

        # Defensively clamp scores to actual cache length to prevent
        # index-out-of-bounds when sizes drift (e.g. on MPS with async ops).
        score_len = scores.size(-1)
        if score_len > seq_len:
            scores = scores[:, :seq_len]
            self._acc_scores[layer_idx] = scores
        elif score_len < seq_len:
            scores = F.pad(scores, (0, seq_len - score_len))
            self._acc_scores[layer_idx] = scores

        recent_start = max(0, seq_len - self.recent_size)
        if recent_start == 0:
            return

        # Compute keep indices on CPU to force MPS synchronisation.
        # On Apple Silicon, MPS queues ops asynchronously; topk can execute
        # on a stale buffer if we stay on-device.  Pulling scores to CPU
        # flushes the MPS queue and guarantees correct index values.
        # top_idx ∈ [0, recent_start), recent_idx ∈ [recent_start, seq_len) — non-overlapping.
        device = self.key_cache[layer_idx].device
        hh_scores_cpu = scores[:, :recent_start].cpu().float()
        k = min(self.hh_size, recent_start)
        _, top_local_cpu = hh_scores_cpu.topk(k, dim=-1, sorted=False)
        top_idx = top_local_cpu[0].sort().values          # CPU, values in [0, recent_start)
        recent_idx = torch.arange(recent_start, seq_len)  # CPU, values in [recent_start, seq_len)
        keep = torch.cat([top_idx, recent_idx]).to(device)

        self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep, :]
        self._acc_scores[layer_idx] = scores[:, keep]


def _make_rotary_patch(rotary_emb):
    """Return a patched rotary_emb.forward that always serves the full cos/sin table.

    Problem: Qwen2Attention computes
        kv_seq_len = q_len + cache.get_usable_length(...)   # = physical cache size + 1
        cos, sin   = rotary_emb(value_states, seq_len=kv_seq_len)
        ...
        apply_rotary_pos_emb(q, k, cos, sin, position_ids)  # does cos[position_ids]

    After H2O eviction the physical cache has `budget` tokens, so kv_seq_len ≈ 44.
    But position_ids encodes the ABSOLUTE position (e.g. 86 for the 87th token).
    cos_cached[:44][86] → index out of bounds.

    Fix: return the full precomputed cos_cached / sin_cached table (up to
    max_position_embeddings, e.g. 4096 entries) so every absolute position is
    reachable.  The attention-weight shape check uses kv_seq_len (unchanged),
    so that validation still passes.
    """
    def patched_forward(x, seq_len=None):
        if seq_len is not None and seq_len > rotary_emb.max_seq_len_cached:
            rotary_emb._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype
            )
        return (
            rotary_emb.cos_cached.to(dtype=x.dtype),
            rotary_emb.sin_cached.to(dtype=x.dtype),
        )
    return patched_forward


def patch_model_for_h2o(model: Any) -> Any:
    """Monkey-patch every attention layer in *model* for H2O.

    Works for Mistral and Llama family models (both use `model.model.layers`
    with a `.self_attn` sub-module).

    Must be called **once** before inference.  The model object is patched
    in-place and the same object is returned.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if getattr(attn, "_h2o_patched", False):
            continue  # already wrapped — skip to keep patches idempotent
        attn.forward = _make_h2o_forward(attn.forward, layer_idx)
        attn.rotary_emb.forward = _make_rotary_patch(attn.rotary_emb)
        attn._h2o_patched = True
    return model
