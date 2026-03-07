"""
StreamingLLM: Efficient Streaming Language Models with Attention Sinks
Paper: https://arxiv.org/abs/2309.17453  (ICLR 2024)

Algorithmic approach adapted from KVCache-Factory:
  https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py

Key insight
───────────
LLMs assign surprisingly large attention weights to the very first token(s)
in a sequence ("attention sinks"), regardless of their semantic content.
Keeping those sink tokens in the cache alongside a sliding window of recent
tokens is sufficient to maintain generation quality — no importance scoring
or attention-weight tracking required.

Cache layout (each decoder layer):
  ┌──────────────┬──────────────────────────────┐
  │  sink tokens │  ←  recent window  →         │
  │  (positions  │  oldest_recent … newest_recent│
  │   0 … S-1)  │                              │
  └──────────────┴──────────────────────────────┘
       S tokens          W tokens
  Total budget B = S + W

Eviction rule: when a new token is added and len > B,
  drop the oldest token that is NOT a sink (i.e. position S).

Implementation
──────────────
HuggingFace Transformers 4.38-4.44 ships a built-in `SinkCache` that
implements exactly this layout.  `StreamingLLMCache` is a thin wrapper
that adds convenience constructors and a uniform project API.

Compatibility note
──────────────────
`SinkCache` exists in transformers 4.38 - 4.44.x.  It was removed in
transformers 5.0.  This project pins transformers==4.44.2 in requirements.txt.
"""

from __future__ import annotations

from transformers import SinkCache   # requires transformers 4.38 - 4.44.x


class StreamingLLMCache(SinkCache):
    """Thin wrapper around HuggingFace SinkCache.

    Adds `from_ratio()` / `from_budget()` convenience constructors so the
    rest of the codebase can specify cache budgets as fractions of the
    sequence length, consistent with the H2O interface.

    Args:
        sink_size:   Number of "attention sink" tokens pinned at the start.
                     The original paper found 4 tokens sufficient.
        window_size: Sliding window of recent tokens.
    """

    def __init__(self, sink_size: int = 4, window_size: int = 92) -> None:
        super().__init__(window_length=window_size, num_sink_tokens=sink_size)
        self.sink_size = sink_size
        self.window_size = window_size
        self.budget = sink_size + window_size

    # ------------------------------------------------------------------ #
    #  Convenience constructors                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_budget(
        cls,
        total_budget: int,
        sink_size: int = 4,
    ) -> "StreamingLLMCache":
        """Create a cache with exactly *total_budget* KV slots."""
        window_size = max(1, total_budget - sink_size)
        return cls(sink_size=sink_size, window_size=window_size)

    @classmethod
    def from_ratio(
        cls,
        max_seq_len: int,
        ratio: float,
        sink_size: int = 4,
    ) -> "StreamingLLMCache":
        """Create a cache sized as a fraction of *max_seq_len*.

        Args:
            max_seq_len: Full context length (tokens).
            ratio:       Fraction to keep, e.g. 0.20 for 20%.
            sink_size:   Tokens reserved as attention sinks.
        """
        total_budget = max(sink_size + 1, int(max_seq_len * ratio))
        return cls.from_budget(total_budget, sink_size=sink_size)

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Override SinkCache to avoid negative kv_seq_len during bulk prefill.

        SinkCache.get_usable_length returns (max_length - new_seq_length) when
        the cache would overflow.  During the initial prefill where new_seq_length
        is the full prompt length (e.g. 86) and the cache is empty (prev=0),
        this yields a negative number, making kv_seq_len = q_len + negative too
        small.  But SinkCache.update() on its first call stores all tokens
        without eviction, so the actual key count is q_len — the two are
        inconsistent and Qwen2Attention's size validation raises ValueError.

        Fix: return 0 when this layer's cache is empty (prefill, first call),
        so kv_seq_len == q_len, matching what update() will return.
        """
        if self.get_seq_length(layer_idx) == 0:
            return 0
        return super().get_usable_length(new_seq_length, layer_idx)

    def __repr__(self) -> str:
        return (
            f"StreamingLLMCache("
            f"sink={self.sink_size}, "
            f"window={self.window_size}, "
            f"budget={self.budget})"
        )
