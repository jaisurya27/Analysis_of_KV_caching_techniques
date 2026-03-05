"""
StreamingLLM: Efficient Streaming Language Models with Attention Sinks
Paper: https://arxiv.org/abs/2309.17453
ICLR 2024

Key insight:
  Autoregressive LLMs assign disproportionately large attention weights to a
  few "sink" tokens at the very beginning of the sequence (typically positions
  0-3), regardless of their semantic content.  Keeping these sink tokens in the
  KV cache alongside a sliding window of the most-recent tokens allows stable
  generation over arbitrarily long sequences.

  Cache layout for each layer:
    [ sink_0, …, sink_{S-1} | oldest_recent, …, newest_recent ]
    |<------ sink_size ------>|<---------- window_size -------->|

  Total budget = sink_size + window_size.

  When the sequence grows beyond the budget:
    - The sink tokens are never evicted.
    - The oldest token in the recent window is dropped.

Implementation:
  HuggingFace Transformers (≥4.38) ships a built-in `SinkCache` that
  implements exactly this policy.  This module wraps it with a uniform API
  and provides a helper to convert a cache-budget fraction to concrete sizes.

Usage:
  >>> cache = StreamingLLMCache(sink_size=4, window_size=92)
  >>> outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)
"""

from __future__ import annotations

from typing import Optional

from transformers import SinkCache


class StreamingLLMCache(SinkCache):
    """Thin wrapper around HuggingFace SinkCache for a uniform project API.

    The total KV budget is:
        budget = sink_size + window_size

    Args:
        sink_size:   Number of initial "attention sink" tokens to always keep.
                     The original paper found 4 sinks to be sufficient.
        window_size: Number of most-recent tokens to keep in the sliding window.
    """

    def __init__(self, sink_size: int = 4, window_size: int = 92) -> None:
        super().__init__(window_length=window_size, num_sink_tokens=sink_size)
        self.sink_size = sink_size
        self.window_size = window_size
        self.budget = sink_size + window_size

    @classmethod
    def from_budget(
        cls,
        total_budget: int,
        sink_size: int = 4,
    ) -> "StreamingLLMCache":
        """Create a cache whose total capacity equals *total_budget* tokens.

        Args:
            total_budget: Total number of KV slots (sink + window).
            sink_size:    Tokens reserved for attention sinks (default 4).

        Returns:
            StreamingLLMCache instance.
        """
        window_size = max(1, total_budget - sink_size)
        return cls(sink_size=sink_size, window_size=window_size)

    @classmethod
    def from_ratio(
        cls,
        max_seq_len: int,
        ratio: float,
        sink_size: int = 4,
    ) -> "StreamingLLMCache":
        """Create a cache sized as a fraction of the full context length.

        Args:
            max_seq_len: Full context length of the model / prompt.
            ratio:       Fraction in (0, 1] to keep; e.g. 0.2 for 20 %.
            sink_size:   Tokens reserved for attention sinks (default 4).

        Returns:
            StreamingLLMCache instance.
        """
        total_budget = max(sink_size + 1, int(max_seq_len * ratio))
        return cls.from_budget(total_budget, sink_size=sink_size)

    def __repr__(self) -> str:
        return (
            f"StreamingLLMCache("
            f"sink_size={self.sink_size}, "
            f"window_size={self.window_size}, "
            f"budget={self.budget})"
        )
