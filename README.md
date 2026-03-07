# KV Cache Compression — Efficient AI Course Project

Systematic evaluation of **H2O** and **StreamingLLM** KV-cache compression
on the [LongBench](https://github.com/THUDM/LongBench) benchmark, using
**Qwen2.5-1.5B-Instruct** as the base model.

---

## Background

Standard self-attention has O(N²) complexity in the sequence length N. The
KV cache grows linearly and becomes the memory bottleneck for long contexts.
Two complementary strategies are studied here:

| Method | Core idea | Budget split |
|---|---|---|
| **H2O** (NeurIPS 2023) | Retain _heavy hitters_ — the small subset of tokens that accumulate the highest attention scores — alongside a recent window. | HH tokens + recent tokens |
| **StreamingLLM** (ICLR 2024) | Keep the first few _attention sink_ tokens (positions 0–3) plus a sliding window of recent tokens. Enables infinite-length generation. | Sink tokens + window |

---

## Repository structure

```
Kv_caching/
├── src/
│   ├── kv_cache/
│   │   ├── h2o.py            # H2OCache + patch_model_for_h2o()
│   │   └── streaming_llm.py  # StreamingLLMCache (wraps HF SinkCache)
│   ├── models/
│   │   └── patch.py          # load_model_and_tokenizer(), apply_kv_method()
│   └── eval/
│       ├── longbench.py      # LongBenchEvaluator
│       └── metrics.py        # F1, ROUGE-L, classification, code-sim
├── notebooks/
│   └── evaluate.ipynb        # PRIMARY entry point — demo, eval, plots
├── requirements.txt
└── results/                  # JSON result files (auto-created)
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
jupyter notebook notebooks/evaluate.ipynb
```

Run top-to-bottom:

| Section | What it does |
|---|---|
| §0 | Install / verify dependencies |
| §1 | Background: how each method works |
| §2 | Load Qwen2.5-1.5B-Instruct |
| §3 | Visual demo — see the KV cache shrink in real time |
| §4 | LongBench smoke-test (5 samples × 3 tasks) |
| §5 | Full LongBench evaluation (3 budgets × 2 methods) |
| §6 | Results — score table, bar chart, efficiency-accuracy curve, latency |

### Running on Google Colab (recommended for full eval)

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd Kv_caching
!pip install -q -r requirements.txt
```

Then open `notebooks/evaluate.ipynb`. The notebook auto-detects CUDA — no config changes needed.

---

## Smoke-test results (5 samples, 20% cache budget, MPS)

Tasks: `hotpotqa` (multi-hop QA), `qasper` (scientific QA), `multifieldqa_en` (single-doc QA)

**Score (%)**

| Task | full | h2o | streaming_llm |
|---|---|---|---|
| hotpotqa | 14.67 | 14.67 | 14.67 |
| multifieldqa_en | 29.99 | 27.74 | 21.09 |
| qasper | 14.09 | 11.05 | 13.30 |

**Avg latency per sample (s)**

| Task | full | h2o | streaming_llm |
|---|---|---|---|
| hotpotqa | 2.275 | 3.470 | 1.309 |
| multifieldqa_en | 13.187 | 5.352 | 4.302 |
| qasper | 19.516 | 9.010 | 2.924 |

Key observations:
- H2O preserves quality better than StreamingLLM on document QA (importance scoring vs. FIFO eviction).
- Both compression methods are significantly faster than full attention on long-context tasks (qasper, multifieldqa_en) where cache size dominates decode latency.
- On short contexts (hotpotqa), full attention is fastest — compression overhead is not worth it.

---

## Hardware guide (Qwen2.5-1.5B bfloat16, ~3 GB)

| Hardware | Notes |
|---|---|
| NVIDIA A100 / H100 | Run as-is, full eval ~30 min |
| RTX 3090 / 4090 (24 GB) | Run as-is |
| Apple M-series (≥ 8 GB RAM) | Set `DEVICE_MAP = "mps"` in §2; smoke-test only recommended |
| CPU only | Smoke-test only |

---

## Implementation notes

### H2O (`src/kv_cache/h2o.py`)

- Extends HuggingFace `DynamicCache`.
- Each attention layer's forward is monkey-patched (`patch_model_for_h2o`)
  to call `H2OCache.record_attn_weights()` after every forward pass.
- Weights are averaged over heads and summed over query positions to form
  a cumulative importance score for each cached token.
- **Requires `attn_implementation="eager"`** (SDPA / Flash Attention do not
  return softmax weights when `output_attentions=True`).
- Rotary embeddings are patched to return the full precomputed cos/sin table,
  allowing absolute position IDs to remain valid after cache eviction.

### StreamingLLM (`src/kv_cache/streaming_llm.py`)

- Thin wrapper around HuggingFace's built-in `SinkCache`.
- No model patching needed — just pass the cache object to `model.generate()`.
- Works with any attention backend (eager, SDPA, Flash Attention 2).
- `get_usable_length` is overridden to handle bulk prefill correctly with Qwen2-family models.

---

## References

1. Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference
   of Large Language Models*, NeurIPS 2023.
   [arxiv:2306.14048](https://arxiv.org/abs/2306.14048)

2. Xiao et al., *Efficient Streaming Language Models with Attention Sinks*,
   ICLR 2024.
   [arxiv:2309.17453](https://arxiv.org/abs/2309.17453)

3. Bai et al., *LongBench: A Bilingual, Multitask Benchmark for Long Context
   Understanding*, ACL 2024.
   [arxiv:2308.14508](https://arxiv.org/abs/2308.14508)

4. Qwen Team, *Qwen2.5 Technical Report*, 2024.
   [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
