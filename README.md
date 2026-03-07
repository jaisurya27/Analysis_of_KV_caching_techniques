# KV Cache Compression — Efficient AI Course Project

Systematic evaluation of **H2O** and **StreamingLLM** KV-cache compression
on the [LongBench](https://github.com/THUDM/LongBench) benchmark, using
**Mistral-7B-Instruct-v0.2** as the base model.

---

## Background

Standard self-attention has O(N²) complexity in the sequence length N.  The
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
├── scripts/
│   ├── run_full.sh           # Baseline (no compression)
│   ├── run_h2o.sh            # H2O
│   ├── run_streaming_llm.sh  # StreamingLLM
│   └── eval_all.sh           # Full sweep (all methods × 3 budgets)
├── run_eval.py               # Main evaluation entry point
├── run_benchmark.py          # Speed / memory benchmark
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

> **Flash Attention 2** (optional, CUDA only): `pip install flash-attn --no-build-isolation`
> **4-bit quantisation** (optional, CUDA): `pip install bitsandbytes`

### 2. Run a smoke-test (5 samples per task)

```bash
# Full-attention baseline
NUM_SAMPLES=5 bash scripts/run_full.sh

# H2O at 20% budget (10% HH + 10% recent)
NUM_SAMPLES=5 bash scripts/run_h2o.sh

# StreamingLLM at 20% budget
NUM_SAMPLES=5 bash scripts/run_streaming_llm.sh
```

### 3. Full LongBench sweep

```bash
bash scripts/eval_all.sh          # all methods × 10 / 20 / 50 % budgets
```

Results are written as JSON to `results/`.

### 4. Speed benchmark

```bash
python run_benchmark.py --method full h2o streaming_llm \
    --seq_len 512 1024 2048 4096 --gen_len 128
```

---

## Configuration

All scripts honour environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model ID |
| `MAX_SEQ_LEN` | `4096` | Context window (tokens) |
| `NUM_SAMPLES` | _(all)_ | Max samples per task |
| `OUTPUT_DIR` | `results` | Output directory |
| `HH_RATIO` | `0.10` | H2O: heavy-hitter fraction |
| `RECENT_RATIO` | `0.10` | H2O: recent-token fraction |
| `CACHE_RATIO` | `0.20` | StreamingLLM: total budget fraction |
| `SINK_SIZE` | `4` | StreamingLLM: number of sink tokens |

You can also call `run_eval.py` directly with full argument control:

```bash
python run_eval.py \
    --method h2o \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --hh_ratio 0.05 --recent_ratio 0.05 \
    --max_seq_len 4096 \
    --tasks narrativeqa hotpotqa gov_report \
    --num_samples 20 \
    --output_dir results/h2o_10pct
```

---

## LongBench tasks

| Task | Domain | Metric |
|---|---|---|
| `narrativeqa` | Single-doc QA | F1 |
| `qasper` | Scientific QA | F1 |
| `hotpotqa` | Multi-hop QA | F1 |
| `2wikimqa` | Multi-hop QA | F1 |
| `gov_report` | Summarisation | ROUGE-L |
| `trec` | Few-shot classification | Accuracy |
| `passage_count` | Synthetic counting | Accuracy |

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

### StreamingLLM (`src/kv_cache/streaming_llm.py`)

- Thin wrapper around HuggingFace's built-in `SinkCache`.
- No model patching needed — just pass the cache object to `model.generate()`.
- Works with any attention backend (eager, SDPA, Flash Attention 2).

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

4. Qwen Team, *Mistral Technical Report*, 2025.
   [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
