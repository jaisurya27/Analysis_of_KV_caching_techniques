# CLAUDE.md — Project Instructions & Memory

## Project Overview

**Goal:** Efficient AI course project evaluating H2O and StreamingLLM KV cache compression
on LongBench using Qwen/Qwen2.5-1.5B-Instruct.

**Primary workflow:** Run everything from `notebooks/evaluate_qwen2.5_1.5B.ipynb`.
Do not suggest running scripts — the notebook is the only entry point.

---

## Repository Structure

```
src/
  kv_cache/
    h2o.py              # H2OKVCluster, H2OCache (extends DynamicCache), patch_model_for_h2o()
    streaming_llm.py    # StreamingLLMCache (wraps HF SinkCache) with get_usable_length fix
  models/
    patch.py            # load_model_and_tokenizer(), apply_kv_method(), create_cache()
  eval/
    longbench.py        # LongBenchEvaluator — downloads data, runs eval, saves JSON
    metrics.py          # Official LongBench metrics: qa_f1_score, rouge_l_score, classification_score
notebooks/
  evaluate_qwen2.5_1.5B.ipynb   # LOCAL version (MPS/Apple Silicon)
  Qwen2_5_kvpress_LongBench_Colab.ipynb  # Colab version using kvpress library (separate approach)
  evaluate.ipynb                # Older notebook (ignore)
results/
  qwen2.5-1.5b/         # JSON result files per task+method, one file each
```

---

## Critical Implementation Details

### H2O
- **Must use `attn_implementation="eager"`** — SDPA/FlashAttention don't return softmax weights even with `output_attentions=True`.
- Monkey-patches each attention layer's forward to call `H2OCache.record_attn_weights()`.
- Rotary embeddings patched to return full cos/sin table (absolute positions stay valid after eviction).
- Budget split: `hh_ratio` (heavy hitters) + `recent_ratio` (sliding window).

### StreamingLLM
- Wraps HF `SinkCache` — no model patching needed.
- **`get_usable_length` override is required for Qwen2-family models**: on bulk prefill with empty cache, the default returns a negative number causing a shape mismatch. Fix: return 0 when layer cache is empty.
- Works with any attention backend (eager, SDPA, Flash2).

### create_cache / apply_kv_method
- `create_cache()` must be called **fresh per inference request** — never reuse.
- `create_cache` strips budget tag suffixes via `re.sub(r"_\d+pct$", "", method)` so `"h2o_10pct"` resolves to `"h2o"`. Using `split("_")[0]` would break `"streaming_llm"`.

### transformers version
- **Pinned to 4.44.2** — SinkCache was removed in 5.x; DynamicCache API changed.

---

## Model

- **Qwen/Qwen2.5-1.5B-Instruct** (~3 GB bfloat16) — default across all files.
- Config: 28 layers, 2 KV heads (GQA), 12 attention heads, hidden_size=1536, head_dim=128.
- Uses `model.model.layers[i].self_attn` — same as Mistral/Llama; H2O patching works unchanged.
- Avoid Phi models (use `model.layers` not `model.model.layers` — breaks H2O patching).
- Results stored under `results/qwen2.5-1.5b/` via `MODEL_SLUG = "qwen2.5-1.5b"`.

---

## Evaluation Setup

- **Tasks (6 English):** `narrativeqa`, `qasper`, `multifieldqa_en`, `gov_report`, `hotpotqa`, `2wikimqa`
- **Samples per task:** 50
- **Max input tokens:** 8192
- **Max new tokens:** per-task defaults (do NOT override globally)
- **Budgets tested:** 10%, 20%, 50%
- **Method run_ids:** `h2o_10pct`, `h2o_20pct`, `h2o_50pct`, `streaming_llm_10pct`, `streaming_llm_20pct`, `streaming_llm_50pct`, `full`

### Metrics
- QA tasks → token-level F1 (`qa_f1_score` from `metrics.py`) — follows official LongBench scoring.
- `gov_report` → ROUGE-L (LCS-based, implemented in `metrics.py`).
- Scores are stored as **0–100** in JSON files. Divide by 100 for display to match kvpress convention.
- `metrics.py` implements `_normalize_answer`: lowercase → remove punctuation → remove articles → collapse whitespace. This matches the official THUDM LongBench repo.

---

## Results Format

Each result file: `results/qwen2.5-1.5b/{task}_{method}.json`
```json
{
  "dataset": "hotpotqa",
  "method": "h2o_20pct",
  "num_samples": 50,
  "metric": "qa_f1",
  "score": 31.11,          // stored as 0-100
  "avg_latency_sec": 1.635,
  "predictions": [...],
  "individual_scores": [...]
}
```

### Loading results for display
Use `groupby().mean().unstack()` instead of `pivot()` — avoids duplicate index errors when smoke + full results both exist.
Divide `score` by 100 to display on 0–1 scale matching kvpress.

### Unified table (task | method | kv_budget | score_mean | latency_mean | kv_cache_mb)
Parse `kv_budget` from method name: `re.match(r"(h2o|streaming_llm)_(\d+)pct", method)`.
Compute `kv_cache_mb` analytically: `2 × n_layers × n_kv_heads × head_dim × budget_tokens × 2 bytes`.
`budget_tokens = int(MAX_SEQ_LEN × kv_budget)`.

---

## Memory Measurement

**What we measure:** Theoretical KV cache tensor size — NOT peak GPU memory.
Formula: `2 × n_layers × n_kv_heads × head_dim × budget_tokens × 2 bytes (bfloat16)`

**Why not `torch.cuda.max_memory_allocated()`:** Captures transient prefill activations (N×N attention matrices in eager mode) which dominate the peak and are identical across methods — shows only 2–4% difference. Theoretical formula correctly shows ~80% reduction at 20% budget.

**Correct presentation language:** "We analytically compute KV cache tensor size as `2 × layers × KV-heads × head-dim × budget-tokens × 2 bytes (bfloat16)`, isolating cache cost from weights and activations."

---

## Key Findings

- **Memory:** Both H2O and StreamingLLM achieve ~80–83% KV cache reduction at 20% budget. Reduction converges to exactly `1 - budget_ratio` at long sequences.
- **Quality:** H2O consistently outperforms StreamingLLM on QA tasks (importance scoring vs. recency-only eviction). StreamingLLM flat at 10–20% budget on retrieval tasks.
- **Latency:** Compression does NOT reliably reduce latency at short-to-medium sequences. H2O's eager attention overhead can *increase* latency. Primary benefit is memory reduction.
- **Primary benefit of KV compression = memory**, not speed, for moderate sequence lengths.

---

## Common Bugs & Fixes

| Bug | Cause | Fix |
|-----|-------|-----|
| `Attention weights should be of size (1,12,86,39), but is (1,12,86,86)` | `SinkCache.get_usable_length(86,0)` returns negative on empty cache | Override in `StreamingLLMCache` to return 0 when cache empty |
| `Dataset scripts are no longer supported, found LongBench.py` | datasets v3.x dropped script-based loading | Load JSONL directly: `load_dataset("json", data_files=str(jsonl_path))` |
| `Unknown method 'h2o_10pct'` | `create_cache` didn't strip `_10pct` suffix | `re.sub(r"_\d+pct$", "", method)` in `create_cache` |
| `ValueError: Index contains duplicate entries, cannot reshape` | `pivot()` fails with duplicate (task,method) from smoke+full dirs | Use `groupby().mean().unstack()` |
| `ValueError: inhomogeneous shape` in bar chart | `sub.loc[t]` returns Series when duplicates exist | Use `groupby("task")["score"].mean()` then `sub.loc[t]` |

---

## Hardware

| Hardware | Notes |
|----------|-------|
| NVIDIA A100 (Colab) | Full eval ~30 min; use `evaluate_qwen2.5_1.5B.ipynb` |
| Apple M-series (≥8 GB) | `DEVICE_MAP="mps"`; smoke-test recommended |
| CPU | Smoke-test only |

---

## What NOT to Do

- Do not suggest running `run_eval.py` or `run_benchmark.py` — notebook only.
- Do not reuse a `create_cache()` object across requests.
- Do not use `split("_")[0]` to parse method names — breaks `streaming_llm`.
- Do not use `torch.cuda.max_memory_allocated()` for KV cache comparison — use theoretical formula.
- Do not override `max_new_tokens` globally — use per-task defaults from `MAX_NEW_TOKENS` dict in `longbench.py`.
- Do not suggest Phi models for H2O — incompatible layer naming.
