#!/usr/bin/env python3
"""
Speed benchmark: compare throughput and memory of full vs H2O vs StreamingLLM.

Measures:
  - Time-to-first-token (TTFT) — prefill latency.
  - Tokens-per-second during decoding.
  - Peak GPU memory (via torch.cuda.max_memory_allocated).

Usage:
  python run_benchmark.py --method full h2o streaming_llm \
      --seq_len 1024 2048 4096 --gen_len 128 --model Qwen/Qwen3-1.7B
"""

import argparse
import logging
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.models.patch import apply_kv_method, create_cache, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


def benchmark_one(
    model,
    tokenizer,
    method: str,
    seq_len: int,
    gen_len: int,
    device: str,
    warmup: int = 1,
    repeats: int = 3,
) -> dict:
    """Run timing + memory benchmark for a single (method, seq_len) pair."""

    # Build a synthetic prompt of exactly seq_len tokens.
    dummy_ids = torch.randint(
        100, tokenizer.vocab_size - 1, (1, seq_len), device=device
    )

    ttfts, tpss, mems = [], [], []

    for run in range(warmup + repeats):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        cache = create_cache(model, method)
        gen_kwargs: dict = dict(
            max_new_tokens=gen_len,
            do_sample=False,
            use_cache=True,
        )
        if cache is not None:
            gen_kwargs["past_key_values"] = cache

        t_start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(dummy_ids, **gen_kwargs)
        if device == "cuda":
            torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        actual_gen = out.shape[-1] - seq_len
        total_time = t_end - t_start
        tps = actual_gen / total_time if total_time > 0 else 0.0

        if device == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
        else:
            peak_mem_mb = 0.0

        if run >= warmup:
            ttfts.append(total_time - actual_gen / max(tps, 1e-6))
            tpss.append(tps)
            mems.append(peak_mem_mb)

    return {
        "method": method,
        "seq_len": seq_len,
        "gen_len": gen_len,
        "avg_tps": round(sum(tpss) / len(tpss), 2),
        "avg_total_time_sec": round(sum(tpss) / len(tpss) * gen_len, 3) if tpss else 0.0,
        "peak_mem_mb": round(sum(mems) / max(len(mems), 1), 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--method", nargs="+", default=["full", "h2o", "streaming_llm"])
    p.add_argument("--seq_len", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    p.add_argument("--gen_len", type=int, default=128)
    p.add_argument("--cache_ratio", type=float, default=0.20)
    p.add_argument("--hh_ratio", type=float, default=0.10)
    p.add_argument("--recent_ratio", type=float, default=0.10)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--load_in_4bit", action="store_true")
    args = p.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    results = []

    for method in args.method:
        attn_impl = "eager" if method == "h2o" else "sdpa"
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model,
            attn_implementation=attn_impl,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
        )
        max_seq = max(args.seq_len)
        apply_kv_method(
            model,
            method=method,
            hh_ratio=args.hh_ratio,
            recent_ratio=args.recent_ratio,
            max_seq_len=max_seq,
            cache_ratio=args.cache_ratio,
        )

        for sl in args.seq_len:
            logger.info("Benchmarking method=%s  seq_len=%d …", method, sl)
            r = benchmark_one(
                model, tokenizer, method, sl, args.gen_len, device
            )
            results.append(r)
            logger.info(
                "  tps=%.2f  peak_mem=%.1f MB",
                r["avg_tps"],
                r["peak_mem_mb"],
            )

        del model

    # Print table.
    print("\n" + "=" * 68)
    print(f"{'Method':<16} {'SeqLen':>8} {'TPS':>10} {'PeakMem(MB)':>14}")
    print("-" * 68)
    for r in results:
        print(
            f"  {r['method']:<14} {r['seq_len']:>8} "
            f"{r['avg_tps']:>10.2f} {r['peak_mem_mb']:>14.1f}"
        )
    print("=" * 68)

    import json, pathlib
    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Benchmark saved to results/benchmark.json")


if __name__ == "__main__":
    main()
