#!/usr/bin/env python3
"""
Main evaluation entry point for KV-cache compression experiments.

Usage examples
--------------
# Full-attention baseline (no compression):
python run_eval.py --method full --model Qwen/Qwen3-8B

# H2O with 20% cache budget (10% HH + 10% recent):
python run_eval.py --method h2o --hh_ratio 0.10 --recent_ratio 0.10

# StreamingLLM with 20% cache budget:
python run_eval.py --method streaming_llm --cache_ratio 0.20

# Quick smoke-test (5 samples per task):
python run_eval.py --method h2o --num_samples 5 --tasks narrativeqa hotpotqa
"""

import argparse
import logging
import os
import sys
import time

import torch

# Make the project importable without installing it.
sys.path.insert(0, os.path.dirname(__file__))

from src.eval.longbench import DEFAULT_TASKS, LongBenchEvaluator
from src.models.patch import apply_kv_method, load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate KV-cache compression on LongBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID.",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit (requires bitsandbytes).",
    )
    p.add_argument(
        "--device_map",
        default="auto",
        help='HuggingFace device_map ("auto", "mps", "cpu", "cuda:0", …).',
    )

    # KV method
    p.add_argument(
        "--method",
        choices=["full", "h2o", "streaming_llm"],
        default="full",
        help="KV-cache compression method.",
    )

    # H2O specific
    p.add_argument(
        "--hh_ratio",
        type=float,
        default=0.10,
        help="[H2O] Fraction of context kept as heavy hitters.",
    )
    p.add_argument(
        "--recent_ratio",
        type=float,
        default=0.10,
        help="[H2O] Fraction of context always kept as recent tokens.",
    )

    # StreamingLLM specific
    p.add_argument(
        "--sink_size",
        type=int,
        default=4,
        help="[StreamingLLM] Number of attention-sink tokens.",
    )
    p.add_argument(
        "--cache_ratio",
        type=float,
        default=0.20,
        help="[StreamingLLM] Total cache budget as fraction of max_seq_len.",
    )

    # Evaluation
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length (tokens) fed to the model.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="LongBench tasks to evaluate.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit samples per task (None = full test set).",
    )
    p.add_argument(
        "--output_dir",
        default="results",
        help="Directory to write result JSON files.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # H2O requires eager attention to capture softmax weights.
    attn_impl = "eager" if args.method == "h2o" else "sdpa"

    logger.info("=" * 60)
    logger.info("Method       : %s", args.method)
    logger.info("Model        : %s", args.model)
    logger.info("Attn impl    : %s", attn_impl)
    logger.info("Max seq len  : %d", args.max_seq_len)
    logger.info("Tasks        : %s", args.tasks)
    if args.num_samples:
        logger.info("Num samples  : %d (per task)", args.num_samples)
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load model + tokenizer                                            #
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        attn_implementation=attn_impl,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
    )

    # ------------------------------------------------------------------ #
    # 2. Apply KV-compression patches                                      #
    # ------------------------------------------------------------------ #
    apply_kv_method(
        model,
        method=args.method,
        hh_ratio=args.hh_ratio,
        recent_ratio=args.recent_ratio,
        sink_size=args.sink_size,
        max_seq_len=args.max_seq_len,
        cache_ratio=args.cache_ratio,
    )

    # ------------------------------------------------------------------ #
    # 3. Determine inference device                                        #
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Inference device: %s", device)

    # ------------------------------------------------------------------ #
    # 4. Run LongBench evaluation                                          #
    # ------------------------------------------------------------------ #
    evaluator = LongBenchEvaluator(
        model=model,
        tokenizer=tokenizer,
        method=args.method,
        tasks=args.tasks,
        max_length=args.max_seq_len,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=device,
    )

    t0 = time.perf_counter()
    results = evaluator.run()
    elapsed = time.perf_counter() - t0

    logger.info("Total evaluation time: %.1f min", elapsed / 60)

    # ------------------------------------------------------------------ #
    # 5. Print concise summary table                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"{'Task':<30} {'Score':>8}  {'Latency':>10}")
    print("-" * 60)
    for task, r in results.items():
        print(f"  {task:<28} {r['score']:>7.2f}%  {r['avg_latency_sec']:>9.3f}s")
    macro = sum(r["score"] for r in results.values()) / max(1, len(results))
    print("-" * 60)
    print(f"  {'Macro average':<28} {macro:>7.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
