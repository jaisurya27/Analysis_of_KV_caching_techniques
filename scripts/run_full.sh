#!/usr/bin/env bash
# Full-attention baseline (no KV compression).
# Runtime: ~2–4 h for the full LongBench test sets on a single A100.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-}"    # empty = full test set
OUTPUT_DIR="${OUTPUT_DIR:-results}"

EXTRA_ARGS=""
if [ -n "$NUM_SAMPLES" ]; then
  EXTRA_ARGS="--num_samples $NUM_SAMPLES"
fi

python run_eval.py \
  --method full \
  --model "$MODEL" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --output_dir "$OUTPUT_DIR" \
  $EXTRA_ARGS \
  "$@"
