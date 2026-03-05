#!/usr/bin/env bash
# StreamingLLM: attention-sink + sliding-window KV-cache compression.
# Total budget = CACHE_RATIO × max_seq_len.
#
# Example — 20 % budget:
#   ./scripts/run_streaming_llm.sh
#
# Example — 50 % budget:
#   CACHE_RATIO=0.50 ./scripts/run_streaming_llm.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
CACHE_RATIO="${CACHE_RATIO:-0.20}"
SINK_SIZE="${SINK_SIZE:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

EXTRA_ARGS=""
if [ -n "$NUM_SAMPLES" ]; then
  EXTRA_ARGS="--num_samples $NUM_SAMPLES"
fi

python run_eval.py \
  --method streaming_llm \
  --model "$MODEL" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --cache_ratio "$CACHE_RATIO" \
  --sink_size "$SINK_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  $EXTRA_ARGS \
  "$@"
