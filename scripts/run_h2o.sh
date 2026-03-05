#!/usr/bin/env bash
# H2O: Heavy-Hitter Oracle KV-cache compression.
# Cache budget = HH_RATIO + RECENT_RATIO of the full context length.
#
# Example — 20 % budget (10 % HH + 10 % recent):
#   ./scripts/run_h2o.sh
#
# Example — 10 % budget:
#   HH_RATIO=0.05 RECENT_RATIO=0.05 ./scripts/run_h2o.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
HH_RATIO="${HH_RATIO:-0.10}"
RECENT_RATIO="${RECENT_RATIO:-0.10}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

EXTRA_ARGS=""
if [ -n "$NUM_SAMPLES" ]; then
  EXTRA_ARGS="--num_samples $NUM_SAMPLES"
fi

# H2O requires eager attention to capture softmax weights.
python run_eval.py \
  --method h2o \
  --model "$MODEL" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --hh_ratio "$HH_RATIO" \
  --recent_ratio "$RECENT_RATIO" \
  --output_dir "$OUTPUT_DIR" \
  $EXTRA_ARGS \
  "$@"
