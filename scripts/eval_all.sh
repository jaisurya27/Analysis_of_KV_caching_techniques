#!/usr/bin/env bash
# Run full evaluation suite: baseline + H2O + StreamingLLM at 10/20/50% budgets.
# Writes all results to $OUTPUT_DIR.
#
# For a quick smoke-test, set NUM_SAMPLES=5:
#   NUM_SAMPLES=5 ./scripts/eval_all.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

export MODEL MAX_SEQ_LEN NUM_SAMPLES OUTPUT_DIR

echo "=== [1/7] Full-attention baseline ==="
bash scripts/run_full.sh

for RATIO in 0.10 0.20 0.50; do
  HALF=$(python3 -c "print($RATIO / 2)")

  echo ""
  echo "=== H2O — budget ${RATIO} (HH=${HALF}, Recent=${HALF}) ==="
  HH_RATIO=$HALF RECENT_RATIO=$HALF bash scripts/run_h2o.sh

  echo ""
  echo "=== StreamingLLM — budget ${RATIO} ==="
  CACHE_RATIO=$RATIO bash scripts/run_streaming_llm.sh
done

echo ""
echo "=== Speed benchmark ==="
python run_benchmark.py \
  --model "$MODEL" \
  --method full h2o streaming_llm \
  --seq_len 512 1024 2048 4096 \
  --gen_len 128 \
  --cache_ratio 0.20

echo ""
echo "All results written to: $OUTPUT_DIR"
