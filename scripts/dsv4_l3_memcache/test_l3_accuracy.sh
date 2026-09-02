#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: test_l3_accuracy.sh REQUEST_LENGTH REQUEST_COUNT [100]

Required environment:
  SERVER_LOG   Log file of the running SGLang server

Optional environment:
  SGLANG_DIR   Default: /home/l00951280/code/sglang
  MODEL_PATH   Default: /home/weights/DeepSeek-V4-Flash-w8a8-mtp
  BASE_URL     Default: http://127.0.0.1:30000
  DP_RANKS     Auto-detected from the running server when unset
  RESULT_ROOT  Default: /home/l00951280/dsv4-l3-results/l3-accuracy
  OUTPUT_LEN   Default: 32
  SEED_BASE    Default: 60000

The server must already be running with the ascend_memcache storage backend.
Run this test while no other client is sending requests to the server.

The test compares two executions with the same C128 cache boundary:
  1. resident L1 cache hit;
  2. MemCache L3 restore after flushing L1/L2.

Comparing a cold prefill with a partial-cache prefill is intentionally avoided,
because their different numerical execution paths can produce different output
tokens even when the restored cache data is correct.
EOF
}

if (( $# < 2 || $# > 3 )); then
  usage >&2
  exit 2
fi

REQ_LEN=$1
REQUEST_COUNT=$2
HIT=${3:-100}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:-/home/l00951280/code/sglang}
MODEL_PATH=${MODEL_PATH:-/home/weights/DeepSeek-V4-Flash-w8a8-mtp}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
RESULT_ROOT=${RESULT_ROOT:-/home/l00951280/dsv4-l3-results/l3-accuracy}
OUTPUT_LEN=${OUTPUT_LEN:-32}
SEED_BASE=${SEED_BASE:-60000}
SERVER_LOG=${SERVER_LOG:-}
BENCH_SCRIPT=${BENCH_SCRIPT:-${SCRIPT_DIR}/bench_ids_dsv4.py}

if ! [[ "$REQ_LEN" =~ ^[1-9][0-9]*$ ]]; then
  echo "REQUEST_LENGTH must be a positive integer: $REQ_LEN" >&2
  exit 2
fi
if ! [[ "$REQUEST_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "REQUEST_COUNT must be a positive integer: $REQUEST_COUNT" >&2
  exit 2
fi
if [[ "$HIT" != "100" ]]; then
  echo "This same-boundary accuracy test currently supports HIT=100 only: $HIT" >&2
  exit 2
fi

# DeepSeek V4 C128 stores a complete group of 128 pages x 16 tokens.
if (( REQ_LEN % 2048 != 0 )); then
  echo "REQUEST_LENGTH must be a multiple of 2048: $REQ_LEN" >&2
  exit 2
fi
if (( REQ_LEN < 4096 )); then
  echo "REQUEST_LENGTH must be at least 4096: $REQ_LEN" >&2
  exit 2
fi

if [[ ! -f "$SGLANG_DIR/python/sglang/__init__.py" ]]; then
  echo "SGLANG_DIR is not an SGLang checkout: $SGLANG_DIR" >&2
  exit 2
fi
if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Accuracy driver not found: $BENCH_SCRIPT" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ -z "$SERVER_LOG" ]]; then
  echo "SERVER_LOG must point to the active SGLang server.log" >&2
  exit 2
fi
if [[ ! -f "$SERVER_LOG" ]]; then
  echo "SERVER_LOG does not exist: $SERVER_LOG" >&2
  exit 2
fi

detect_dp_ranks() {
  ps -eo args= | sed -nE \
    '/python[^ ]*.*sglang\.launch_server/ {s/.*--dp-size(=|[[:space:]])([0-9]+).*/\2/p; q;}'
}

DP_RANKS=${DP_RANKS:-$(detect_dp_ranks)}
if ! [[ "$DP_RANKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Cannot detect --dp-size from the running server; export DP_RANKS explicitly." >&2
  exit 2
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
TAG="memcache_l3_accuracy_${REQ_LEN}_${HIT}_${RUN_ID}"
RESULT_DIR="${RESULT_ROOT}/${TAG}"
FIRST_OUT="${RESULT_DIR}/first_out.json"
WARM_LOG="${RESULT_DIR}/warm.log"
BASELINE_LOG="${RESULT_DIR}/l1-baseline.log"
REPLAY_LOG="${RESULT_DIR}/replay.log"
SUMMARY_FILE="${RESULT_DIR}/summary.txt"
mkdir -p "$RESULT_DIR"

# A C128 group contains 128 pages x 16 tokens. SGLang keeps one token out of
# the radix-cache hit and the DSV4 sidecars restore only complete C128 groups,
# so an aligned request reuses one complete group less than its input length.
C128_GROUP_TOKENS=2048
EXPECTED_CACHED_PER_REQUEST=$((REQ_LEN - C128_GROUP_TOKENS))
EXPECTED_CACHED_TOKEN_SUM=$((EXPECTED_CACHED_PER_REQUEST * REQUEST_COUNT))

export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
# Native bench can flush on its own in CI. This accuracy-only flow owns every
# flush boundary, so prevent an unexpected extra flush.
unset SGLANG_IS_IN_CI

flush_l1_l2_when_idle() {
  local attempt
  local output

  for attempt in $(seq 1 15); do
    if output=$(curl -sS --fail-with-body -X POST \
      "${BASE_URL}/flush_cache?timeout=60" --max-time 70 2>&1); then
      echo "[OK] L1/L2 flushed: $output"
      return 0
    fi
    echo "[WAIT] flush attempt ${attempt}/15 did not reach idle: $output" >&2
    sleep 5
  done

  echo "[FAIL] L1/L2 could not reach an idle flush boundary" >&2
  return 1
}

clear_l3_storage() {
  local output
  output=$(curl -sS --fail-with-body -X POST \
    "${BASE_URL}/hicache/storage-backend/clear" --max-time 900)
  echo "[OK] L3 cleared: $output"
}

collect_cached_token_sum() {
  local start_line=$1
  local output_file=$2
  local expected_sum=$3
  local cached_sum=0

  for _ in $(seq 1 30); do
    tail -n "+$((start_line + 1))" "$SERVER_LOG" >"$output_file"
    cached_sum=$(
      sed -nE 's/.*#cached-token: ([0-9]+).*/\1/p' "$output_file" \
        | awk '{sum += $1} END {print sum + 0}'
    )
    if (( cached_sum >= expected_sum )); then
      break
    fi
    sleep 1
  done

  printf '%s\n' "$cached_sum"
}

echo "=== DeepSeek V4 MemCache L3 accuracy test ==="
echo "request_length=$REQ_LEN"
echo "request_count=$REQUEST_COUNT"
echo "stored_prefix_percent=$HIT"
echo "comparison=l1_partial_hit_vs_l3_partial_hit"
echo "expected_cached_per_request=$EXPECTED_CACHED_PER_REQUEST"
echo "output_length=$OUTPUT_LEN"
echo "dp_ranks=$DP_RANKS"
echo "server_log=$SERVER_LOG"
echo "result_dir=$RESULT_DIR"

echo "=== Check server ==="
curl -fsS "${BASE_URL}/health_generate" --max-time 900 >/dev/null
echo "[OK] health_generate"

echo "=== Establish an empty and idle L1/L2/L3 boundary ==="
flush_l1_l2_when_idle
clear_l3_storage
flush_l1_l2_when_idle

echo "=== Warm deterministic prefixes into the resident cache ==="
python3 -u "$BENCH_SCRIPT" \
  --model-path "$MODEL_PATH" \
  --input-len "$REQ_LEN" \
  --num-prompts "$REQUEST_COUNT" \
  --output-len 1 \
  --route roundrobin \
  --dp-ranks "$DP_RANKS" \
  --concurrency "$DP_RANKS" \
  --pop-conc "$DP_RANKS" \
  --hit 100 \
  --populate-only \
  --tag "${TAG}_warm" \
  --server-log "$SERVER_LOG" \
  --seed-base "$SEED_BASE" \
  2>&1 | tee "$WARM_LOG"

# Capture the reference through the same partial-cache execution path that the
# L3 replay will use. The only intended difference is the source of the cached
# pages: resident cache here, MemCache L3 after the flush below.
BASELINE_LOG_START_LINE=$(wc -l <"$SERVER_LOG")

echo "=== Generate resident-cache baseline and save reference output_ids ==="
python3 -u "$BENCH_SCRIPT" \
  --model-path "$MODEL_PATH" \
  --input-len "$REQ_LEN" \
  --num-prompts "$REQUEST_COUNT" \
  --output-len "$OUTPUT_LEN" \
  --route roundrobin \
  --dp-ranks "$DP_RANKS" \
  --concurrency "$DP_RANKS" \
  --pop-conc "$DP_RANKS" \
  --hit 100 \
  --populate-only \
  --tag "${TAG}_l1_baseline" \
  --save-first-out "$FIRST_OUT" \
  --server-log "$SERVER_LOG" \
  --seed-base "$SEED_BASE" \
  2>&1 | tee "$BASELINE_LOG"

if [[ ! -s "$FIRST_OUT" ]]; then
  echo "[FAIL] resident-cache baseline did not save output_ids: $FIRST_OUT" >&2
  exit 1
fi

L1_WINDOW="${RESULT_DIR}/server-l1-baseline-window.log"
L1_CACHED_TOKEN_SUM=$(collect_cached_token_sum \
  "$BASELINE_LOG_START_LINE" "$L1_WINDOW" "$EXPECTED_CACHED_TOKEN_SUM")
if (( L1_CACHED_TOKEN_SUM != EXPECTED_CACHED_TOKEN_SUM )); then
  echo "[FAIL] resident-cache baseline did not use the expected cache boundary" >&2
  echo "expected_cached_token_sum=$EXPECTED_CACHED_TOKEN_SUM" >&2
  echo "actual_cached_token_sum=$L1_CACHED_TOKEN_SUM" >&2
  echo "Inspect $L1_WINDOW" >&2
  exit 1
fi
echo "[OK] resident-cache baseline cached-token sum: $L1_CACHED_TOKEN_SUM"

echo "=== Wait for L2-to-L3 write-through, then drop L1/L2 only ==="
flush_l1_l2_when_idle

# Only replay traffic is allowed after this offset. An exact cached-token sum
# proves that the compared output used the same boundary, now restored from L3.
REPLAY_LOG_START_LINE=$(wc -l <"$SERVER_LOG")

echo "=== Replay from L3 and compare output_ids token by token ==="
python3 -u "$BENCH_SCRIPT" \
  --model-path "$MODEL_PATH" \
  --input-len "$REQ_LEN" \
  --num-prompts "$REQUEST_COUNT" \
  --output-len "$OUTPUT_LEN" \
  --route roundrobin \
  --dp-ranks "$DP_RANKS" \
  --concurrency "$DP_RANKS" \
  --pop-conc "$DP_RANKS" \
  --hit 100 \
  --skip-populate \
  --skip-measure \
  --tag "${TAG}_replay" \
  --load-first-out "$FIRST_OUT" \
  --server-log "$SERVER_LOG" \
  --seed-base "$SEED_BASE" \
  2>&1 | tee "$REPLAY_LOG"

L3_WINDOW="${RESULT_DIR}/server-replay-window.log"
L3_CACHED_TOKEN_SUM=$(collect_cached_token_sum \
  "$REPLAY_LOG_START_LINE" "$L3_WINDOW" "$EXPECTED_CACHED_TOKEN_SUM")
if (( L3_CACHED_TOKEN_SUM != EXPECTED_CACHED_TOKEN_SUM )); then
  echo "[FAIL] L3 replay did not use the expected cache boundary" >&2
  echo "expected_cached_token_sum=$EXPECTED_CACHED_TOKEN_SUM" >&2
  echo "actual_cached_token_sum=$L3_CACHED_TOKEN_SUM" >&2
  echo "Inspect $L3_WINDOW" >&2
  exit 1
fi
echo "[OK] L3 replay cached-token sum: $L3_CACHED_TOKEN_SUM"

if grep -qE 'DIVERGED|FAILED|\[replay\] NOTE:' "$REPLAY_LOG"; then
  echo "[FAIL] L3 replay differs from the same-boundary resident-cache baseline" >&2
  exit 1
fi

EXPECTED_REPLAY="[replay] ${REQUEST_COUNT}/${REQUEST_COUNT} identical"
if ! grep -Fq "$EXPECTED_REPLAY" "$REPLAY_LOG"; then
  echo "[FAIL] expected accuracy result not found: $EXPECTED_REPLAY" >&2
  exit 1
fi

{
  echo "PASS"
  echo "request_length=$REQ_LEN"
  echo "request_count=$REQUEST_COUNT"
  echo "stored_prefix_percent=$HIT"
  echo "comparison=l1_partial_hit_vs_l3_partial_hit"
  echo "output_length=$OUTPUT_LEN"
  echo "dp_ranks=$DP_RANKS"
  echo "identical_outputs=${REQUEST_COUNT}/${REQUEST_COUNT}"
  echo "expected_cached_per_request=$EXPECTED_CACHED_PER_REQUEST"
  echo "l1_cached_token_sum=$L1_CACHED_TOKEN_SUM"
  echo "l3_cached_token_sum=$L3_CACHED_TOKEN_SUM"
  echo "result_dir=$RESULT_DIR"
} | tee "$SUMMARY_FILE"

echo "[PASS] MemCache L3 matches the same-boundary resident-cache baseline"
