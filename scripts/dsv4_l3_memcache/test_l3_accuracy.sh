#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: test_l3_accuracy.sh REQUEST_LENGTH REQUEST_COUNT [100|50]

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
if [[ "$HIT" != "100" && "$HIT" != "50" ]]; then
  echo "HIT must be 100 or 50: $HIT" >&2
  exit 2
fi

# DeepSeek V4 C128 stores a complete group of 128 pages x 16 tokens.
if (( REQ_LEN % 2048 != 0 )); then
  echo "REQUEST_LENGTH must be a multiple of 2048: $REQ_LEN" >&2
  exit 2
fi
if [[ "$HIT" == "50" ]] && (( REQ_LEN % 4096 != 0 )); then
  echo "50% mode requires REQUEST_LENGTH to be a multiple of 4096: $REQ_LEN" >&2
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
POPULATE_LOG="${RESULT_DIR}/populate.log"
REPLAY_LOG="${RESULT_DIR}/replay.log"
SUMMARY_FILE="${RESULT_DIR}/summary.txt"
mkdir -p "$RESULT_DIR"

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

echo "=== DeepSeek V4 MemCache L3 accuracy test ==="
echo "request_length=$REQ_LEN"
echo "request_count=$REQUEST_COUNT"
echo "stored_prefix_percent=$HIT"
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

echo "=== Populate deterministic prefixes and save reference output_ids ==="
python3 -u "$BENCH_SCRIPT" \
  --model-path "$MODEL_PATH" \
  --input-len "$REQ_LEN" \
  --num-prompts "$REQUEST_COUNT" \
  --output-len "$OUTPUT_LEN" \
  --route roundrobin \
  --dp-ranks "$DP_RANKS" \
  --concurrency "$DP_RANKS" \
  --pop-conc "$DP_RANKS" \
  --hit "$HIT" \
  --populate-only \
  --tag "${TAG}_populate" \
  --save-first-out "$FIRST_OUT" \
  --server-log "$SERVER_LOG" \
  --seed-base "$SEED_BASE" \
  2>&1 | tee "$POPULATE_LOG"

if [[ ! -s "$FIRST_OUT" ]]; then
  echo "[FAIL] populate did not save reference output_ids: $FIRST_OUT" >&2
  exit 1
fi

echo "=== Wait for L2-to-L3 write-through, then drop L1/L2 only ==="
flush_l1_l2_when_idle

# Only replay traffic is allowed after this offset. A positive cached-token
# count therefore proves that the compared output used data retained in L3.
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
  --hit "$HIT" \
  --skip-populate \
  --skip-measure \
  --tag "${TAG}_replay" \
  --load-first-out "$FIRST_OUT" \
  --server-log "$SERVER_LOG" \
  --seed-base "$SEED_BASE" \
  2>&1 | tee "$REPLAY_LOG"

if grep -qE 'DIVERGED|FAILED|\[replay\] NOTE:' "$REPLAY_LOG"; then
  echo "[FAIL] replay output_ids differ from populate output_ids" >&2
  exit 1
fi

EXPECTED_REPLAY="[replay] ${REQUEST_COUNT}/${REQUEST_COUNT} identical"
if ! grep -Fq "$EXPECTED_REPLAY" "$REPLAY_LOG"; then
  echo "[FAIL] expected accuracy result not found: $EXPECTED_REPLAY" >&2
  exit 1
fi

L3_WINDOW="${RESULT_DIR}/server-replay-window.log"
CACHED_TOKEN_SUM=0
for _ in $(seq 1 30); do
  tail -n "+$((REPLAY_LOG_START_LINE + 1))" "$SERVER_LOG" >"$L3_WINDOW"
  CACHED_TOKEN_SUM=$(
    sed -nE 's/.*#cached-token: ([0-9]+).*/\1/p' "$L3_WINDOW" \
      | awk '{sum += $1} END {print sum + 0}'
  )
  if (( CACHED_TOKEN_SUM > 0 )); then
    break
  fi
  sleep 1
done
if (( CACHED_TOKEN_SUM <= 0 )); then
  echo "[FAIL] replay was accurate but no post-flush cache hit was recorded" >&2
  echo "Inspect $L3_WINDOW and confirm that this server build logs #cached-token." >&2
  exit 1
fi

{
  echo "PASS"
  echo "request_length=$REQ_LEN"
  echo "request_count=$REQUEST_COUNT"
  echo "stored_prefix_percent=$HIT"
  echo "output_length=$OUTPUT_LEN"
  echo "dp_ranks=$DP_RANKS"
  echo "identical_outputs=${REQUEST_COUNT}/${REQUEST_COUNT}"
  echo "post_flush_cached_token_sum=$CACHED_TOKEN_SUM"
  echo "result_dir=$RESULT_DIR"
} | tee "$SUMMARY_FILE"

echo "[PASS] MemCache L3 replay output_ids are token-for-token identical"
