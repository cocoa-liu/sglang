#!/usr/bin/env bash

# Compare deterministic GSM8K accuracy before and after dropping L1/L2. The
# second pass must retain accuracy and report an actual MemCache L3 hit.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the served model path}
RESULT_ROOT=${RESULT_ROOT:-/home/l00951280/dsv4-l3-results/gsm8k}
NUM_EXAMPLES=${NUM_EXAMPLES:-500}
NUM_THREADS=${NUM_THREADS:-64}
NUM_SHOTS=${NUM_SHOTS:-20}
MAX_TOKENS=${MAX_TOKENS:-512}
MIN_ACCURACY=${MIN_ACCURACY:-0.90}
MAX_ACCURACY_DRIFT=${MAX_ACCURACY_DRIFT:-0.03}
WRITE_SETTLE_SECONDS=${WRITE_SETTLE_SECONDS:-30}
GSM8K_DATA_PATH=${GSM8K_DATA_PATH:-}

RUN_ID=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=${1:-$RESULT_ROOT/gsm8k-l3-$RUN_ID}
mkdir -p "$RESULT_DIR"
RESULT_DIR=$(cd "$RESULT_DIR" && pwd)
INITIAL_JSON="$RESULT_DIR/initial.json"
REPLAY_JSON="$RESULT_DIR/l3-replay.json"

flush_l1_l2_when_idle() {
  local output
  for attempt in $(seq 1 15); do
    if output=$(curl -sS --fail-with-body -X POST \
      "$BASE_URL/flush_cache?timeout=60" --max-time 70 2>&1); then
      echo "[OK] L1/L2 flushed: $output"
      return 0
    fi
    echo "[WAIT] flush attempt $attempt/15: $output" >&2
    sleep 5
  done
  echo "[FAIL] L1/L2 did not reach an idle boundary" >&2
  return 1
}

run_phase() {
  local phase=$1
  local output=$2
  local data_args=()
  if [[ -n "$GSM8K_DATA_PATH" ]]; then
    data_args=(--gsm8k-data-path "$GSM8K_DATA_PATH")
  fi
  python3 -u "$SCRIPT_DIR/gsm8k_l3_eval.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL_PATH" \
    --output "$output" \
    --num-examples "$NUM_EXAMPLES" \
    --num-threads "$NUM_THREADS" \
    --num-shots "$NUM_SHOTS" \
    --max-tokens "$MAX_TOKENS" \
    "${data_args[@]}" 2>&1 | tee "$RESULT_DIR/$phase.log"
}

test -f "$SCRIPT_DIR/gsm8k_l3_eval.py" || {
  echo "Missing evaluator: $SCRIPT_DIR/gsm8k_l3_eval.py" >&2
  exit 1
}
curl -fsS "$BASE_URL/health_generate" --max-time 900 >/dev/null

echo "=== GSM8K MemCache L3 accuracy test ==="
echo "examples=$NUM_EXAMPLES threads=$NUM_THREADS shots=$NUM_SHOTS"
echo "min_accuracy=$MIN_ACCURACY max_drift=$MAX_ACCURACY_DRIFT"
echo "result_dir=$RESULT_DIR"

echo "=== Establish empty L1/L2/L3 boundary ==="
flush_l1_l2_when_idle
curl -sS --fail-with-body -X POST \
  "$BASE_URL/hicache/storage-backend/clear" --max-time 900
flush_l1_l2_when_idle

echo "=== Phase 1: GSM8K baseline and L3 population ==="
run_phase initial "$INITIAL_JSON"

echo "=== Wait for write-through and drop L1/L2 only ==="
sleep "$WRITE_SETTLE_SECONDS"
flush_l1_l2_when_idle

echo "=== Phase 2: GSM8K replay from MemCache L3 ==="
run_phase l3-replay "$REPLAY_JSON"

python3 - "$INITIAL_JSON" "$REPLAY_JSON" \
  "$MIN_ACCURACY" "$MAX_ACCURACY_DRIFT" <<'PY'
import json
import sys

initial_path, replay_path = sys.argv[1:3]
minimum = float(sys.argv[3])
max_drift = float(sys.argv[4])
with open(initial_path, encoding="utf-8") as file:
    initial = json.load(file)
with open(replay_path, encoding="utf-8") as file:
    replay = json.load(file)

initial_score = float(initial["score"])
replay_score = float(replay["score"])
drift = abs(replay_score - initial_score)
errors = []
if initial["failed_requests"]:
    errors.append(f"baseline failed_requests={initial['failed_requests']}")
if replay["failed_requests"]:
    errors.append(f"replay failed_requests={replay['failed_requests']}")
if initial_score < minimum:
    errors.append(f"baseline score {initial_score:.4f} < {minimum:.4f}")
if replay_score < minimum:
    errors.append(f"L3 score {replay_score:.4f} < {minimum:.4f}")
if drift > max_drift:
    errors.append(f"accuracy drift {drift:.4f} > {max_drift:.4f}")
if replay["storage_cached_tokens"] <= 0:
    errors.append("L3 replay reported zero storage cached tokens")

summary = {
    "baseline_score": initial_score,
    "l3_score": replay_score,
    "accuracy_drift": drift,
    "l3_cached_tokens": replay["cached_tokens"],
    "l3_storage_cached_tokens": replay["storage_cached_tokens"],
    "l3_requests_with_storage_cache": replay["requests_with_storage_cache"],
}
print("GSM8K_L3_SUMMARY " + json.dumps(summary))
if errors:
    for error in errors:
        print("[FAIL] " + error, file=sys.stderr)
    raise SystemExit(1)
print("[PASS] GSM8K accuracy is stable after MemCache L3 reload")
PY
