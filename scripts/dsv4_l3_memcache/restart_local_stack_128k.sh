#!/usr/bin/env bash

# Recreate the local DeepSeek-V4 MemCache stack in a known order:
#   stop SGLang -> stop Holder -> stop Meta
#   start/verify Meta -> start/verify Holder -> start/verify SGLang
#
# The script intentionally does not reset any NPU device.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SGLANG_DIR=${SGLANG_DIR:-/home/l00951280/code/sglang}
MODEL_PATH=${MODEL_PATH:-/home/weights/DeepSeek-V4-Flash-w8a8-mtp}
RESULT_ROOT=${RESULT_ROOT:-/home/l00951280/dsv4-l3-results}
SERVER_PORT=${SERVER_PORT:-30000}
NCCL_PORT=${NCCL_PORT:-34001}
HOLDER_CAPACITY=${HOLDER_CAPACITY:-224GB}
HOLDER_DEVICE_ID=${HOLDER_DEVICE_ID:-0}
MEMCACHE_WORLD_SIZE=${MEMCACHE_WORLD_SIZE:-17}
META_HEALTH_URL=${META_HEALTH_URL:-http://127.0.0.1:8000/health}
CAPACITY_URL=${CAPACITY_URL:-http://127.0.0.1:8000/api/v1/capacity/usage}
SERVER_HEALTH_URL=${SERVER_HEALTH_URL:-http://127.0.0.1:${SERVER_PORT}/health_generate}
META_READY_TIMEOUT=${META_READY_TIMEOUT:-120}
HOLDER_READY_TIMEOUT=${HOLDER_READY_TIMEOUT:-900}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-1800}
STOP_TIMEOUT=${STOP_TIMEOUT:-60}

META_CONFIG_SOURCE=${META_CONFIG_SOURCE:-$SCRIPT_DIR/local_meta.example.json}
MEMCACHE_CONFIG_SOURCE=${MEMCACHE_CONFIG_SOURCE:-$SCRIPT_DIR/local_memcache.example.json}

RUN_ID=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=${1:-$RESULT_ROOT/local-stack-$RUN_ID}
mkdir -p "$RESULT_DIR/server"
RESULT_DIR=$(cd "$RESULT_DIR" && pwd)

META_CONFIG="$RESULT_DIR/local_meta.json"
MEMCACHE_CONFIG="$RESULT_DIR/local_memcache.json"
META_LOG="$RESULT_DIR/meta.log"
HOLDER_LOG="$RESULT_DIR/holder.log"
SERVER_LOG="$RESULT_DIR/server/server.log"

SUCCESS=0
STACK_TOUCHED=0

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*" >&2
  return 1
}

matching_pids() {
  local pattern=$1
  pgrep -f -- "$pattern" 2>/dev/null | awk -v self="$$" '$1 != self' || true
}

pid_still_matches() {
  local pid=$1
  local pattern=$2
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Eq -- "$pattern"
}

stop_matching() {
  local label=$1
  local pattern=$2
  local pids
  local deadline
  local alive

  pids=$(matching_pids "$pattern")
  if [[ -z "$pids" ]]; then
    log "$label is not running"
    return 0
  fi

  log "Stopping $label: $(tr '\n' ' ' <<<"$pids")"
  # The process list comes from the exact command patterns below, rather than
  # an unscoped killall/pkill operation.
  kill -TERM $pids 2>/dev/null || true
  deadline=$((SECONDS + STOP_TIMEOUT))
  while (( SECONDS < deadline )); do
    alive=""
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      if pid_still_matches "$pid" "$pattern"; then
        alive+="$pid "
      fi
    done <<<"$pids"
    if [[ -z "$alive" ]]; then
      log "$label stopped"
      return 0
    fi
    sleep 1
  done

  log "$label did not stop after ${STOP_TIMEOUT}s; sending SIGKILL to: $alive"
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if pid_still_matches "$pid" "$pattern"; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done < <(tr ' ' '\n' <<<"$alive")
}

stop_stack() {
  # Stop the HTTP parent first so it can perform normal scheduler cleanup.
  stop_matching "SGLang launcher" 'python[^ ]* .*sglang\.launch_server'
  # Catch an orphaned worker only after the launcher has had time to exit.
  stop_matching "orphaned SGLang workers" 'sglang::(scheduler|detokenizer|tokenizer|engine)'
  stop_matching "MemCache Holder" 'python[^ ]* .*start_local_holder\.py'
  stop_matching "MemCache Meta" 'python[^ ]* .*start_meta_service\.py'
}

show_failure_logs() {
  local path
  for path in "$META_LOG" "$HOLDER_LOG" "$SERVER_LOG"; do
    if [[ -f "$path" ]]; then
      log "Last 80 lines of $path" >&2
      tail -n 80 "$path" >&2 || true
    fi
  done
}

cleanup_on_exit() {
  local status=$?
  if (( STACK_TOUCHED == 1 && (status != 0 || SUCCESS == 0) )); then
    set +e
    log "Startup did not complete; cleaning up the partial stack" >&2
    stop_stack
    show_failure_logs
  fi
}
trap cleanup_on_exit EXIT

wait_for_http() {
  local label=$1
  local url=$2
  local timeout=$3
  local deadline=$((SECONDS + timeout))

  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 10 "$url" >/dev/null 2>&1; then
      log "$label is healthy: $url"
      return 0
    fi
    sleep 2
  done
  die "$label did not become healthy within ${timeout}s: $url"
}

wait_for_holder() {
  local pid=$1
  local deadline=$((SECONDS + HOLDER_READY_TIMEOUT))

  while (( SECONDS < deadline )); do
    if grep -q 'LOCAL_HOLDER_READY' "$HOLDER_LOG" 2>/dev/null; then
      log "Holder emitted LOCAL_HOLDER_READY"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      die "Holder process $pid exited before becoming ready"
    fi
    sleep 2
  done
  die "Holder did not become ready within ${HOLDER_READY_TIMEOUT}s"
}

wait_for_capacity() {
  local deadline=$((SECONDS + HOLDER_READY_TIMEOUT))
  local total_bytes

  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 10 "$CAPACITY_URL" \
      -o "$RESULT_DIR/capacity-usage.json" 2>/dev/null; then
      total_bytes=$(python3 - "$RESULT_DIR/capacity-usage.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    payload = json.load(file)
print(int(payload.get("cpu", {}).get("total_bytes", 0)))
PY
      )
      if [[ "$total_bytes" =~ ^[0-9]+$ ]] && (( total_bytes > 0 )); then
        log "Holder capacity registered: cpu.total_bytes=$total_bytes"
        return 0
      fi
    fi
    sleep 2
  done
  die "Holder stayed ready but cpu.total_bytes was not registered"
}

wait_for_server() {
  local pid=$1
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  local next_progress=$((SECONDS + 30))

  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 10 "$SERVER_HEALTH_URL" >/dev/null 2>&1; then
      log "SGLang is healthy: $SERVER_HEALTH_URL"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      die "SGLang launcher $pid exited before /health_generate became ready"
    fi
    if (( SECONDS >= next_progress )); then
      log "Waiting for SGLang /health_generate (elapsed=$((SERVER_READY_TIMEOUT - (deadline - SECONDS)))s)"
      next_progress=$((SECONDS + 30))
    fi
    sleep 5
  done
  die "SGLang did not become healthy within ${SERVER_READY_TIMEOUT}s"
}

require_file() {
  [[ -f "$1" ]] || die "Required file does not exist: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Required directory does not exist: $1"
}

require_file "$SCRIPT_DIR/start_meta_service.py"
require_file "$SCRIPT_DIR/start_local_holder.py"
require_file "$SCRIPT_DIR/run_server_128k.sh"
require_file "$META_CONFIG_SOURCE"
require_file "$MEMCACHE_CONFIG_SOURCE"
require_file "$SGLANG_DIR/python/sglang/__init__.py"
require_dir "$MODEL_PATH"
command -v curl >/dev/null || die "curl is required"
command -v pgrep >/dev/null || die "pgrep is required"

log "Result directory: $RESULT_DIR"
log "Stopping the existing local stack"
STACK_TOUCHED=1
stop_stack

cp -- "$META_CONFIG_SOURCE" "$META_CONFIG"
cp -- "$MEMCACHE_CONFIG_SOURCE" "$MEMCACHE_CONFIG"
git -C "$SGLANG_DIR" rev-parse HEAD >"$RESULT_DIR/sglang-commit.txt"
git -C "$(cd "$SCRIPT_DIR/../.." && pwd)" rev-parse HEAD \
  >"$RESULT_DIR/tools-commit.txt"
python3 -m pip show deep_ep >"$RESULT_DIR/deepep-package.txt" 2>&1 || true

log "Starting MemCache Meta"
nohup python3 "$SCRIPT_DIR/start_meta_service.py" "$META_CONFIG" \
  >"$META_LOG" 2>&1 &
META_PID=$!
echo "$META_PID" >"$RESULT_DIR/meta.pid"
wait_for_http "MemCache Meta" "$META_HEALTH_URL" "$META_READY_TIMEOUT"

log "Starting MemCache Holder: capacity=$HOLDER_CAPACITY device=$HOLDER_DEVICE_ID world_size=$MEMCACHE_WORLD_SIZE"
MEMCACHE_WORLD_SIZE="$MEMCACHE_WORLD_SIZE" \
  nohup python3 "$SCRIPT_DIR/start_local_holder.py" \
  "$HOLDER_CAPACITY" "$HOLDER_DEVICE_ID" >"$HOLDER_LOG" 2>&1 &
HOLDER_PID=$!
echo "$HOLDER_PID" >"$RESULT_DIR/holder.pid"
wait_for_holder "$HOLDER_PID"
wait_for_http "MemCache Meta after Holder registration" \
  "$META_HEALTH_URL" "$META_READY_TIMEOUT"
wait_for_capacity

log "Starting SGLang: model=$MODEL_PATH port=$SERVER_PORT"
SGLANG_DIR="$SGLANG_DIR" \
MODEL_PATH="$MODEL_PATH" \
MEMCACHE_CONFIG="$MEMCACHE_CONFIG" \
SERVER_PORT="$SERVER_PORT" \
NCCL_PORT="$NCCL_PORT" \
SGLANG_DSV4_L3_DIAGNOSTICS="${SGLANG_DSV4_L3_DIAGNOSTICS:-0}" \
  bash "$SCRIPT_DIR/run_server_128k.sh" "$RESULT_DIR/server"

SERVER_PID=$(cat "$RESULT_DIR/server/launch.pid")
wait_for_server "$SERVER_PID"

ln -sfn "$RESULT_DIR" "$RESULT_ROOT/latest-local-stack"
SUCCESS=1

log "Local DSV4 MemCache stack is ready"
printf '\n'
printf 'RESULT_DIR=%q\n' "$RESULT_DIR"
printf 'META_PID=%s\n' "$META_PID"
printf 'HOLDER_PID=%s\n' "$HOLDER_PID"
printf 'SERVER_PID=%s\n' "$SERVER_PID"
printf 'SERVER_LOG=%q\n' "$SERVER_LOG"
printf 'Run the accuracy test with:\n'
printf '  DP_RANKS=16 SERVER_LOG=%q bash %q 131072 32 100\n' \
  "$SERVER_LOG" "$SCRIPT_DIR/test_l3_accuracy.sh"
