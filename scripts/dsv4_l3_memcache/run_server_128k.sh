#!/bin/bash

set -e
# CANN vendor environment scripts read optional variables without defaults.
# Disable nounset explicitly because it may be inherited through SHELLOPTS.
set +u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:?set SGLANG_DIR to the lc-dsv4-l3 checkout}
test -f "$SGLANG_DIR/python/sglang/__init__.py" || {
  echo "SGLANG_DIR does not contain an SGLang checkout: $SGLANG_DIR" >&2
  exit 1
}
MODEL_PATH=${MODEL_PATH:-/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp}
MEMCACHE_CONFIG=${MEMCACHE_CONFIG:-$SCRIPT_DIR/local_memcache.example.json}
SERVER_PORT=${SERVER_PORT:-30000}
NCCL_PORT=${NCCL_PORT:-34001}
RUN_E=${1:?usage: run_server_128k.sh RESULT_DIR}
mkdir -p "$RUN_E"
RUN_E=$(cd "$RUN_E" && pwd)

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH=$SGLANG_DIR/python:/sgl-workspace/sglang/python:${PYTHONPATH:-}
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export FORCE_DRAFT_MODEL_NON_QUANT=1
export HCCL_BUFFSIZE=1500
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export INF_NAN_MODE_FORCE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
cd "$SGLANG_DIR"
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --page-size 128 --tp-size 16 --trust-remote-code --device npu \
  --attention-backend ascend --watchdog-timeout 9000 \
  --disable-cuda-graph \
  --host 0.0.0.0 --port "$SERVER_PORT" --nccl-port "$NCCL_PORT" \
  --mem-fraction-static 0.60 --swa-full-tokens-ratio 0.5 \
  --prefill-max-requests 1 --chunked-prefill-size 32768 \
  --max-running-requests 16 --dp-size 16 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal \
  --quantization modelslim --enable-dp-lm-head \
  --kv-cache-dtype auto --random-seed 20260807 --context-length 264448 \
  --max-total-tokens 264448 \
  --enable-hierarchical-cache --hicache-io-backend kernel_ascend \
  --hicache-ratio 2.0 --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct \
  --hicache-storage-backend ascend_memcache \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "@$MEMCACHE_CONFIG" \
  >>"$RUN_E/server.log" 2>&1 &

echo "$RUN_E/server.log"
echo $! >"$RUN_E/launch.pid"
