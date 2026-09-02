# DeepSeek V4 HiCache L3 test scripts

This directory contains the scripts used to deploy local MemCache and validate
the DeepSeek V4 Flash 128K HiCache L3 path on Ascend NPU. The
`lc-dsv4-l3-test-tools` branch contains only these tools. Keep the SGLang
runtime code in a separate `lc-dsv4-l3` checkout.

## Files

- `start_meta_service.py`: starts the MemCache metadata service.
- `start_local_holder.py`: allocates the local MemCache holder.
- `local_meta.example.json`: metadata service configuration.
- `local_memcache.example.json`: SGLang MemCache client configuration.
- `restart_local_stack_128k.sh`: cleanly recreates and validates Meta, Holder,
  and the 16-NPU SGLang service in one command.
- `run_server_128k.sh`: starts the 16-NPU DeepSeek V4 Flash server.
- `local_128k_capacity_smoke.py`: performs a small cold-write and L3-hit probe.
- `bench_l3_matrix.py`: runs the two-cohort 128K validation workload.
- `verify_128k_result.py`: checks the expected hit coverage in the result JSON.
- `bench_ids_dsv4.py`: deterministic token-ID populate/replay driver copied
  from `cx22757/sglang` `main-L3` commit `36eeed06a72a`.
- `test_l3_accuracy.sh`: validates deterministic output-token accuracy after
  flushing L1/L2 and replaying the saved prefix from MemCache L3.

## Validated topology

The scripts use the following topology by default:

- Ascend NPU with 16 visible devices.
- CANN 9.0.0 container environment.
- DeepEP built from the latest `main` branch of
  `sgl-project/sgl-kernel-npu`. The latest commit checked on 2026-08-27 was
  `c28ea2a940a53c00f3a0322d9576210a7f5ae92f`.
- Model path: `/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp`.
- TP=16, DP=16, EP=16, DeepEP `normal` mode.
- Local MemCache using `device_sdma`.
- One 224 GB holder and 16 SGLang clients. `world_size=17` represents those 17
  MemCache participants.
- HiCache `page_first_direct`, page size 128, and write-through storage.

Build and install DeepEP from the current upstream `main` branch before each
new test cycle. Record the exact commit in the result directory. Do not apply
the earlier private `roundMagic/dataState` patch.

## Prepare the environment

Run all processes in the same container. The container must provide
`memcache_hybrid`, PyTorch NPU, the custom CANN operators, and the model. Use
separate directories for the tools-only branch and the SGLang functional
branch.

```bash
git clone --branch lc-dsv4-l3-test-tools --single-branch \
  git@github.com:cocoa-liu/sglang.git /path/to/dsv4-l3-tools
git clone --branch lc-dsv4-l3 --single-branch \
  git@github.com:cocoa-liu/sglang.git /path/to/sglang-lc-dsv4-l3

export TOOLS_DIR=/path/to/dsv4-l3-tools/scripts/dsv4_l3_memcache
export SGLANG_DIR=/path/to/sglang-lc-dsv4-l3
export RESULT_DIR=/data/dsv4-l3-128k-$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"

git -C "$SGLANG_DIR" rev-parse HEAD >"$RESULT_DIR/sglang-commit.txt"
git -C /path/to/sgl-kernel-npu rev-parse HEAD \
  >"$RESULT_DIR/deepep-commit.txt"
python3 -m pip show deep_ep >"$RESULT_DIR/deepep-package.txt"
```

The 224 GB value is the holder allocation used by the two-cohort test. It is
not a disk allocation. MemCache must obtain host RAM backed by enough contiguous
physical pages. The `max_dram_size=600GB` client setting is an upper bound; it
does not allocate 600 GB for each rank.

## Start MemCache

### One-command clean restart (recommended)

The following script stops the existing local SGLang, Holder, and Meta
processes, then starts and validates the complete stack in dependency order. It
does not reset the NPU. Defaults match the established test-machine layout:

```bash
export TOOLS_DIR=/home/l00951280/code/sglang-tools/scripts/dsv4_l3_memcache
bash "$TOOLS_DIR/restart_local_stack_128k.sh"
```

It verifies all of the following before returning successfully:

- the Meta health endpoint returns HTTP 200;
- the Holder emits `LOCAL_HOLDER_READY` and reports a nonzero
  `cpu.total_bytes`;
- the SGLang launcher remains alive during initialization;
- `http://127.0.0.1:30000/health_generate` returns HTTP 200.

Each run gets a new directory under
`/home/l00951280/dsv4-l3-results/local-stack-<timestamp>`. The stable symlink
`/home/l00951280/dsv4-l3-results/latest-local-stack` points to the latest
successful run. Override paths or capacity with environment variables when
needed:

```bash
SGLANG_DIR=/path/to/sglang \
MODEL_PATH=/path/to/DeepSeek-V4-Flash-w8a8-mtp \
RESULT_ROOT=/data/dsv4-l3-results \
HOLDER_CAPACITY=224GB \
bash "$TOOLS_DIR/restart_local_stack_128k.sh"
```

If any stage fails, the script stops the partial stack and prints the tail of
the Meta, Holder, and SGLang logs. The manual commands below remain useful for
debugging individual stages.

Start the metadata service first:

```bash
nohup python3 "$TOOLS_DIR/start_meta_service.py" \
  "$TOOLS_DIR/local_meta.example.json" \
  >"$RESULT_DIR/meta.log" 2>&1 &
echo $! >"$RESULT_DIR/meta.pid"
```

Start one 224 GB local holder:

```bash
MEMCACHE_WORLD_SIZE=17 nohup python3 "$TOOLS_DIR/start_local_holder.py" \
  224GB 0 >"$RESULT_DIR/holder.log" 2>&1 &
echo $! >"$RESULT_DIR/holder.pid"
```

Wait until the holder log contains `LOCAL_HOLDER_READY`, then check the metrics
endpoint:

```bash
grep LOCAL_HOLDER_READY "$RESULT_DIR/holder.log"
curl -f http://127.0.0.1:8000/health
```

If the holder cannot allocate contiguous physical pages, stop the holder,
release unused host-memory processes, and retry the holder allocation. Do not
use an NPU reset to solve a host-memory allocation failure.

## Start SGLang

The tools-only branch does not contain the SGLang Python package. Set
`SGLANG_DIR` to the separate `lc-dsv4-l3` checkout. You may also override
`MODEL_PATH`, `MEMCACHE_CONFIG`, or `SERVER_PORT` when the defaults do not match
your container.

```bash
SGLANG_DIR=/path/to/sglang-lc-dsv4-l3 \
MODEL_PATH=/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp \
MEMCACHE_CONFIG="$TOOLS_DIR/local_memcache.example.json" \
SERVER_PORT=30000 \
bash "$TOOLS_DIR/run_server_128k.sh" "$RESULT_DIR/server"

tail -f "$RESULT_DIR/server/server.log"
```

The script enables structured DSV4 L3 diagnostics by default. Every record is
prefixed with `[DSV4_L3_DIAG]`; a forward or MemCache call blocked for 60 seconds
also dumps all Python thread stacks into `server.log`. The first run does not
force NPU synchronization, so it preserves the original execution timing.

After a failure, collect the compact timeline and the surrounding stack dumps:

```bash
grep -n 'DSV4_L3_DIAG' "$RESULT_DIR/server/server.log" \
  >"$RESULT_DIR/server/dsv4-l3-diag.log"
grep -nE 'DSV4_L3_DIAG|Current thread|Thread 0x|507014|507015|AICore|Traceback' \
  "$RESULT_DIR/server/server.log" | tail -n 2000
```

Only if the first diagnostic run reaches a DeepEP boundary but cannot identify
which previously queued NPU operation failed, rerun with the intrusive sync
probe enabled:

```bash
SGLANG_DSV4_L3_DIAGNOSTICS_SYNC=1 \
  bash "$TOOLS_DIR/run_server_128k.sh" "$RESULT_DIR/server-sync"
```

Do not use sync-mode throughput or latency as performance results.

Do not start another SGLang service on the same 16 devices. Wait for the server
health endpoint before sending the 128K workload:

```bash
curl -f http://127.0.0.1:30000/health_generate
```

## Run a 128K smoke test

The smoke test writes four 128K prompts, flushes L1/L2, and reads one prompt
from L3:

```bash
python3 "$TOOLS_DIR/local_128k_capacity_smoke.py" \
  --base-url http://127.0.0.1:30000 \
  | tee "$RESULT_DIR/smoke-128k.log"
```

It exits successfully only when the final request reports 129024 L3 tokens.
That is 98.4375% of a 131072-token prompt; the first 2048 tokens are not part of
the eligible storage prefix in this configuration.

## Run the two-cohort 128K validation

This is the previously used 128K acceptance workload. It runs two cohorts
without restarting the server between them. Each cohort populates 16 prompts
in two batches of eight, then sends 16 full-hit requests and 16 half-hit
requests. Storage is cleared only between cohorts.

```bash
python3 "$TOOLS_DIR/bench_l3_matrix.py" \
  --base-url http://127.0.0.1:30000 \
  --phase warm128 \
  --count-override 32 \
  --cohort-size 16 \
  --output "$RESULT_DIR/warm128-two-cohorts.json" \
  | tee "$RESULT_DIR/warm128-two-cohorts.log"

python3 "$TOOLS_DIR/verify_128k_result.py" \
  "$RESULT_DIR/warm128-two-cohorts.json"
```

The result passes when:

- all 32 full-hit requests report 129024 storage tokens, or 98.4375%;
- all 32 half-hit requests report 65536 storage tokens, or 50%;
- both cohorts complete without restarting the server;
- the server remains healthy after the workload;
- the server log has no AICore 507014/507015, DeepEP timeout, scheduler exit, or
  MemCache transfer error.

Check the server after the verifier passes:

```bash
curl -f http://127.0.0.1:30000/health_generate
grep -nE '507014|507015|AICore|DeepEP.*timeout|Scheduler.*exit|Traceback' \
  "$RESULT_DIR/server/server.log"
```

An empty `grep` result is expected. If DeepEP fails, preserve the exact DeepEP
commit, complete server log, request phase, and device health output. Do not
classify the error as an L3 failure until the same request sequence is compared
with L3 disabled on the same software stack.

## Run the L3 output accuracy test

This test uses the included `bench_ids_dsv4.py`. It generates
deterministic token-id inputs, saves the populate output IDs, waits for
write-through to finish, flushes L1/L2 while retaining MemCache L3, replays the
same inputs, and compares every generated token. Unlike a performance run,
`--skip-measure` is intentional: only populate and correctness replay are run.

Use the established test-machine layout:

```bash
export TOOLS_DIR=/home/l00951280/code/sglang-tools/scripts/dsv4_l3_memcache
export SGLANG_DIR=/home/l00951280/code/sglang
export MODEL_PATH=/home/weights/DeepSeek-V4-Flash-w8a8-mtp
export SERVER_LOG=/home/l00951280/dsv4-l3-results/<server-run>/server/server.log
```

Confirm that the tools checkout contains the accuracy driver:

```bash
test -f "$TOOLS_DIR/bench_ids_dsv4.py"
```

The established server topology is DP=16. Run the 128K accuracy case with 32
total requests, 16 concurrent requests per wave, and 32 generated tokens per
request:

```bash
DP_RANKS=16 bash "$TOOLS_DIR/test_l3_accuracy.sh" 131072 32 100
```

For staged troubleshooting, keep the same 32 requests and DP=16 while reducing
only the input length:

```bash
DP_RANKS=16 bash "$TOOLS_DIR/test_l3_accuracy.sh" 8192 32 100
DP_RANKS=16 bash "$TOOLS_DIR/test_l3_accuracy.sh" 32768 32 100
DP_RANKS=16 bash "$TOOLS_DIR/test_l3_accuracy.sh" 65536 32 100
```

The test passes only when all replay outputs are token-for-token identical and
the server records a positive cache hit after L1/L2 were flushed. Artifacts are
stored under `/home/l00951280/dsv4-l3-results/l3-accuracy` by default. A
passing summary resembles:

```text
PASS
identical_outputs=32/32
post_flush_cached_token_sum=...
```

The upstream driver's `50` mode stores and replays the first half of each
prefix. It validates the accuracy of that half-length L3 object; it does not
compare the final output of a full-length request whose first half is cached.

## Stop the processes

Stop only the processes recorded by this run. An NPU reset is not part of the
normal test lifecycle.

```bash
kill -TERM "$(cat "$RESULT_DIR/server/launch.pid")"
kill -TERM "$(cat "$RESULT_DIR/holder.pid")"
kill -TERM "$(cat "$RESULT_DIR/meta.pid")"
```

Keep `warm128-two-cohorts.json`, all three logs, both commit files, the DeepEP
package information, and device-health output together as the test artifact.
