#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import time
from typing import Any

import requests


def prompt_tokens(length: int, identity: int, cached_length: int | None = None):
    # The first token makes every prompt's chained page hashes independent.
    cached_identity = identity % 1000
    tokens = [1000 + cached_identity]
    tokens.extend(2000 + ((i // 128 + cached_identity * 17) % 1000) for i in range(1, length))
    if cached_length is not None and cached_length < length:
        # Preserve exactly cached_length tokens, then diverge from the populated key.
        tokens[cached_length] = 5000 + cached_identity
        for i in range(cached_length + 1, length):
            tokens[i] = 6000 + ((i // 128 + cached_identity * 29) % 1000)
    return tokens


def request_body(
    length: int, identities: list[int], cached_length: int | None
) -> bytes:
    payload = {
        "input_ids": [
            prompt_tokens(length, identity, cached_length) for identity in identities
        ],
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def one_batch(
    base_url: str,
    length: int,
    identities: list[int],
    body: bytes,
):
    start = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        timeout=3600,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    outputs = response.json()
    if not isinstance(outputs, list) or len(outputs) != len(identities):
        raise RuntimeError(
            f"expected {len(identities)} batch outputs, got {type(outputs)} "
            f"with length {len(outputs) if isinstance(outputs, list) else 'n/a'}"
        )
    results = []
    for identity, output in zip(identities, outputs):
        meta = output.get("meta_info", {})
        details = meta.get("cached_tokens_details") or {}
        storage_tokens = details.get("storage", 0) or 0
        results.append(
            {
                "identity": identity,
                "dp_rank": None,
                "latency_s": elapsed,
                "server_e2e_s": meta.get("e2e_latency"),
                "cached_tokens": meta.get("cached_tokens", 0) or 0,
                "storage_tokens": storage_tokens,
                "cached_tokens_details": details,
                "prompt_tokens": meta.get("prompt_tokens", length),
                "finish_reason": meta.get("finish_reason"),
            }
        )
    return results


def percentile(values, p):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def merge_summaries(name: str, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge equal-workload cohorts without changing the measurement totals."""
    results = [item for summary in summaries for item in summary["results"]]
    wall = sum(summary["wall_s"] for summary in summaries)
    latencies = [item["latency_s"] for item in results]
    storage_tokens = sum(item["storage_tokens"] for item in results)
    length = summaries[0]["input_length"]
    count = len(results)
    summary = {
        "name": name,
        "input_length": length,
        "request_count": count,
        "concurrency": summaries[0]["concurrency"],
        "cached_length": summaries[0]["cached_length"],
        "wall_s": wall,
        "ttft_mean_ms": statistics.fmean(latencies) * 1000,
        "ttft_p50_ms": percentile(latencies, 0.50) * 1000,
        "ttft_p90_ms": percentile(latencies, 0.90) * 1000,
        "ttft_p99_ms": percentile(latencies, 0.99) * 1000,
        "input_tps": length * count / wall,
        "storage_hit_tokens": storage_tokens,
        "storage_hit_rate_percent": 100 * storage_tokens / (length * count),
        "results": results,
        "cohorts": len(summaries),
    }
    print(
        "MERGED_SUMMARY "
        + json.dumps({k: v for k, v in summary.items() if k != "results"}),
        flush=True,
    )
    return summary


def run_group(
    args,
    name: str,
    count: int,
    concurrency: int,
    length: int,
    identity_base: int,
    cached_length: int | None,
    route_base: int = 0,
):
    # Send all samples as one native SGLang batch. Independent 256K JSON
    # requests are parsed serially by the single tokenizer worker; direct DP
    # routing can therefore put DeepEP ranks on different collective steps.
    # The DP controller round-robins the batch evenly. Each scheduler has
    # max_running_requests=1, so count > 16 becomes equal per-rank queues while
    # the actual model concurrency remains DP=16.
    identities = list(range(identity_base, identity_base + count))
    waves = [(identities, request_body(length, identities, cached_length))]
    started = time.perf_counter()
    results = []
    for identities, body in waves:
        batch_results = one_batch(args.base_url, length, identities, body)
        for result in batch_results:
            results.append(result)
            print(
                f"{name} {len(results)}/{count} "
                f"latency={result['latency_s']:.3f}s storage={result['storage_tokens']}",
                flush=True,
            )
    wall = time.perf_counter() - started
    latencies = [item["latency_s"] for item in results]
    storage_tokens = sum(item["storage_tokens"] for item in results)
    summary = {
        "name": name,
        "input_length": length,
        "request_count": count,
        "concurrency": concurrency,
        "cached_length": cached_length,
        "wall_s": wall,
        "ttft_mean_ms": statistics.fmean(latencies) * 1000,
        "ttft_p50_ms": percentile(latencies, 0.50) * 1000,
        "ttft_p90_ms": percentile(latencies, 0.90) * 1000,
        "ttft_p99_ms": percentile(latencies, 0.99) * 1000,
        "input_tps": length * count / wall,
        "storage_hit_tokens": storage_tokens,
        "storage_hit_rate_percent": 100 * storage_tokens / (length * count),
        "results": sorted(results, key=lambda item: item["identity"]),
    }
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    return summary


def populate_in_batches(args, count: int, length: int, identity_base: int):
    batches = []
    for offset in range(0, count, 8):
        batch_count = min(8, count - offset)
        batches.append(
            run_group(
                args,
                f"populate_b{offset // 8}",
                batch_count,
                8,
                length,
                identity_base + offset,
                None,
                route_base=offset,
            )
        )
        # Keep the completed write-through objects in L3, while freeing the
        # per-DP L1/L2 pages before the next 128K/256K populate request.
        time.sleep(10)
        flush_when_idle(args.base_url)
    return batches


def post(base_url: str, path: str, timeout: int = 900):
    response = requests.post(base_url + path, timeout=timeout)
    response.raise_for_status()
    print(f"POST {path}: {response.text.strip()}", flush=True)


def flush_when_idle(base_url: str, timeout: int = 900):
    """Wait for asynchronous write-through work before clearing a cohort.

    A generation response can arrive before every DP rank has acknowledged its
    L2->L3 backup. Clearing storage during that interval invalidates in-flight
    memcache puts, so retry only the non-destructive L1/L2 flush until all ranks
    report idle.
    """
    deadline = time.monotonic() + timeout
    while True:
        response = requests.post(base_url + "/flush_cache", timeout=timeout)
        if response.ok:
            print(f"POST /flush_cache: {response.text.strip()}", flush=True)
            return
        if response.status_code != 400 or time.monotonic() >= deadline:
            response.raise_for_status()
        print("POST /flush_cache: pending write-through; retrying", flush=True)
        time.sleep(5)


def initialize_memcache(args):
    # device_sdma is a 16-rank distributed service. With lazy initialization,
    # all ranks must enter init once before smaller populate batches are used.
    # Use one explicitly routed request per rank here.  A native batch is a
    # measurement primitive, but it is not a reliable rank-initialization
    # primitive because the DP controller may keep the whole batch on one rank.
    def initialize_rank(dp_rank: int):
        identity = 90000 + dp_rank
        response = requests.post(
            args.base_url + "/generate",
            json={
                "input_ids": prompt_tokens(2048, identity),
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "routed_dp_rank": dp_rank,
            },
            timeout=1800,
        )
        response.raise_for_status()
        print(f"memcache_init DP{dp_rank}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(initialize_rank, range(16)))
    time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase",
        choices=[
            "cold256",
            "warm128",
            "warm256",
            "populate128",
            "hit100_128",
            "hit50_128",
            "populate256",
            "hit100_256",
            "hit50_256",
        ],
        required=True,
    )
    parser.add_argument(
        "--skip-initialize",
        action="store_true",
        help="Skip the separate 16-rank lazy-init probe when the measured wave already uses all ranks.",
    )
    parser.add_argument("--identity-offset", type=int, default=0)
    parser.add_argument("--count-override", type=int)
    parser.add_argument(
        "--cohort-size",
        type=int,
        help="Split warm phases into capacity-bounded cohorts; concurrency is unchanged.",
    )
    parser.add_argument("--clear-storage", action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output: dict[str, Any] = {"phase": args.phase, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    split_phase = args.phase.startswith("populate") or args.phase.startswith("hit")
    if not args.skip_initialize and not split_phase:
        initialize_memcache(args)

    if args.phase == "cold256":
        post(args.base_url, "/flush_cache")
        output["cold256"] = run_group(args, "cold256", 16, 16, 262144, 80000, None)
    elif args.phase in ("warm128", "warm256"):
        length = 131072 if args.phase == "warm128" else 262144
        count = args.count_override or (64 if args.phase == "warm128" else 32)
        identity_base = 60000 + args.identity_offset
        cohort_size = args.cohort_size or count
        if cohort_size > count or count % cohort_size:
            raise ValueError("cohort-size must divide the warm-phase request count")
        populate_batches = []
        hit100_cohorts = []
        hit50_cohorts = []
        for offset in range(0, count, cohort_size):
            cohort_base = identity_base + offset
            print(f"Starting cohort {offset // cohort_size}", flush=True)
            flush_when_idle(args.base_url)
            post(args.base_url, "/hicache/storage-backend/clear")
            post(args.base_url, "/flush_cache")
            populate_batches.extend(
                populate_in_batches(args, cohort_size, length, cohort_base)
            )
            print("Waiting 60 seconds for write-through completion", flush=True)
            time.sleep(60)
            post(args.base_url, "/flush_cache")
            print("Waiting 30 seconds for flush-triggered writes", flush=True)
            time.sleep(30)
            hit100_cohorts.append(
                run_group(
                    args,
                    f"hit100_c{offset // cohort_size}",
                    cohort_size,
                    16,
                    length,
                    cohort_base,
                    None,
                )
            )
            post(args.base_url, "/flush_cache")
            time.sleep(30)
            hit50_cohorts.append(
                run_group(
                    args,
                    f"hit50_c{offset // cohort_size}",
                    cohort_size,
                    16,
                    length,
                    cohort_base,
                    length // 2,
                )
            )
            flush_when_idle(args.base_url)
        output["populate_batches"] = populate_batches
        output["hit100"] = merge_summaries("hit100", hit100_cohorts)
        output["hit50"] = merge_summaries("hit50", hit50_cohorts)
    else:
        length = 131072 if args.phase.endswith("128") else 262144
        count = args.count_override or (64 if length == 131072 else 32)
        identity_base = 60000 + args.identity_offset
        if args.phase.startswith("populate"):
            if args.clear_storage:
                post(args.base_url, "/hicache/storage-backend/clear")
            post(args.base_url, "/flush_cache")
            output["populate"] = run_group(
                args, "populate", count, 16, length, identity_base, None
            )
            print("Waiting 120 seconds for write-through completion", flush=True)
            time.sleep(120)
        else:
            cached_length = None if args.phase.startswith("hit100") else length // 2
            name = "hit100" if cached_length is None else "hit50"
            output[name] = run_group(
                args, name, count, 16, length, identity_base, cached_length
            )

    output["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
