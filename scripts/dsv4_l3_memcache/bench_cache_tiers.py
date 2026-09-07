#!/usr/bin/env python3
"""Compare DeepSeek-V4 cache-hit latency across device, host, and L3 tiers."""

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


def prompt_tokens(length: int, identity: int) -> list[int]:
    return [1000 + identity % 1000] + [
        2000 + ((index // 128 + identity * 17) % 1000)
        for index in range(1, length)
    ]


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(ratio * len(ordered)) - 1))
    return ordered[index]


def post(base_url: str, path: str, timeout: int = 900) -> None:
    response = requests.post(base_url + path, timeout=timeout)
    response.raise_for_status()


def flush_when_idle(base_url: str, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while True:
        response = requests.post(base_url + "/flush_cache?timeout=60", timeout=70)
        if response.ok:
            return
        if response.status_code != 400 or time.monotonic() >= deadline:
            response.raise_for_status()
        time.sleep(5)


def one_request(base_url: str, length: int, identity: int, dp_rank: int) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": prompt_tokens(length, identity),
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "routed_dp_rank": dp_rank,
        },
        timeout=3600,
    )
    latency = time.perf_counter() - started
    if not response.ok:
        raise RuntimeError(
            f"DP{dp_rank} request failed with HTTP {response.status_code}: "
            f"{response.text[:2000]}"
        )
    output = response.json()
    meta = (
        output[0].get("meta_info", {})
        if isinstance(output, list)
        else output.get("meta_info", {})
    )
    return {
        "dp_rank": dp_rank,
        "identity": identity,
        "latency_s": latency,
        "server_e2e_s": meta.get("e2e_latency"),
        "cached_tokens": int(meta.get("cached_tokens") or 0),
        "cached_tokens_details": meta.get("cached_tokens_details") or {},
    }


def run_wave(base_url: str, length: int, identities: list[int]) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(identities)) as pool:
        futures = [
            pool.submit(one_request, base_url, length, identity, rank)
            for rank, identity in enumerate(identities)
        ]
        results = [future.result() for future in futures]
    return {"wall_s": time.perf_counter() - started, "results": results}


def validate_tier(name: str, wave: dict[str, Any], expected: str) -> None:
    errors = []
    for item in wave["results"]:
        details = item["cached_tokens_details"]
        expected_tokens = int(details.get(expected) or 0)
        lower_tokens = {
            tier: int(details.get(tier) or 0)
            for tier in ("device", "host", "storage")
            if tier != expected
        }
        if expected_tokens <= 0 or any(lower_tokens.values()):
            errors.append(
                f"DP{item['dp_rank']} cached={item['cached_tokens']} details={details}"
            )
    if errors:
        raise RuntimeError(f"{name} did not hit only {expected}: " + "; ".join(errors))


def summarize(name: str, waves: list[dict[str, Any]], length: int) -> dict[str, Any]:
    results = [item for wave in waves for item in wave["results"]]
    latencies = [item["latency_s"] for item in results]
    server_latencies = [
        float(item["server_e2e_s"])
        for item in results
        if item["server_e2e_s"] is not None
    ]
    wall_s = sum(wave["wall_s"] for wave in waves)
    summary = {
        "tier": name,
        "request_count": len(results),
        "mean_latency_ms": statistics.fmean(latencies) * 1000,
        "p50_latency_ms": percentile(latencies, 0.50) * 1000,
        "p90_latency_ms": percentile(latencies, 0.90) * 1000,
        "mean_server_e2e_ms": statistics.fmean(server_latencies) * 1000,
        "input_tps": length * len(results) / wall_s,
        "cached_tokens": sum(item["cached_tokens"] for item in results),
        "wall_s": wall_s,
    }
    print("TIER_SUMMARY " + json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-length", type=int, default=49152)
    parser.add_argument("--dp-ranks", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--filler-count", type=int, default=1)
    parser.add_argument("--write-settle-seconds", type=int, default=30)
    args = parser.parse_args()

    requests.get(args.base_url + "/health_generate", timeout=900).raise_for_status()
    all_waves: dict[str, list[dict[str, Any]]] = {"l1": [], "l2": [], "l3": []}

    for round_index in range(args.rounds):
        base = 70000 + round_index * 100
        target = [base + rank for rank in range(args.dp_ranks)]
        fillers = [
            [base + (filler_index + 1) * 10 + rank for rank in range(args.dp_ranks)]
            for filler_index in range(args.filler_count)
        ]
        print(f"ROUND {round_index + 1}/{args.rounds}: reset and populate target", flush=True)
        flush_when_idle(args.base_url)
        post(args.base_url, "/hicache/storage-backend/clear")
        flush_when_idle(args.base_url)
        run_wave(args.base_url, args.input_length, target)

        l1 = run_wave(args.base_url, args.input_length, target)
        validate_tier("L1", l1, "device")
        all_waves["l1"].append(l1)

        print(f"ROUND {round_index + 1}/{args.rounds}: evict target to host", flush=True)
        for filler in fillers:
            run_wave(args.base_url, args.input_length, filler)
        l2 = run_wave(args.base_url, args.input_length, target)
        validate_tier("L2", l2, "host")
        all_waves["l2"].append(l2)

        print(f"ROUND {round_index + 1}/{args.rounds}: flush local tiers and load L3", flush=True)
        time.sleep(args.write_settle_seconds)
        flush_when_idle(args.base_url)
        l3 = run_wave(args.base_url, args.input_length, target)
        validate_tier("L3", l3, "storage")
        all_waves["l3"].append(l3)

    summaries = {
        tier: summarize(tier.upper(), waves, args.input_length)
        for tier, waves in all_waves.items()
    }
    l3 = summaries["l3"]
    comparisons = {}
    for tier in ("l1", "l2"):
        baseline = summaries[tier]
        comparisons[f"l3_vs_{tier}"] = {
            "latency_increase_percent":
                (l3["mean_latency_ms"] / baseline["mean_latency_ms"] - 1) * 100,
            "server_e2e_increase_percent":
                (l3["mean_server_e2e_ms"] / baseline["mean_server_e2e_ms"] - 1) * 100,
            "input_tps_decrease_percent":
                (1 - l3["input_tps"] / baseline["input_tps"]) * 100,
        }
    output = {
        "input_length": args.input_length,
        "dp_ranks": args.dp_ranks,
        "rounds": args.rounds,
        "summaries": summaries,
        "comparisons": comparisons,
        "waves": all_waves,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print("CACHE_TIER_COMPARISON " + json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
