#!/usr/bin/env python3
import argparse
import json
import time

import requests


def post(base_url, path):
    response = requests.post(base_url + path, timeout=900)
    response.raise_for_status()
    print(path, response.text, flush=True)


def generate(base_url, identity):
    length = 131072
    cached_identity = identity % 1000
    tokens = [1000 + cached_identity]
    tokens.extend(
        2000 + ((i // 128 + cached_identity * 17) % 1000)
        for i in range(1, length)
    )
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": tokens,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "routed_dp_rank": 0,
        },
        timeout=1800,
    )
    response.raise_for_status()
    meta = response.json().get("meta_info", {})
    result = {
        "identity": identity,
        "elapsed_s": time.perf_counter() - started,
        "cached_tokens": meta.get("cached_tokens"),
        "cached_tokens_details": meta.get("cached_tokens_details"),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    args = parser.parse_args()

    post(args.base_url, "/hicache/storage-backend/clear")
    post(args.base_url, "/flush_cache")
    for identity in range(61000, 61004):
        generate(args.base_url, identity)
        time.sleep(10)
        post(args.base_url, "/flush_cache")
    time.sleep(60)
    post(args.base_url, "/flush_cache")
    time.sleep(30)
    result = generate(args.base_url, 61000)
    storage_tokens = (result["cached_tokens_details"] or {}).get("storage", 0) or 0
    if storage_tokens != 129024:
        raise RuntimeError(
            f"expected 129024 L3 hit tokens for the 128K probe, got {storage_tokens}"
        )
    print("PASS: 128K L3 smoke hit 129024/131072 tokens", flush=True)


if __name__ == "__main__":
    main()
