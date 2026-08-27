#!/usr/bin/env python3
import argparse
import json
import math


def require_equal(actual, expected, name):
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected}, got {actual}")


def require_close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError(f"{name}: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="JSON written by bench_l3_matrix.py")
    args = parser.parse_args()

    with open(args.result, encoding="utf-8") as file:
        result = json.load(file)

    require_equal(result.get("phase"), "warm128", "phase")
    hit100 = result["hit100"]
    hit50 = result["hit50"]
    require_equal(hit100["cohorts"], 2, "hit100 cohorts")
    require_equal(hit50["cohorts"], 2, "hit50 cohorts")
    require_equal(hit100["request_count"], 32, "hit100 request count")
    require_equal(hit50["request_count"], 32, "hit50 request count")
    require_close(
        hit100["storage_hit_rate_percent"], 98.4375, "hit100 storage hit rate"
    )
    require_close(hit50["storage_hit_rate_percent"], 50.0, "hit50 storage hit rate")
    for item in hit100["results"]:
        require_equal(item["storage_tokens"], 129024, "hit100 storage tokens")
    for item in hit50["results"]:
        require_equal(item["storage_tokens"], 65536, "hit50 storage tokens")
    print("PASS: two 128K cohorts have the expected L3 hit coverage")


if __name__ == "__main__":
    main()
