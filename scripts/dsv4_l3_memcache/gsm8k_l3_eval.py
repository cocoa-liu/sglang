#!/usr/bin/env python3
"""Run deterministic GSM8K through /generate and record cache-tier evidence."""

import argparse
import json
import threading
import time
from pathlib import Path

from sglang.test.simple_eval_common import GenerateSampler
from sglang.test.simple_eval_mixed_prefix_gsm8k import GSM8KEval


class RecordingGenerateSampler(GenerateSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stats_lock = threading.Lock()
        self.cached_tokens = []
        self.storage_cached_tokens = []
        self.failed_requests = 0

    def __call__(self, message_list):
        prompt = "\n".join(
            message["content"]
            for message in message_list
            if isinstance(message.get("content"), str)
        )
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_tokens,
                "stop": self.stop,
            },
            "stream": False,
        }

        for trial in range(6):
            try:
                response = self.client.post(self.generate_url, json=payload)
                if response.status_code == 400:
                    raise RuntimeError(f"bad request: {response.text[:500]}")
                response.raise_for_status()
                data = response.json()
                meta = data.get("meta_info") or {}
                details = meta.get("cached_tokens_details") or {}
                with self._stats_lock:
                    self._completion_tokens.append(
                        int(meta.get("completion_tokens") or 0)
                    )
                    self.cached_tokens.append(int(meta.get("cached_tokens") or 0))
                    self.storage_cached_tokens.append(
                        int(details.get("storage") or 0)
                    )
                return data.get("text") or ""
            except Exception as error:
                if trial == 5:
                    print(f"request failed after retries: {error!r}", flush=True)
                    with self._stats_lock:
                        self.failed_requests += 1
                    return ""
                delay = 2**trial
                print(
                    f"request retry {trial + 1}/6 after {delay}s: {error!r}",
                    flush=True,
                )
                time.sleep(delay)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--num-shots", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gsm8k-data-path")
    args = parser.parse_args()

    evaluator = GSM8KEval(
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        num_shots=args.num_shots,
        data_path=args.gsm8k_data_path,
    )
    sampler = RecordingGenerateSampler(
        base_url=f"{args.base_url.rstrip('/')}/v1",
        model=args.model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["Question", "Assistant:", "<|separator|>"],
    )

    started = time.perf_counter()
    result = evaluator(sampler)
    latency = time.perf_counter() - started
    payload = {
        "score": float(result.score),
        "metrics": result.metrics,
        "latency_s": latency,
        "num_examples": args.num_examples,
        "num_threads": args.num_threads,
        "num_shots": args.num_shots,
        "max_tokens": args.max_tokens,
        "successful_requests": len(sampler.cached_tokens),
        "failed_requests": sampler.failed_requests,
        "cached_tokens": sum(sampler.cached_tokens),
        "storage_cached_tokens": sum(sampler.storage_cached_tokens),
        "requests_with_cache": sum(value > 0 for value in sampler.cached_tokens),
        "requests_with_storage_cache": sum(
            value > 0 for value in sampler.storage_cached_tokens
        ),
        "completion_tokens": sum(sampler._completion_tokens),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print("GSM8K_RESULT " + json.dumps(payload, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
