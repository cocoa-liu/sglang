#!/usr/bin/env python3
import os
import signal
import sys
import time

from memcache_hybrid import DistributedObjectStore, LocalConfig


def main(capacity: str, device_id: int) -> None:
    config = LocalConfig()
    config.meta_service_url = os.environ.get(
        "MEMCACHE_META_URL", "tcp://127.0.0.1:5000"
    )
    config.config_store_url = os.environ.get(
        "MEMCACHE_CONFIG_STORE_URL", "tcp://127.0.0.1:6000"
    )
    config.protocol = os.environ.get("MEMCACHE_PROTOCOL", "device_sdma")
    config.hcom_url = os.environ.get("MEMCACHE_HCOM_URL", "tcp://127.0.0.1:7000")
    config.dram_size = capacity
    config.max_dram_size = os.environ.get("MEMCACHE_MAX_DRAM_SIZE", "600GB")
    config.world_size = int(os.environ.get("MEMCACHE_WORLD_SIZE", "17"))
    config.client_timeout_seconds = 600
    config.read_thread_pool_size = 64
    config.write_thread_pool_size = 32

    store = DistributedObjectStore()
    if store.setup(config) != 0:
        raise RuntimeError("DistributedObjectStore.setup failed")
    if store.init(device_id, True) != 0:
        raise RuntimeError("DistributedObjectStore.init failed")
    print(
        f"LOCAL_HOLDER_READY capacity={capacity} device_id={device_id}", flush=True
    )

    stopped = False

    def request_stop(*_args):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stopped:
        time.sleep(1)
    store.close()


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "224GB",
        int(sys.argv[2]) if len(sys.argv) > 2 else 0,
    )
