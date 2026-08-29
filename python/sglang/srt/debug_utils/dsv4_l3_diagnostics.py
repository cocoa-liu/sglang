"""Opt-in diagnostics for DSV4 + HiCache L3 startup and forward stalls."""

from __future__ import annotations

import contextlib
import contextvars
import faulthandler
import functools
import importlib.metadata
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from sglang.srt.environ import envs

_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "dsv4_l3_diagnostic_context", default=None
)
_once_lock = threading.Lock()
_once_keys: set[str] = set()


@functools.lru_cache(maxsize=1)
def diagnostics_enabled() -> bool:
    """Cache the opt-in switch so disabled serving has negligible overhead."""
    return envs.SGLANG_DSV4_L3_DIAGNOSTICS.get()


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        values = [_safe_value(item) for item in value[:16]]
        if len(value) > 16:
            values.append(f"...+{len(value) - 16}")
        return values
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in list(value.items())[:16]
        }
    return str(value)


def tensor_metadata(tensor: Any) -> dict[str, Any]:
    """Return shape-only metadata without reading or synchronizing tensor data."""
    if tensor is None:
        return {"tensor": None}
    if isinstance(tensor, tuple):
        return {"tensors": [tensor_metadata(item) for item in tensor]}
    return {
        "shape": list(getattr(tensor, "shape", ())),
        "dtype": str(getattr(tensor, "dtype", None)),
        "device": str(getattr(tensor, "device", None)),
        "numel": int(tensor.numel()) if hasattr(tensor, "numel") else None,
    }


def request_metadata(req: Any) -> dict[str, Any]:
    input_ids = getattr(req, "input_ids", None)
    try:
        input_tokens = len(input_ids) if input_ids is not None else None
    except TypeError:
        input_tokens = None
    return {
        "request_type": type(req).__name__,
        "rid": getattr(req, "rid", None),
        "input_tokens": input_tokens,
        "routed_dp_rank": getattr(req, "routed_dp_rank", None),
    }


def batch_metadata(batch: Any) -> dict[str, Any]:
    reqs = list(getattr(batch, "reqs", None) or [])
    return {
        "forward_iter": getattr(batch, "forward_iter", None),
        "forward_mode": str(getattr(batch, "forward_mode", None)),
        "request_count": len(reqs),
        "rids": [getattr(req, "rid", None) for req in reqs],
        "input_ids": tensor_metadata(getattr(batch, "input_ids", None)),
        "seq_lens_sum": getattr(batch, "seq_lens_sum", None),
    }


def forward_batch_metadata(forward_batch: Any) -> dict[str, Any]:
    return {
        "forward_mode": str(getattr(forward_batch, "forward_mode", None)),
        "batch_size": getattr(forward_batch, "batch_size", None),
        "input_ids": tensor_metadata(getattr(forward_batch, "input_ids", None)),
        "seq_lens_sum": getattr(forward_batch, "seq_lens_sum", None),
    }


def diagnostic_log(
    logger: logging.Logger,
    event: str,
    /,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    if not diagnostics_enabled():
        return
    record = {
        "event": event,
        "monotonic_s": round(time.monotonic(), 6),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        **(_context.get() or {}),
        **fields,
    }
    logger.log(
        level,
        "[DSV4_L3_DIAG] %s",
        json.dumps(_safe_value(record), sort_keys=True, ensure_ascii=False),
    )
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:  # noqa: BLE001 - diagnostics must not break serving
            continue


def diagnostic_log_once(
    logger: logging.Logger, key: str, event: str, /, **fields: Any
) -> None:
    if not diagnostics_enabled():
        return
    process_key = f"{os.getpid()}:{key}"
    with _once_lock:
        if process_key in _once_keys:
            return
        _once_keys.add(process_key)
    diagnostic_log(logger, event, **fields)


@contextlib.contextmanager
def diagnostic_context(**fields: Any) -> Iterator[None]:
    if not diagnostics_enabled():
        yield
        return
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


@contextlib.contextmanager
def diagnostic_operation(
    logger: logging.Logger,
    event: str,
    /,
    *,
    watchdog: bool = False,
    **fields: Any,
) -> Iterator[None]:
    if not diagnostics_enabled():
        yield
        return

    started = time.monotonic()
    done = threading.Event()
    diagnostic_log(logger, f"{event}.begin", **fields)

    if watchdog:
        timeout = envs.SGLANG_DSV4_L3_DIAGNOSTICS_STALL_TIMEOUT.get()

        def dump_if_stalled() -> None:
            if timeout <= 0 or done.wait(timeout):
                return
            diagnostic_log(
                logger,
                f"{event}.stall",
                level=logging.ERROR,
                elapsed_s=round(time.monotonic() - started, 3),
                timeout_s=timeout,
                **fields,
            )
            try:
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                sys.stderr.flush()
            except Exception:  # noqa: BLE001 - best-effort failure diagnostics
                logger.exception("Failed to dump DSV4 L3 diagnostic thread stacks")

        threading.Thread(
            target=dump_if_stalled,
            name=f"dsv4-diag-{event}",
            daemon=True,
        ).start()

    try:
        yield
    except BaseException as exc:
        diagnostic_log(
            logger,
            f"{event}.error",
            level=logging.ERROR,
            elapsed_s=round(time.monotonic() - started, 6),
            error_type=type(exc).__name__,
            error=str(exc),
            **fields,
        )
        raise
    else:
        diagnostic_log(
            logger,
            f"{event}.end",
            elapsed_s=round(time.monotonic() - started, 6),
            **fields,
        )
    finally:
        done.set()


@contextlib.contextmanager
def diagnostic_layer(
    logger: logging.Logger, layer_id: int, hidden_states: Any
) -> Iterator[None]:
    with diagnostic_context(layer_id=layer_id), diagnostic_operation(
        logger,
        "model.layer",
        layer_id=layer_id,
        hidden_states=tensor_metadata(hidden_states),
    ):
        yield
    diagnostic_device_sync(logger, f"model.layer.{layer_id}.after")


def diagnostic_method(
    event: str,
    *,
    argument_index: int,
    argument_name: str,
    metadata_fn: Callable[[Any], dict[str, Any]],
    watchdog: bool = False,
):
    """Decorate a bound method with an opt-in diagnostic scope."""

    def decorate(func):
        method_logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if not diagnostics_enabled():
                return func(*args, **kwargs)
            argument = (
                args[argument_index]
                if len(args) > argument_index
                else kwargs[argument_name]
            )
            metadata = metadata_fn(argument)
            with diagnostic_context(**metadata), diagnostic_operation(
                method_logger, event, watchdog=watchdog, **metadata
            ):
                return func(*args, **kwargs)

        return wrapped

    return decorate


def diagnostic_runtime_snapshot(logger: logging.Logger, component: str) -> None:
    if not diagnostics_enabled():
        return

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    deep_ep_module = sys.modules.get("deep_ep")
    diagnostic_log_once(
        logger,
        "runtime_snapshot",
        "runtime.snapshot",
        component=component,
        python=sys.version.split()[0],
        deep_ep_version=package_version("deep_ep"),
        deep_ep_file=getattr(deep_ep_module, "__file__", None),
        torch_version=package_version("torch"),
        torch_npu_version=package_version("torch_npu"),
        sgl_kernel_npu_version=package_version("sgl_kernel_npu"),
        ascend_home_path=os.getenv("ASCEND_HOME_PATH"),
        ascend_opp_path=os.getenv("ASCEND_OPP_PATH"),
        ld_library_path=os.getenv("LD_LIBRARY_PATH"),
    )


def diagnostic_device_sync(logger: logging.Logger, boundary: str) -> None:
    """Synchronize only in the explicitly requested second-level probe."""
    if not diagnostics_enabled() or not envs.SGLANG_DSV4_L3_DIAGNOSTICS_SYNC.get():
        return
    import torch

    with diagnostic_operation(
        logger, "device.synchronize", watchdog=True, boundary=boundary
    ):
        torch.get_device_module().synchronize()
