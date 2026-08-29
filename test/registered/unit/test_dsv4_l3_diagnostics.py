import logging
import time
from types import SimpleNamespace

import pytest

from sglang.srt.debug_utils import dsv4_l3_diagnostics as diag


@pytest.fixture(autouse=True)
def reset_diagnostics(monkeypatch):
    monkeypatch.delenv("SGLANG_DSV4_L3_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("SGLANG_DSV4_L3_DIAGNOSTICS_STALL_TIMEOUT", raising=False)
    diag.diagnostics_enabled.cache_clear()
    yield
    diag.diagnostics_enabled.cache_clear()


def _enable(monkeypatch, *, stall_timeout=None):
    monkeypatch.setenv("SGLANG_DSV4_L3_DIAGNOSTICS", "1")
    if stall_timeout is not None:
        monkeypatch.setenv(
            "SGLANG_DSV4_L3_DIAGNOSTICS_STALL_TIMEOUT", str(stall_timeout)
        )
    diag.diagnostics_enabled.cache_clear()


def test_diagnostic_log_is_disabled_by_default(caplog):
    with caplog.at_level(logging.INFO):
        diag.diagnostic_log(logging.getLogger(__name__), "disabled.event", value=1)

    assert "DSV4_L3_DIAG" not in caplog.text


def test_operation_emits_context_and_balanced_boundaries(monkeypatch, caplog):
    _enable(monkeypatch)

    with (
        caplog.at_level(logging.INFO),
        diag.diagnostic_context(rid="request-1", layer_id=7),
        diag.diagnostic_operation(
            logging.getLogger(__name__), "test.operation", objects=3
        ),
    ):
        pass

    assert '"event": "test.operation.begin"' in caplog.text
    assert '"event": "test.operation.end"' in caplog.text
    assert '"rid": "request-1"' in caplog.text
    assert '"layer_id": 7' in caplog.text
    assert '"objects": 3' in caplog.text


def test_operation_logs_exception_and_preserves_it(monkeypatch, caplog):
    _enable(monkeypatch)

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(ValueError, match="broken"),
        diag.diagnostic_operation(logging.getLogger(__name__), "test.failure"),
    ):
        raise ValueError("broken")

    assert '"event": "test.failure.error"' in caplog.text
    assert '"error_type": "ValueError"' in caplog.text


def test_watchdog_dumps_python_threads(monkeypatch, caplog):
    _enable(monkeypatch, stall_timeout=0.01)
    dumped = []
    monkeypatch.setattr(
        diag.faulthandler,
        "dump_traceback",
        lambda **kwargs: dumped.append(kwargs),
    )

    with caplog.at_level(logging.INFO), diag.diagnostic_operation(
        logging.getLogger(__name__), "test.stall", watchdog=True
    ):
        time.sleep(0.05)

    assert dumped and dumped[0]["all_threads"] is True
    assert '"event": "test.stall.stall"' in caplog.text


def test_watchdog_accepts_operation_timeout_field(monkeypatch, caplog):
    _enable(monkeypatch, stall_timeout=0.01)
    monkeypatch.setattr(diag.faulthandler, "dump_traceback", lambda **kwargs: None)

    with caplog.at_level(logging.INFO), diag.diagnostic_operation(
        logging.getLogger(__name__),
        "test.timeout_field",
        watchdog=True,
        timeout_s=600,
    ):
        time.sleep(0.05)

    assert '"event": "test.timeout_field.stall"' in caplog.text
    assert '"timeout_s": 600' in caplog.text
    assert '"watchdog_timeout_s": 0.01' in caplog.text


def test_metadata_helpers_do_not_read_tensor_values():
    tensor = SimpleNamespace(
        shape=(2, 4),
        dtype="int8",
        device="npu:3",
        numel=lambda: 8,
    )
    batch = SimpleNamespace(
        forward_mode="EXTEND",
        batch_size=1,
        input_ids=tensor,
        seq_lens_sum=8,
    )

    assert diag.tensor_metadata(tensor) == {
        "shape": [2, 4],
        "dtype": "int8",
        "device": "npu:3",
        "numel": 8,
    }
    assert diag.forward_batch_metadata(batch)["input_ids"]["numel"] == 8


def test_diagnostic_method_accepts_keyword_argument(monkeypatch, caplog):
    _enable(monkeypatch)

    class Worker:
        @diag.diagnostic_method(
            "test.method",
            argument_index=1,
            argument_name="batch",
            metadata_fn=lambda batch: {"batch_id": batch},
        )
        def run(self, batch):
            return batch

    with caplog.at_level(logging.INFO):
        assert Worker().run(batch="keyword-batch") == "keyword-batch"

    assert '"event": "test.method.end"' in caplog.text
    assert '"batch_id": "keyword-batch"' in caplog.text
