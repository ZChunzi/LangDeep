"""Unit tests for structured logging and trace context."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from langdeep.core.logging import (
    set_trace_context, get_trace_id, get_request_id, clear_trace_context,
    get_logger, configure, StructuredFormatter,
)


def test_trace_context_lifecycle():
    clear_trace_context()
    assert get_trace_id() is None
    assert get_request_id() is None

    tid = set_trace_context("my-trace-123")
    assert tid == "my-trace-123"
    assert get_trace_id() == "my-trace-123"
    rid = get_request_id()
    assert rid is not None and len(rid) > 0

    clear_trace_context()
    assert get_trace_id() is None
    assert get_request_id() is None


def test_auto_trace_id():
    clear_trace_context()
    tid = set_trace_context()
    assert tid is not None
    assert len(tid) == 12
    assert get_trace_id() == tid
    clear_trace_context()


def test_get_logger():
    logger = get_logger("test.module")
    assert logger.name == "test.module"
    assert isinstance(logger, logging.Logger)


def test_logger_structured_output():
    logger = get_logger("test.structured")
    logger.info("hello", extra={"key": "value"})
    # No crash — manually verify output format via handler


def test_configure():
    import io
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    configure(level=logging.DEBUG, handler=handler)
    root_lg = logging.getLogger("langdeep")
    root_lg.info("configured")
    output = buf.getvalue()
    assert "configured" in output


def test_structured_formatter_with_trace():
    fmt = StructuredFormatter()
    record = logging.LogRecord("test", logging.INFO, "file.py", 42, "msg", (), None)
    result = fmt.format(record)
    assert "msg=\"msg\"" in result


def test_trace_in_log_output():
    clear_trace_context()
    set_trace_context("trace-456")
    fmt = StructuredFormatter()
    record = logging.LogRecord("test", logging.INFO, "file.py", 42, "traced msg", (), None)
    result = fmt.format(record)
    assert "trace-456" in result
    assert "traced msg" in result
    clear_trace_context()


def test_structured_formatter_redacts_sensitive_extras():
    fmt = StructuredFormatter()
    record = logging.LogRecord("test", logging.INFO, "file.py", 42, "msg", (), None)
    record.api_key = "sk-live-secret"
    record.password = "plain-password"
    record.user = "alice"

    result = fmt.format(record)

    assert "api_key=[REDACTED]" in result
    assert "password=[REDACTED]" in result
    assert "sk-live-secret" not in result
    assert "plain-password" not in result
    assert "user=alice" in result
