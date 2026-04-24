"""Centralized structured logging with trace-id propagation.

Usage:
    from langdeep.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Agent invoked", extra={"agent_name": "web_search", "task_id": "t1"})

Trace IDs:
    A trace_id is generated per orchestrator invocation and stored in a
    context variable so that all log entries for the same request are
    correlated without threading through every function signature.
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

# ── Trace-ID context variable ────────────────────────────────────────────────────
_trace_id: ContextVar[Optional[str]] = ContextVar("langdeep_trace_id", default=None)
_request_id: ContextVar[Optional[str]] = ContextVar("langdeep_request_id", default=None)


def set_trace_context(trace_id: Optional[str] = None) -> str:
    """Set the trace-id for the current async/sync context. Returns the id."""
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    _request_id.set(uuid.uuid4().hex[:8])
    return tid


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def get_request_id() -> Optional[str]:
    return _request_id.get()


def clear_trace_context() -> None:
    _trace_id.set(None)
    _request_id.set(None)


# ── Structured formatter ─────────────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """Emits log records as key=value lines with trace/request context."""

    def format(self, record: logging.LogRecord) -> str:
        extras: dict = getattr(record, "__dict__", {})

        parts = [
            f"level={record.levelname}",
            f"logger={record.name}",
        ]

        tid = _trace_id.get()
        if tid:
            parts.append(f"trace_id={tid}")

        rid = _request_id.get()
        if rid:
            parts.append(f"req_id={rid}")

        # Include any extra keys passed via logger.info(..., extra={...})
        standard = {
            "name", "msg", "args", "levelno", "levelname", "pathname",
            "filename", "module", "lineno", "funcName", "created", "msecs",
            "relativeCreated", "thread", "threadName", "process",
            "message", "exc_info", "exc_text", "stack_info", "taskName",
        }
        for key, value in sorted(extras.items()):
            if key not in standard and not key.startswith("_"):
                parts.append(f"{key}={value}")

        parts.append(f"msg=\"{record.getMessage()}\"")

        if record.exc_info and record.exc_info[1]:
            parts.append(f"exception={record.exc_info[1]!r}")

        return " ".join(parts)


# ── Factory ──────────────────────────────────────────────────────────────────────

_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger("langdeep")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the structured formatter configured."""
    _ensure_initialized()
    return logging.getLogger(name)


def configure(level: int = logging.INFO, handler: Optional[logging.Handler] = None) -> None:
    """Replace the default handler with a custom one at the given level."""
    root = logging.getLogger("langdeep")
    root.handlers.clear()
    h = handler or logging.StreamHandler()
    if not h.formatter:
        h.setFormatter(StructuredFormatter())
    root.addHandler(h)
    root.setLevel(level)
