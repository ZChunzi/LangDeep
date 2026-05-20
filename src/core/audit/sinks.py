"""Audit sink implementations."""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import AuditEvent
from .redaction import redact_audit_payload


class AuditSink:
    """Base audit sink interface."""

    def record(self, event: Union[AuditEvent, Dict[str, Any]]) -> AuditEvent:
        """Persist an audit event and return the normalized event."""
        raise NotImplementedError

    def list_events(self) -> List[AuditEvent]:
        """Return recorded events when supported by the sink."""
        return []

    def clear(self) -> None:
        """Clear recorded events when supported by the sink."""


class InMemoryAuditSink(AuditSink):
    """Thread-safe in-memory audit sink for tests and local development."""

    def __init__(self, *, redact: bool = True, max_events: Optional[int] = None):
        self._redact = redact
        self._max_events = max_events
        self._events: List[AuditEvent] = []
        self._lock = threading.RLock()

    def record(self, event: Union[AuditEvent, Dict[str, Any]]) -> AuditEvent:
        normalized = normalize_audit_event(event, redact=self._redact)
        with self._lock:
            self._events.append(normalized)
            if self._max_events is not None and len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
        return normalized

    def list_events(self) -> List[AuditEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class JsonlAuditSink(AuditSink):
    """Append-only JSON Lines audit sink."""

    def __init__(self, path: Union[str, Path], *, redact: bool = True):
        self._path = Path(path)
        self._redact = redact
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: Union[AuditEvent, Dict[str, Any]]) -> AuditEvent:
        normalized = normalize_audit_event(event, redact=self._redact)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(normalized.as_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        return normalized

    def list_events(self) -> List[AuditEvent]:
        if not self._path.exists():
            return []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        return [AuditEvent.from_dict(json.loads(line)) for line in lines if line.strip()]

    def clear(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._path.write_text("", encoding="utf-8")


def normalize_audit_event(
    event: Union[AuditEvent, Dict[str, Any]],
    *,
    redact: bool = True,
) -> AuditEvent:
    """Normalize a mapping or event instance and optionally redact attributes."""
    normalized = event if isinstance(event, AuditEvent) else AuditEvent.from_dict(event)
    if not redact:
        return normalized
    return AuditEvent.from_dict(
        {
            **normalized.as_dict(),
            "attributes": redact_audit_payload(normalized.attributes),
        }
    )
