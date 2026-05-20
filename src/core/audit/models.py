"""Structured audit event contracts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from ..errors import ConfigurationError

AUDIT_SCHEMA_VERSION = "langdeep.audit.v1"


@dataclass
class AuditEvent:
    """Enterprise audit event for cross-boundary runtime activity."""

    event_type: str
    outcome: str = "unknown"
    request_id: str = ""
    trace_id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    session_id: str = ""
    process_id: str = ""
    workflow_id: str = ""
    task_id: str = ""
    policy_result: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _require_text(self.schema_version, "schema_version")
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported audit schema '{self.schema_version}'",
                context={"schema_version": self.schema_version, "supported": AUDIT_SCHEMA_VERSION},
            )
        self.event_type = _require_text(self.event_type, "event_type")
        self.outcome = self.outcome or "unknown"
        self.attributes = dict(self.attributes or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Build an event from a mapping."""
        if not isinstance(data, dict):
            raise ConfigurationError(
                "Audit event must be a mapping",
                context={"actual": type(data).__name__},
            )
        return cls(
            schema_version=data.get("schema_version", AUDIT_SCHEMA_VERSION),
            event_id=data.get("event_id") or uuid4().hex,
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            event_type=data.get("event_type", ""),
            outcome=data.get("outcome", "unknown"),
            request_id=data.get("request_id", ""),
            trace_id=data.get("trace_id", ""),
            tenant_id=data.get("tenant_id", ""),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            process_id=data.get("process_id", ""),
            workflow_id=data.get("workflow_id", ""),
            task_id=data.get("task_id", ""),
            policy_result=data.get("policy_result", ""),
            attributes=dict(data.get("attributes") or {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable audit event."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "outcome": self.outcome,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "process_id": self.process_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "policy_result": self.policy_result,
            "attributes": dict(self.attributes),
        }


def make_audit_event(event_type: str, **kwargs: Any) -> AuditEvent:
    """Convenience factory for structured audit events."""
    return AuditEvent(event_type=event_type, **kwargs)


def _require_text(value: Optional[str], field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Audit field '{field_name}' must be a non-empty string")
    return value.strip()
