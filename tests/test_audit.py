"""Tests for enterprise audit event contracts and sinks."""

import json

import pytest

from langdeep import (
    AuditEvent,
    InMemoryAuditSink,
    JsonlAuditSink,
    make_audit_event,
    redact_audit_payload,
)
from langdeep.core.audit import AUDIT_SCHEMA_VERSION, REDACTED, normalize_audit_event
from langdeep.core.errors import ConfigurationError


def test_audit_event_defaults_and_round_trip():
    event = AuditEvent(
        event_type="tool.call",
        outcome="success",
        request_id="req-1",
        trace_id="trace-1",
        tenant_id="tenant-a",
        user_id="user-a",
        session_id="session-a",
        process_id="process-a",
        workflow_id="workflow-a",
        task_id="task-a",
        policy_result="allowed",
        attributes={"tool": "search"},
    )

    payload = event.as_dict()

    assert payload["schema_version"] == AUDIT_SCHEMA_VERSION
    assert payload["event_id"]
    assert payload["timestamp"]
    assert payload["event_type"] == "tool.call"
    assert payload["attributes"] == {"tool": "search"}
    assert AuditEvent.from_dict(payload).as_dict() == payload


def test_audit_event_rejects_missing_event_type_and_unknown_schema():
    with pytest.raises(ConfigurationError):
        AuditEvent(event_type="")

    with pytest.raises(ConfigurationError):
        AuditEvent(event_type="tool.call", schema_version="unknown")


def test_make_audit_event_factory():
    event = make_audit_event("workflow.plan", outcome="success", request_id="req-1")

    assert event.event_type == "workflow.plan"
    assert event.outcome == "success"
    assert event.request_id == "req-1"


def test_redact_audit_payload_recurses_sensitive_keys():
    payload = {
        "api_key": "sk-test",
        "nested": {
            "authorization": "Bearer token",
            "safe": "visible",
            "items": [{"password": "secret"}, {"name": "ok"}],
        },
    }

    redacted = redact_audit_payload(payload)

    assert redacted["api_key"] == REDACTED
    assert redacted["nested"]["authorization"] == REDACTED
    assert redacted["nested"]["safe"] == "visible"
    assert redacted["nested"]["items"][0]["password"] == REDACTED
    assert payload["api_key"] == "sk-test"


def test_in_memory_audit_sink_records_redacted_events_and_limits():
    sink = InMemoryAuditSink(max_events=2)

    first = sink.record(
        {
            "event_type": "tool.call",
            "outcome": "success",
            "attributes": {"token": "secret", "visible": "ok"},
        }
    )
    second = sink.record(AuditEvent(event_type="agent.route", outcome="success"))
    third = sink.record(AuditEvent(event_type="workflow.done", outcome="success"))

    events = sink.list_events()

    assert first.attributes["token"] == REDACTED
    assert [event.event_type for event in events] == [second.event_type, third.event_type]

    sink.clear()
    assert sink.list_events() == []


def test_jsonl_audit_sink_writes_and_reads_events(tmp_path):
    path = tmp_path / "audit" / "events.jsonl"
    sink = JsonlAuditSink(path)

    event = sink.record(
        AuditEvent(
            event_type="skill.activate",
            outcome="success",
            tenant_id="tenant-a",
            attributes={"secret": "hidden", "skill": "invoice"},
        )
    )

    assert path.exists()
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert raw[0]["event_id"] == event.event_id
    assert raw[0]["attributes"]["secret"] == REDACTED
    assert sink.list_events()[0].attributes["skill"] == "invoice"

    sink.clear()
    assert path.read_text(encoding="utf-8") == ""


def test_normalize_audit_event_can_disable_redaction():
    event = normalize_audit_event(
        {"event_type": "secret.read", "attributes": {"api_key": "visible"}},
        redact=False,
    )

    assert event.attributes["api_key"] == "visible"
