"""Tests for runtime diagnostics and enterprise preflight checks."""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool as lc_tool

from langdeep import (
    DiagnosticIssue,
    InMemoryAuditSink,
    JsonlAuditSink,
    RuntimeDiagnostics,
    RuntimeValidator,
    build_doctor_report,
    register_provider,
    validate_runtime,
)
from langdeep.core.errors import ConfigurationError
from langdeep.core.registry.agent_registry import AgentMetadata, agent_registry
from langdeep.core.registry.model_registry import ModelConfig, model_registry, provider_registry
from langdeep.core.registry.tool_registry import ToolMetadata, tool_registry

from conftest import clean_registries


def setup_function():
    clean_registries()
    provider_registry.reset()


def _register_valid_tool():
    @lc_tool
    def lookup(query: str) -> str:
        """Lookup a value."""
        return query

    tool_registry.register(lookup, ToolMetadata(name="lookup", description="Lookup tool"))


def _register_valid_agent():
    class Agent:
        def invoke(self, state):
            return {"messages": [AIMessage(content="ok")]}

    agent_registry.register(
        "assistant",
        lambda: Agent(),
        AgentMetadata(
            name="assistant",
            description="Production assistant",
            model_name="gpt4o",
            tools=["lookup"],
        ),
    )


def test_runtime_diagnostics_pass_for_consistent_registries():
    model_registry.register("gpt4o", ModelConfig(provider="mock", model_name="gpt4o"))
    _register_valid_tool()
    _register_valid_agent()

    diagnostics = validate_runtime()

    assert diagnostics.ok is True
    assert diagnostics.error_count == 0
    assert diagnostics.to_dict()["ok"] is True


def test_runtime_diagnostics_report_model_agent_and_tool_errors():
    model_registry.register(
        "",
        ModelConfig(
            provider="",
            model_name="",
            temperature="hot",
            max_tokens=0,
        ),
    )
    model_registry.register(
        "bad_model",
        ModelConfig(provider="missing", model_name="bad", temperature=3),
    )
    agent_registry.register(
        "bad_agent",
        lambda: object(),
        AgentMetadata(
            name="bad_agent",
            description="",
            model_name="ghost_model",
            tools=["ghost_tool"],
        ),
    )

    diagnostics = RuntimeValidator().validate()

    assert diagnostics.ok is False
    assert diagnostics.error_count >= 6
    messages = [issue.message for issue in diagnostics.issues]
    assert any("provider" in message for message in messages)
    assert any("temperature must be numeric" in message for message in messages)
    assert any("max_tokens" in message for message in messages)
    assert any("agent model" in message for message in messages)
    assert any("tools" in message for message in messages)
    assert diagnostics.warning_count >= 1


def test_runtime_diagnostics_report_agent_and_tool_metadata_mismatch():
    class BareTool:
        name = "bare"
        description = ""

    tool_registry.register(BareTool(), ToolMetadata(name="other", description=""))
    agent_registry.register(
        "agent_key",
        lambda: object(),
        AgentMetadata(name="agent_meta", description="Agent", model_name="default"),
    )

    diagnostics = validate_runtime()
    messages = [issue.message for issue in diagnostics.issues]

    assert any("agent metadata name" in message for message in messages)
    assert any("tool metadata name" in message for message in messages)
    assert any("tool description is empty" in message for message in messages)


def test_runtime_diagnostics_noop_raise_when_ok():
    RuntimeDiagnostics().raise_for_errors()


def test_runtime_diagnostics_can_instantiate_agents():
    model_registry.register("gpt4o", ModelConfig(provider="mock", model_name="gpt4o"))
    agent_registry.register(
        "broken",
        lambda: object(),
        AgentMetadata(name="broken", description="Broken", model_name="gpt4o"),
    )

    diagnostics = validate_runtime(instantiate_agents=True)

    assert diagnostics.ok is False
    assert any(issue.message == "agent failed to instantiate" for issue in diagnostics.issues)


def test_runtime_diagnostics_raise_for_errors():
    diagnostics = RuntimeDiagnostics([
        DiagnosticIssue("error", "model", "m", "broken"),
    ])

    try:
        diagnostics.raise_for_errors()
        assert False, "Should raise"
    except ConfigurationError as exc:
        assert "Runtime diagnostics found configuration errors" in str(exc)


def test_model_registry_config_snapshot_is_copy():
    model_registry.register("gpt4o", ModelConfig(provider="mock", model_name="gpt4o"))

    config = model_registry.get_config("gpt4o")
    config.model_name = "mutated"

    assert model_registry.get_config("gpt4o").model_name == "gpt4o"


def test_model_registry_config_accessors_are_safe_snapshots():
    model_registry.register("gpt4o", ModelConfig(provider="mock", model_name="gpt4o"))

    configs = model_registry.list_model_configs()
    configs["gpt4o"].model_name = "mutated"

    assert model_registry.list_model_configs()["gpt4o"].model_name == "gpt4o"
    try:
        model_registry.get_config("missing")
        assert False, "Should raise"
    except Exception as exc:
        assert "MODEL_NOT_FOUND" in str(exc)


def test_doctor_report_includes_environment_dependencies_and_security():
    report = build_doctor_report()

    assert report["status"] in ("ok", "warning")
    assert report["environment"]["python_supported"] is True
    assert "langchain_core" in report["dependencies"]
    assert "sandbox" in report["registries"]
    assert report["audit"]["schema_version"] == "langdeep.audit.v1"
    assert report["audit"]["configured"] is False
    assert report["security"]["response_cache"]["type"] == "MemoryCache"
    assert report["warning_count"] >= 1


def test_doctor_report_strict_marks_warnings_not_ok():
    report = build_doctor_report(strict=True)

    assert report["warning_count"] >= 1
    assert report["ok"] is False


def test_doctor_report_surfaces_runtime_errors():
    agent_registry.register(
        "broken",
        lambda: object(),
        AgentMetadata(name="broken", description="Broken", model_name="missing_model"),
    )

    report = build_doctor_report()

    assert report["status"] == "error"
    assert report["error_count"] >= 1
    assert any(
        issue["component"] == "agent"
        for issue in report["diagnostics"]["issues"]
    )


def test_doctor_report_warns_on_hardcoded_api_key():
    model_registry.register(
        "hardcoded",
        ModelConfig(provider="mock", model_name="mock", api_key="sk-test-hardcoded"),
    )

    report = build_doctor_report()

    assert any(
        issue["component"] == "secrets"
        and issue["name"] == "model_api_key"
        for issue in report["issues"]
    )


def test_doctor_report_classifies_audit_sinks(tmp_path):
    in_memory = build_doctor_report(audit_sink=InMemoryAuditSink())
    assert in_memory["audit"]["configured"] is True
    assert in_memory["audit"]["durable"] is False
    assert any(issue["component"] == "audit" for issue in in_memory["audit"]["issues"])

    path = tmp_path / "audit" / "events.jsonl"
    jsonl = build_doctor_report(audit_sink=JsonlAuditSink(path))
    assert jsonl["audit"]["configured"] is True
    assert jsonl["audit"]["durable"] is True
    assert jsonl["audit"]["path"] == str(path)
    assert jsonl["audit"]["issues"] == []


def test_register_provider_function_returns_factory():
    def factory(config):
        return None

    returned = register_provider("custom_v2", factory)

    assert returned is factory
    assert provider_registry.get_provider("custom_v2") is factory
