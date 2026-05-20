"""Tests for metadata-driven agent builders."""

import langgraph.prebuilt
from langchain_core.messages import AIMessage
from langchain_core.tools import tool as lc_tool

from langdeep.core.agent_builder import (
    AgentBuilderRegistry,
    BaseAgentBuilder,
    ReActAgentBuilder,
    validate_agent_runnable,
)
from langdeep.core.errors import AgentBuildError, PromptNotFoundError
from langdeep.core.registry.agent_registry import AgentMetadata
from langdeep.core.registry.model_registry import ModelConfig, model_registry
from langdeep.core.registry.tool_registry import ToolMetadata, tool_registry
from langdeep.core.tools import PolicyAwareTool

from conftest import SmartMockLLM, clean_registries


def setup_function():
    clean_registries()


def test_builder_registry_rejects_empty_agent_type():
    registry = AgentBuilderRegistry()
    try:
        registry.register("", ReActAgentBuilder())
        assert False, "Should raise"
    except AgentBuildError:
        pass


def test_builder_registry_unknown_type_raises():
    registry = AgentBuilderRegistry()
    meta = AgentMetadata(name="unknown", description="", agent_type="missing")
    try:
        registry.build(meta)
        assert False, "Should raise"
    except AgentBuildError as exc:
        assert "missing" in str(exc)


def test_builder_registry_uses_registered_builder():
    class TestBuilder(BaseAgentBuilder):
        def build(self, metadata):
            return {"built": metadata.name}

    registry = AgentBuilderRegistry()
    registry.register("test", TestBuilder())
    result = registry.build(AgentMetadata(name="agent1", description="", agent_type="test"))
    assert result == {"built": "agent1"}
    assert "test" in registry.list_builders()


def test_validate_agent_runnable_accepts_sync_async_or_both():
    class SyncOnly:
        def invoke(self, state):
            return state

    class AsyncOnly:
        async def ainvoke(self, state):
            return state

    class SyncAndAsync:
        def invoke(self, state):
            return state

        async def ainvoke(self, state):
            return state

    validate_agent_runnable(SyncOnly())
    validate_agent_runnable(AsyncOnly())
    validate_agent_runnable(SyncAndAsync())


def test_validate_agent_runnable_rejects_missing_call_contract():
    try:
        validate_agent_runnable(object())
        assert False, "Should raise"
    except AgentBuildError as exc:
        assert "invoke(state), ainvoke(state), or both" in str(exc)


def test_react_builder_resolves_inline_and_file_prompt(tmp_path):
    builder = ReActAgentBuilder()
    inline = AgentMetadata(name="a", description="", system_prompt="system text")
    assert builder._resolve_prompt(inline) == "system text"

    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("file prompt", encoding="utf-8")
    from_file = AgentMetadata(name="a", description="", prompt_path=str(prompt_file))
    assert builder._resolve_prompt(from_file) == "file prompt"

    missing = AgentMetadata(name="a", description="", prompt_path=str(tmp_path / "missing.md"))
    try:
        builder._resolve_prompt(missing)
        assert False, "Should raise"
    except PromptNotFoundError:
        pass


def test_react_builder_build_passes_model_tools_and_prompt(monkeypatch):
    model_registry.register("mock", ModelConfig(provider="mock", model_name="mock"))
    model_registry.set_model_instance("mock", SmartMockLLM(model_name="mock"))

    @lc_tool
    def sample_tool(query: str) -> str:
        """Sample tool."""
        return query

    tool_registry.register(sample_tool, ToolMetadata(name="sample_tool", description="sample"))

    calls = {}

    def fake_create_react_agent(**kwargs):
        calls.update(kwargs)

        class Agent:
            def invoke(self, state):
                return {"messages": [AIMessage(content="ok")]}

        return Agent()

    monkeypatch.setattr(langgraph.prebuilt, "create_react_agent", fake_create_react_agent)

    metadata = AgentMetadata(
        name="react_agent",
        description="",
        model_name="mock",
        tools=["sample_tool"],
        system_prompt="be helpful",
    )
    agent = ReActAgentBuilder().build(metadata)

    assert agent.invoke({})["messages"][0].content == "ok"
    assert calls["model"] is model_registry.get_model("mock")
    assert [tool.name for tool in calls["tools"]] == ["sample_tool"]
    assert isinstance(calls["tools"][0], PolicyAwareTool)
    assert calls["prompt"] == "be helpful"
