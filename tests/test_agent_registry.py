"""Unit tests for AgentRegistry, AgentMetadata, and agent audit."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langdeep.core.registry.agent_registry import agent_registry, AgentRegistry, AgentMetadata
from langdeep.core.registry.tool_registry import tool_registry, ToolMetadata
from langdeep.core.agent_builder import BaseAgentBuilder, agent_builder_registry
from langdeep.core.errors import AgentBuildError, AgentNotFoundError, ConfigurationError
from langchain_core.tools import tool as lc_tool
from langchain_core.messages import AIMessage, HumanMessage

from conftest import clean_registries


def setup_function():
    clean_registries()


def test_singleton():
    r1 = AgentRegistry()
    r2 = AgentRegistry()
    assert r1 is r2


def test_register_and_get_agent():
    def factory():
        class Agent:
            def invoke(self, state):
                return {"messages": [AIMessage(content="hello")]}
            async def ainvoke(self, state):
                return self.invoke(state)
        return Agent()

    meta = AgentMetadata(name="greeter", description="Says hello",
                         capabilities=["chat"], routing_keywords=["hello"],
                         model_name="gpt4o", tools=[], priority=1)
    agent_registry.register("greeter", factory, meta)
    assert "greeter" in agent_registry.list_agents()
    instance = agent_registry.get_agent("greeter")
    assert instance is not None
    result = instance.invoke({"messages": [HumanMessage(content="hi")]})
    assert "hello" in str(result)
    # Second call returns cached instance
    assert agent_registry.get_agent("greeter") is instance


def test_get_agent_not_found():
    try:
        agent_registry.get_agent("ghost")
        assert False, "Should raise"
    except AgentNotFoundError:
        pass


def test_get_agent_factory_none_requires_auto_build():
    agent_registry.register(
        "empty",
        lambda: None,
        AgentMetadata(name="empty", description="empty"),
    )
    try:
        agent_registry.get_agent("empty")
        assert False, "Should raise"
    except AgentBuildError as exc:
        assert "auto_build=True" in str(exc)


def test_get_agent_auto_build_uses_registered_builder():
    class BuiltAgent:
        def invoke(self, state):
            return {"messages": [AIMessage(content="built")]}

    class TestBuilder(BaseAgentBuilder):
        def build(self, metadata):
            assert metadata.name == "auto"
            return BuiltAgent()

    agent_builder_registry.register("test", TestBuilder())
    agent_registry.register(
        "auto",
        lambda: None,
        AgentMetadata(name="auto", description="auto", auto_build=True, agent_type="test"),
    )

    instance = agent_registry.get_agent("auto")
    result = instance.invoke({"messages": [HumanMessage(content="hi")]})
    assert "built" in str(result)


def test_get_metadata():
    meta = AgentMetadata(name="test", description="test agent",
                         capabilities=["a"], routing_keywords=["kw"],
                         model_name="m", tools=["t1", "t2"], priority=5)
    agent_registry.register("test", lambda: object(), meta)
    retrieved = agent_registry.get_metadata("test")
    assert retrieved.description == "test agent"
    assert retrieved.tools == ["t1", "t2"]
    assert retrieved.priority == 5
    assert agent_registry.get_metadata("ghost") is None


def test_list_agents():
    clean_registries()
    assert agent_registry.list_agents() == []
    agent_registry.register("a1", lambda: object(), AgentMetadata(name="a1", description=""))
    agent_registry.register("a2", lambda: object(), AgentMetadata(name="a2", description=""))
    assert sorted(agent_registry.list_agents()) == ["a1", "a2"]


def test_get_agents_by_capability():
    agent_registry.register("cap_a", lambda: object(), AgentMetadata(name="cap_a", description="", capabilities=["search"]))
    agent_registry.register("cap_b", lambda: object(), AgentMetadata(name="cap_b", description="", capabilities=["search", "analysis"]))
    search_agents = agent_registry.get_agents_by_capability("search")
    assert "cap_a" in search_agents
    assert "cap_b" in search_agents
    analysis_agents = agent_registry.get_agents_by_capability("analysis")
    assert "cap_b" in analysis_agents
    assert "cap_a" not in analysis_agents


def test_audit_tools_all_valid():
    @lc_tool
    def valid_tool(x: str) -> str:
        """Valid test tool."""
        return x
    from langdeep.core.registry.tool_registry import tool_registry, ToolMetadata
    tool_registry.register(valid_tool, ToolMetadata(name="valid_tool", description="ok"))

    agent_registry.register("good_agent", lambda: object(), AgentMetadata(
        name="good_agent", description="", tools=["valid_tool"],
    ))
    issues = agent_registry.audit_tools()
    assert issues == []


def test_audit_tools_missing():
    agent_registry.register("bad_agent", lambda: object(), AgentMetadata(
        name="bad_agent", description="", tools=["ghost_tool", "also_missing"],
    ))
    issues = agent_registry.audit_tools()
    assert len(issues) == 1
    assert "bad_agent" in issues[0]
    assert "ghost_tool" in issues[0]


def test_registry_lifecycle_snapshot_reset_and_duplicate_policy():
    meta = AgentMetadata(name="life_agent", description="life")
    agent_registry.register("life_agent", lambda: object(), meta)

    snapshot = agent_registry.snapshot()
    assert snapshot["namespace"] == "default"
    assert "life_agent" in snapshot["metadata"]
    assert "life_agent" in snapshot["factories"]

    try:
        agent_registry.register("life_agent", lambda: object(), meta, replace=False)
        assert False, "Should reject duplicate registration when replace=False"
    except ConfigurationError:
        pass

    agent_registry.reset()
    assert agent_registry.list_agents() == []


def test_registry_namespace_isolation():
    tenant = AgentRegistry.for_namespace("tenant-a")
    tenant.reset()
    tenant.register(
        "tenant_agent",
        lambda: object(),
        AgentMetadata(name="tenant_agent", description="tenant"),
    )

    assert tenant.namespace == "tenant-a"
    assert AgentRegistry("tenant-a") is tenant
    assert "tenant_agent" in tenant.list_agents()
    assert "tenant_agent" not in agent_registry.list_agents()
    tenant.reset()
