"""Unit/integration tests for FlowOrchestrator — construction, routing, streaming."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from langchain_core.messages import AIMessage, HumanMessage

from langdeep import FlowOrchestrator, ExecutionPolicy, RoutingStrategy
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata

from conftest import clean_registries, populate_minimal_registries, _mock, orch, _last_ai


def setup_function():
    clean_registries()
    populate_minimal_registries()


def test_orchestrator_construction_defaults():
    o = orch()
    assert o._supervisor_model is not None
    assert o._graph is not None


def test_orchestrator_with_custom_policy():
    policy = ExecutionPolicy(strategy="sequential", max_concurrency=1)
    o = orch(execution_policy=policy)
    assert o._policy.strategy == "sequential"


def test_invoke_basic():
    o = orch()
    result = o.invoke("你好")
    assert "messages" in result
    ans = _last_ai(result["messages"])
    assert len(ans) > 5


def test_invoke_with_context():
    o = orch()
    result = o.invoke("test", context={"session": "abc"})
    assert result is not None


def test_invoke_with_workflow_plan():
    o = orch()
    plan = [
        {"id": "t1", "agent": "test_echo", "depends_on": [], "status": "pending"},
    ]
    result = o.invoke("do it", workflow_plan=plan)
    assert "agent_results" in result


def test_ainvoke():
    o = orch()
    async def run():
        result = await o.ainvoke("你好")
        ans = _last_ai(result["messages"])
        assert len(ans) > 5
        return result
    r = asyncio.run(run())
    assert r is not None


def test_astream():
    o = orch()
    async def run():
        nodes = set()
        async for chunk in o.astream("搜索 news"):
            for node_name in chunk:
                nodes.add(node_name)
        assert len(nodes) >= 1
    asyncio.run(run())


def test_orchestrator_custom_nodes():
    def my_node(state):
        return {
            "messages": state.get("messages", []),
            "agent_results": {"custom": "custom result"},
        }

    class ToCustom(RoutingStrategy):
        def route(self, user_input, agents):
            return "my_custom"

    o = orch(custom_nodes={"my_custom": my_node}, routing_strategy=ToCustom())
    result = o.invoke("trigger custom")
    assert "custom" in result.get("agent_results", {})
    assert "custom result" in result["agent_results"]["custom"]


def test_orchestrator_custom_routing_strategy():
    class AlwaysEcho(RoutingStrategy):
        def route(self, user_input, agents):
            return "test_echo"

    o = orch(routing_strategy=AlwaysEcho())
    result = o.invoke("anything")
    assert "echo" in _last_ai(result["messages"]).lower()


def test_router_valid_targets():
    o = orch()
    targets = o._get_valid_targets()
    assert "planner" in targets
    assert "end" in targets
    assert "test_echo" in targets


def test_initial_state():
    o = orch()
    state = o._initial_state("hello", context={"k": "v"}, workflow_plan=None)
    assert state["messages"][0].content == "hello"
    assert state["task_context"] == {"k": "v"}
    assert state["workflow_plan"] is None
    assert state["aggregation_done"] is False
    assert state["error_count"] == 0


def test_route_from_supervisor():
    o = orch()
    assert o._route_from_supervisor({"next": "end"}) == "end"
    assert o._route_from_supervisor({"next": "planner"}) == "planner"
    assert o._route_from_supervisor({"next": "invalid"}) == "end"


def test_get_available_agents():
    o = orch()
    agents = o._get_available_agents()
    assert len(agents) >= 1
    names = [a["name"] for a in agents]
    assert "test_echo" in names


def test_error_on_invoke():
    """Orchestrator should raise OrchestrationError on graph failure."""
    class BrokenGraph:
        def invoke(self, initial):
            raise RuntimeError("graph crash")
        async def ainvoke(self, initial):
            raise RuntimeError("graph crash")

    from langdeep.core.errors import OrchestrationError
    o = orch()
    # Replace the compiled graph with a broken one
    o._graph = BrokenGraph()  # type: ignore
    try:
        o.invoke("hello")
        assert False, "Should raise"
    except OrchestrationError:
        pass
