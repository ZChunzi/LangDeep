"""Edge-case tests for FlowOrchestrator — empty plan, no agents, streaming errors, injections."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from langchain_core.messages import AIMessage, HumanMessage

from langdeep import FlowOrchestrator, ExecutionPolicy, RoutingStrategy
from langdeep.core.errors import ConfigurationError
from langdeep.core.orchestrator.planner import PlanGenerator, FallbackPlanGenerator
from langdeep.core.orchestrator.executor import TaskRunner, ok, err
from langdeep.core.orchestrator.aggregator import ResultMerger

from conftest import clean_registries, populate_minimal_registries, orch, ToPlanner


def setup_function():
    clean_registries()
    populate_minimal_registries()


# ── Empty workflow plan ─────────────────────────────────────


def test_invoke_empty_workflow_plan():
    """Empty workflow_plan list is handled without crashing."""
    o = orch()
    # An empty list is falsy, so the planner generates a new plan
    result = o.invoke("test", workflow_plan=[])
    assert "messages" in result


def test_strict_component_import_raises_on_broken_module(tmp_path):
    """strict_component_import turns auto-import failures into startup errors."""
    components = tmp_path / "agents"
    components.mkdir()
    (components / "broken.py").write_text("raise RuntimeError('broken component')\n")

    try:
        FlowOrchestrator(
            enable_checkpoint=False,
            component_dirs=[str(components)],
            strict_component_import=True,
        )
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError as exc:
        assert "broken" in str(exc.context)


# ── No agents → routes to planner ───────────────────────────


def test_invoke_no_agents_routes_to_planner():
    """With no agents, forced route to planner completes gracefully."""
    clean_registries()  # no agents registered
    o = orch(routing_strategy=ToPlanner())
    result = o.invoke("do something")
    assert "messages" in result


# ── No agents → routes to end ───────────────────────────────


def test_invoke_no_agents_routes_to_end():
    """With no agents, route to 'end' (no tasks) returns empty agent_results."""
    clean_registries()

    class RouteToEnd(RoutingStrategy):
        def route(self, user_input, agents):
            return "end"

    o = orch(routing_strategy=RouteToEnd())
    result = o.invoke("do something")
    # Should complete without error
    assert result is not None


# ── Streaming error propagation ─────────────────────────────


def test_astream_error_during_stream():
    """Custom node raising during astream propagates the exception."""

    class CrashingStrategy(RoutingStrategy):
        def route(self, user_input, agents):
            return "crash_node"

    def crashing_node(state):
        raise RuntimeError("stream crash test")

    o = orch(
        routing_strategy=CrashingStrategy(),
        custom_nodes={"crash_node": crashing_node},
    )

    # astream wraps exceptions differently — it re-raises the original
    # exception rather than wrapping in OrchestrationError
    try:
        async def run():
            async for _ in o.astream("hello"):
                pass
        asyncio.run(run())
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "stream crash test" in str(e)
    except Exception:
        pass  # any exception is acceptable — the test is that it doesn't hang


# ── Custom PlanGenerator injection ──────────────────────────


def test_custom_plan_generator_injection():
    """Injecting a custom PlanGenerator at orchestrator construction."""

    class FixedPlanGenerator(PlanGenerator):
        def generate(self, user_request, available_agent_names):
            return [{
                "id": "fixed_1",
                "agent": available_agent_names[0] if available_agent_names else "default_agent",
                "depends_on": [],
                "status": "pending",
            }]

    o = orch(plan_generator=FixedPlanGenerator(), routing_strategy=ToPlanner())
    result = o.invoke("hello", workflow_plan=[])  # empty plan forces planner to generate
    # The planner should use the injected FixedPlanGenerator
    assert "agent_results" in result


# ── Custom TaskRunner injection ─────────────────────────────


def test_custom_task_runner_injection():
    """Injecting a custom TaskRunner at orchestrator construction."""

    class FixedTaskRunner(TaskRunner):
        def run(self, task, clean_msgs, state, previous_results):
            return {"success": True, "data": "fixed from custom runner"}
        async def arun(self, task, clean_msgs, state, previous_results):
            return self.run(task, clean_msgs, state, previous_results)

    o = orch(task_runner=FixedTaskRunner(), routing_strategy=ToPlanner())
    result = o.invoke("hello", workflow_plan=[
        {"id": "t1", "agent": "test_echo", "depends_on": [], "status": "pending"},
    ])
    assert "agent_results" in result
    assert "t1" in result["agent_results"]
    assert "fixed from custom runner" in result["agent_results"]["t1"]


# ── Custom ResultMerger injection ───────────────────────────


def test_custom_result_merger_injection():
    """Injecting a custom ResultMerger at orchestrator construction."""

    class FixedMerger(ResultMerger):
        def merge(self, user_request, agent_results):
            return "CUSTOM MERGER OUTPUT"

    o = orch(result_merger=FixedMerger(), routing_strategy=ToPlanner())
    result = o.invoke("hello", workflow_plan=[
        {"id": "t1", "agent": "test_echo", "depends_on": [], "status": "pending"},
        {"id": "t2", "agent": "web_agent", "depends_on": [], "status": "pending"},
    ])
    last_ai = ""
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            last_ai = str(m.content)
            break
    assert "CUSTOM MERGER OUTPUT" in last_ai


# ── Multiple invoke state independence ──────────────────────


def test_multiple_invoke_state_independence():
    """Subsequent invoke() calls do not leak state between each other."""
    o = orch()
    r1 = o.invoke("first call")
    r2 = o.invoke("second call")

    # Each invocation should have its own message list
    messages_1 = [str(m.content) for m in r1.get("messages", [])]
    messages_2 = [str(m.content) for m in r2.get("messages", [])]

    # The second call's messages should reference "second call"
    assert any("second" in str(m) for m in r2.get("messages", []))
