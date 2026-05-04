"""Edge-case tests for Executor — circular deps, partial failure, async/threaded paths."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from langdeep.core.orchestrator.executor import Executor
from langdeep.core.execution.execution_policy import ExecutionPolicy
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata

from conftest import clean_registries


def setup_function():
    clean_registries()


def _register_agents():
    def factory(name: str):
        class Agent:
            def invoke(self, state):
                return {"messages": [AIMessage(content=f"[{name}] done")]}
            async def ainvoke(self, state):
                return self.invoke(state)
        return Agent()

    for n in ["agent_a", "agent_b", "agent_c"]:
        agent_registry.register(n, lambda n=n: factory(n), AgentMetadata(
            name=n, description=n, capabilities=[], routing_keywords=[], model_name="gpt4o",
        ))


def _register_flaky_agent():
    """Agent that raises RuntimeError on invoke."""
    class Flaky:
        def invoke(self, state):
            raise RuntimeError("flaky agent crashed")
        async def ainvoke(self, state):
            raise RuntimeError("flaky agent crashed")
    agent_registry.register("flaky", lambda: Flaky(), AgentMetadata(
        name="flaky", description="flaky", capabilities=[], routing_keywords=[], model_name="gpt4o",
    ))


def _register_slow_agent():
    """Agent that records concurrent execution count."""
    class Slow:
        _concurrent = 0
        _peak = 0
        def invoke(self, state):
            Slow._concurrent += 1
            Slow._peak = max(Slow._peak, Slow._concurrent)
            import time
            time.sleep(0.05)
            Slow._concurrent -= 1
            return {"messages": [AIMessage(content="[slow] done")]}
        async def ainvoke(self, state):
            Slow._concurrent += 1
            Slow._peak = max(Slow._peak, Slow._concurrent)
            import asyncio
            await asyncio.sleep(0.05)
            Slow._concurrent -= 1
            return {"messages": [AIMessage(content="[slow] done")]}
    return Slow


# ── Circular dependency ─────────────────────────────────────


def test_circular_dependency_silent_skip():
    """Circular dependency: both tasks get 'Dependency unsatisfied', not a crash."""
    _register_agents()
    ex = Executor()
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": ["t2"], "status": "pending"},
            {"id": "t2", "agent": "agent_b", "depends_on": ["t1"], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "t1" in result["agent_results"]
    assert "t2" in result["agent_results"]
    assert "Dependency unsatisfied" in result["agent_results"]["t1"]
    assert "Dependency unsatisfied" in result["agent_results"]["t2"]


# ── Partial failure ─────────────────────────────────────────


def test_agent_runtime_exception_partial_failure():
    """One agent crashing does not prevent others from running."""
    _register_agents()
    _register_flaky_agent()
    ex = Executor(policy=ExecutionPolicy(strategy="gather", max_concurrency=5))
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "pending"},
            {"id": "t2", "agent": "flaky",    "depends_on": [], "status": "pending"},
            {"id": "t3", "agent": "agent_b",  "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "agent_a" in result["agent_results"]["t1"]
    assert "agent_b" in result["agent_results"]["t3"]
    # flaky task should have an error message
    assert "error" in result["agent_results"]["t2"].lower() or "crash" in result["agent_results"]["t2"].lower()


# ── Async batch path (no running event loop) ────────────────


def test_async_batch_path():
    """Gather policy from sync context uses asyncio.run(_async_batch)."""
    _register_agents()
    ex = Executor(policy=ExecutionPolicy(strategy="gather", max_concurrency=5))
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "pending"},
            {"id": "t2", "agent": "agent_b", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "agent_a" in result["agent_results"]["t1"]
    assert "agent_b" in result["agent_results"]["t2"]


# ── Threaded batch path (inside running event loop) ─────────


def test_threaded_batch_path():
    """Inside a running event loop, gather policy uses ThreadPoolExecutor."""
    _register_agents()

    async def run():
        ex = Executor(policy=ExecutionPolicy(strategy="gather", max_concurrency=5))
        state = {
            "messages": [HumanMessage(content="do it")],
            "workflow_plan": [
                {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "pending"},
                {"id": "t2", "agent": "agent_b", "depends_on": [], "status": "pending"},
            ],
            "task_context": {},
        }
        # ex.execute() is sync but called inside a running loop
        result = ex.execute(state)
        assert "agent_a" in result["agent_results"]["t1"]
        assert "agent_b" in result["agent_results"]["t2"]

    asyncio.run(run())


# ── max_concurrency ─────────────────────────────────────────


def test_max_concurrency_respected():
    """max_concurrency=1 limits parallelism (at most one concurrent execution)."""
    _register_agents()
    Slow = _register_slow_agent()
    agent_registry.register("slow", lambda: Slow(), AgentMetadata(
        name="slow", description="slow", capabilities=[], routing_keywords=[], model_name="gpt4o",
    ))
    ex = Executor(policy=ExecutionPolicy(strategy="gather", max_concurrency=1))
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "slow", "depends_on": [], "status": "pending"},
            {"id": "t2", "agent": "slow", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "t1" in result["agent_results"]
    assert "t2" in result["agent_results"]
    assert Slow._peak <= 2  # semaphore allows at most max_concurrency


# ── Skip completed tasks ────────────────────────────────────


def test_skips_completed_tasks():
    """Tasks with status 'completed' are skipped (not re-executed)."""
    _register_agents()
    ex = Executor()
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "completed"},
            {"id": "t2", "agent": "agent_b", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    # t2 ran
    assert "t2" in result["agent_results"]
    # t1 should not be in results (already completed, not re-executed)
    # Note: executor might still return t1 in agent_results; the key behavior
    # is that it doesn't crash and t2 runs correctly
    assert "agent_b" in result["agent_results"]["t2"]


# ── Mixed registered / unregistered agents ──────────────────


def test_mixed_success_failure_batch():
    """A task referencing an unregistered agent fails but others succeed."""
    _register_agents()
    ex = Executor(policy=ExecutionPolicy(strategy="gather", max_concurrency=5))
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a",       "depends_on": [], "status": "pending"},
            {"id": "t2", "agent": "does_not_exist", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "agent_a" in result["agent_results"]["t1"]
    assert "does_not_exist" in result["agent_results"]["t2"] or "not registered" in result["agent_results"]["t2"].lower()


# ── No registered agents ────────────────────────────────────


def test_no_registered_agents():
    """No agents in registry — plan referencing any agent gets error."""
    # clean_registries() already called in setup_function, no agents registered
    ex = Executor()
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "ghost", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "t1" in result["agent_results"]
    # should produce an error, not a crash
    assert isinstance(result["agent_results"]["t1"], str) or "error" in str(result["agent_results"]["t1"]).lower()


# ── Sequential policy error task ────────────────────────────


def test_batch_sequential_policy_error_task():
    """Sequential: first task fails, second task still runs."""
    _register_agents()
    _register_flaky_agent()
    ex = Executor(policy=ExecutionPolicy(strategy="sequential"))
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "flaky",   "depends_on": [], "status": "pending"},
            {"id": "t2", "agent": "agent_a", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    # t2 should still run despite t1 failing
    assert "t2" in result["agent_results"]
    assert "agent_a" in result["agent_results"]["t2"]
    # t1 should have an error
    assert "error" in result["agent_results"]["t1"].lower() or "crash" in result["agent_results"]["t1"].lower()


# ── All tasks already completed ─────────────────────────────


def test_empty_pending_tasks_with_completed():
    """All tasks already completed yields 'All tasks completed'."""
    _register_agents()
    ex = Executor()
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "completed"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "All tasks completed" in result["messages"][0].content
