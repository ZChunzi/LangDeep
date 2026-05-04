"""Unit tests for Executor — task execution, batch modes, dependency resolution."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage

from langdeep.core.orchestrator.executor import Executor, _dependencies_satisfied
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

    # Agent that captures task_context for verification
    class ContextCapture:
        def invoke(self, state):
            ctx = state.get("task_context", {})
            task = ctx.get("task", {})
            prev = ctx.get("previous_results", {})
            return {"messages": [AIMessage(content=f"task={task.get('id')}|prev={list(prev.keys())}")]}
        async def ainvoke(self, state):
            return self.invoke(state)

    agent_registry.register("ctx_catcher", lambda: ContextCapture(), AgentMetadata(
        name="ctx_catcher", description="captures context",
        capabilities=[], routing_keywords=[], model_name="gpt4o",
    ))


def test_execute_no_pending_tasks():
    _register_agents()
    ex = Executor()
    state = {"messages": [HumanMessage(content="hi")], "workflow_plan": []}
    result = ex.execute(state)
    assert "All tasks completed" in result["messages"][0].content


def test_execute_no_workflow_plan():
    _register_agents()
    ex = Executor()
    state = {"messages": [HumanMessage(content="hi")]}
    result = ex.execute(state)
    assert "All tasks completed" in result["messages"][0].content


def test_execute_single_task():
    _register_agents()
    ex = Executor()
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "t1" in result["agent_results"]
    assert "agent_a" in result["agent_results"]["t1"]


def test_execute_sequential_dependency():
    _register_agents()
    ex = Executor()
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "pending"},
            {"id": "t2", "agent": "agent_b", "depends_on": ["t1"], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "t1" in result["agent_results"]
    assert "t2" in result["agent_results"]


def test_execute_unsatisfied_dependency():
    _register_agents()
    ex = Executor()
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": ["ghost_task"], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "Dependency unsatisfied" in result["agent_results"]["t1"]


def test_execute_gather_policy():
    _register_agents()
    ex = Executor(policy=ExecutionPolicy(strategy="gather", max_concurrency=5))
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "pending"},
            {"id": "t2", "agent": "agent_b", "depends_on": [], "status": "pending"},
            {"id": "t3", "agent": "agent_c", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    for tid in ["t1", "t2", "t3"]:
        assert tid in result["agent_results"]


def test_execute_sequential_policy():
    _register_agents()
    ex = Executor(policy=ExecutionPolicy(strategy="sequential"))
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "t1", "agent": "agent_a", "depends_on": [], "status": "pending"},
            {"id": "t2", "agent": "agent_b", "depends_on": [], "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert len(result["agent_results"]) == 2


def test_execute_priority_policy():
    _register_agents()
    ex = Executor(policy=ExecutionPolicy(strategy="priority_queue"))
    state = {
        "messages": [HumanMessage(content="do it")],
        "workflow_plan": [
            {"id": "low", "agent": "agent_a", "depends_on": [], "priority": 1, "status": "pending"},
            {"id": "high", "agent": "agent_b", "depends_on": [], "priority": 10, "status": "pending"},
        ],
        "task_context": {},
    }
    result = ex.execute(state)
    assert "low" in result["agent_results"]
    assert "high" in result["agent_results"]


def test_execute_context_passing():
    _register_agents()
    ex = Executor()
    state = {
        "messages": [HumanMessage(content="test")],
        "workflow_plan": [
            {"id": "ctx1", "agent": "ctx_catcher", "depends_on": [], "status": "pending"},
        ],
        "task_context": {"global_key": "global_val"},
    }
    result = ex.execute(state)
    assert "ctx1" in result["agent_results"]
    assert "global_val" not in str(result)  # context not leaked to results dict


def test_dependencies_satisfied():
    results = {"t1": {"success": True}}
    assert _dependencies_satisfied({"depends_on": ["t1"]}, results) is True
    assert _dependencies_satisfied({"depends_on": ["t2"]}, results) is False
    assert _dependencies_satisfied({"depends_on": []}, results) is True
    assert _dependencies_satisfied({}, results) is True
