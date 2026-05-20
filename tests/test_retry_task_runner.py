"""Unit tests for RetryTaskRunner — context injection, retry, timeout."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langdeep.core.orchestrator.executor import RetryTaskRunner, ok, err
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata
from langdeep.core.errors import AgentNotFoundError

from conftest import clean_registries


def setup_function():
    clean_registries()


def _register_echo_agent():
    def factory():
        class Agent:
            def invoke(self, state):
                msgs = state.get("messages", [])
                context = state.get("task_context", {})
                return {"messages": [AIMessage(content=f"echo|msgs={len(msgs)}|prev={list(context.get('previous_results', {}).keys())}")]}
            async def ainvoke(self, state):
                return self.invoke(state)
        return Agent()
    agent_registry.register("echo", factory, AgentMetadata(
        name="echo", description="echo", capabilities=[], routing_keywords=[], model_name="gpt4o",
    ))


def test_run_success():
    _register_echo_agent()
    runner = RetryTaskRunner(max_retries=1, timeout=5)
    task = {"id": "t1", "agent": "echo"}
    msgs = [HumanMessage(content="hello")]
    result = runner.run(task, msgs, {"task_context": {}}, {})
    assert result["success"] is True
    assert "echo" in result["data"]


def test_run_agent_not_registered():
    runner = RetryTaskRunner(max_retries=1)
    task = {"id": "t1", "agent": "ghost"}
    result = runner.run(task, [], {}, {})
    assert result["success"] is False
    assert "not registered" in result["error"]


def test_run_retry_then_succeed():
    class Flaky:
        def __init__(self):
            self.tries = 0
        def invoke(self, state):
            self.tries += 1
            if self.tries < 3:
                raise RuntimeError(f"fail attempt {self.tries}")
            return {"messages": [AIMessage(content="finally ok")]}
        async def ainvoke(self, state):
            return self.invoke(state)

    agent_registry.register("flaky", lambda: Flaky(), AgentMetadata(
        name="flaky", description="flaky", capabilities=[], routing_keywords=[], model_name="gpt4o",
    ))
    runner = RetryTaskRunner(max_retries=5)
    result = runner.run({"id": "t1", "agent": "flaky"}, [HumanMessage(content="go")], {}, {})
    assert result["success"] is True
    assert "finally ok" in result["data"]


def test_run_exhaust_retries():
    class AlwaysFail:
        def invoke(self, state):
            raise RuntimeError("persistent failure")
        async def ainvoke(self, state):
            raise RuntimeError("persistent failure")

    agent_registry.register("bad", lambda: AlwaysFail(), AgentMetadata(
        name="bad", description="bad", capabilities=[], routing_keywords=[], model_name="gpt4o",
    ))
    runner = RetryTaskRunner(max_retries=2)
    result = runner.run({"id": "t1", "agent": "bad"}, [HumanMessage(content="x")], {}, {})
    assert result["success"] is False
    assert "exhausted" in result["error"]


def test_run_bridges_async_only_agent():
    class AsyncOnlyAgent:
        async def ainvoke(self, state):
            return {"messages": [AIMessage(content="async from sync")]}

    agent_registry.register("async_only", lambda: AsyncOnlyAgent(), AgentMetadata(
        name="async_only", description="async only", capabilities=[], routing_keywords=[],
        model_name="gpt4o",
    ))
    runner = RetryTaskRunner(max_retries=1)
    result = runner.run({"id": "t1", "agent": "async_only"}, [HumanMessage(content="x")], {}, {})
    assert result == ok("async from sync")


def test_inject_context_with_previous_results():
    _register_echo_agent()
    runner = RetryTaskRunner(max_retries=1)
    task = {"id": "t2", "agent": "echo"}
    prev = {"t1": ok("first result")}
    msgs = [HumanMessage(content="do it")]
    result = runner.run(task, msgs, {}, prev)
    assert result["success"] is True
    # The injected SystemMessage should have increased the message count
    assert "msgs=2" in result["data"]  # 1 SystemMessage + 1 HumanMessage


def test_inject_context_empty_results():
    _register_echo_agent()
    runner = RetryTaskRunner(max_retries=1)
    result = runner.run({"id": "t1", "agent": "echo"}, [HumanMessage(content="hi")], {}, {})
    assert result["success"] is True
    assert "msgs=1" in result["data"]


def test_inject_context_skips_errors():
    _register_echo_agent()
    runner = RetryTaskRunner(max_retries=1)
    task = {"id": "t3", "agent": "echo"}
    prev = {"t_fail": err("something broke"), "t_ok": ok("good result")}
    msgs = [HumanMessage(content="go")]
    result = runner.run(task, msgs, {}, prev)
    assert result["success"] is True
    assert "msgs=2" in result["data"]


# ── Async tests ──────────────────────────────────────────────────────

def test_arun_success():
    _register_echo_agent()
    runner = RetryTaskRunner(max_retries=1)
    task = {"id": "t1", "agent": "echo"}

    async def run():
        return await runner.arun(task, [HumanMessage(content="hello")], {}, {})
    result = asyncio.run(run())
    assert result["success"] is True


def test_arun_falls_back_to_sync_only_agent():
    class SyncOnlyAgent:
        def invoke(self, state):
            return {"messages": [AIMessage(content="sync from async")]}

    agent_registry.register("sync_only", lambda: SyncOnlyAgent(), AgentMetadata(
        name="sync_only", description="sync only", capabilities=[], routing_keywords=[],
        model_name="gpt4o",
    ))
    runner = RetryTaskRunner(max_retries=1)

    async def run():
        return await runner.arun({"id": "t1", "agent": "sync_only"}, [HumanMessage(content="x")], {}, {})

    result = asyncio.run(run())
    assert result == ok("sync from async")


def test_arun_timeout():
    class SlowAgent:
        async def ainvoke(self, state):
            await asyncio.sleep(10)
            return {"messages": [AIMessage(content="slow")]}

    agent_registry.register("slow", lambda: SlowAgent(), AgentMetadata(
        name="slow", description="slow", capabilities=[], routing_keywords=[], model_name="gpt4o",
    ))
    runner = RetryTaskRunner(max_retries=1, timeout=0.05)
    task = {"id": "t1", "agent": "slow"}

    async def run():
        return await runner.arun(task, [HumanMessage(content="x")], {}, {})
    result = asyncio.run(run())
    assert result["success"] is False


def test_static_inject_context_empty():
    msgs = [HumanMessage(content="hello")]
    result = RetryTaskRunner._inject_context(msgs, {})
    assert result is msgs  # same list reference, no change


def test_static_inject_context_with_data():
    msgs = [HumanMessage(content="hello")]
    result = RetryTaskRunner._inject_context(msgs, {"t1": ok("previous task")})
    assert len(result) == 2
    assert isinstance(result[0], SystemMessage)
    assert "Previous" in result[0].content
    assert result[1] is msgs[0]
