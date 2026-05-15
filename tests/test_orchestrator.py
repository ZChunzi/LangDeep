"""Unit/integration tests for FlowOrchestrator — construction, routing, streaming."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from langchain_core.messages import AIMessage, HumanMessage

from langdeep import FlowOrchestrator, ExecutionPolicy, RoutingStrategy
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata
from langdeep.core.memory import InMemoryBackend, memory_registry
from langdeep.core.observability import MetricsCollector
from langdeep.core.process import ProcessManager, ProcessState

from conftest import clean_registries, populate_minimal_registries, _mock, orch, _last_ai, ToPlanner


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


def test_invoke_accepts_langgraph_style_messages_state():
    o = orch()
    result = o.invoke({"messages": [HumanMessage(content="你好")]})

    assert "messages" in result
    assert any(isinstance(message, HumanMessage) for message in result["messages"])
    assert len(_last_ai(result["messages"])) > 5


def test_invoke_accepts_message_sequence():
    o = orch()
    result = o.invoke_messages([HumanMessage(content="你好")])

    assert "messages" in result
    assert len(_last_ai(result["messages"])) > 5


def test_invoke_accepts_single_message_with_memory_context():
    backend = InMemoryBackend()
    memory_registry.register("single_message_mem", lambda: backend)
    o = orch(memory="single_message_mem")

    result = o.invoke(HumanMessage(content="你好"), context={"session_id": "single"})

    assert "messages" in result
    stored = backend.load_messages("single")
    assert any(isinstance(message, HumanMessage) and message.content == "你好" for message in stored)


def test_invoke_with_context():
    o = orch()
    result = o.invoke("test", context={"session": "abc"})
    assert result is not None


def test_invoke_records_shared_metrics():
    metrics = MetricsCollector()
    o = orch(
        routing_strategy=ToPlanner(),
        execution_policy=ExecutionPolicy(strategy="sequential"),
        metrics_collector=metrics,
    )
    plan = [{
        "id": "task_1",
        "agent": "test_echo",
        "depends_on": [],
        "status": "pending",
    }]

    o.invoke("collect metrics", workflow_plan=plan)

    collected = o.get_metrics()
    counters = collected["counters"]
    assert counters["orchestrator.invocations|mode=sync"] == 1
    assert counters["orchestrator.node.calls|node=supervisor"] == 1
    assert counters["routing.fast_path_hits|next=planner"] == 1
    assert counters["planning.reused"] == 1
    assert counters["execution.task_results|status=success"] == 1
    assert counters["aggregation.results|status=single"] == 1
    assert "orchestrator.duration_ms|mode=sync,status=success" in collected["histograms"]

    o.clear_metrics()
    assert o.get_metrics()["counters"] == {}


def test_invoke_loads_and_stores_memory_by_session_id():
    backend = InMemoryBackend()
    memory_registry.register("session_mem", lambda: backend)

    def create_memory_agent():
        class MemoryAgent:
            def invoke(self, state):
                human_count = sum(
                    1 for message in state.get("messages", [])
                    if isinstance(message, HumanMessage)
                )
                return {"messages": [AIMessage(content=f"humans={human_count}")]}

            async def ainvoke(self, state):
                return self.invoke(state)

        return MemoryAgent()

    agent_registry.register(
        "memory_agent",
        create_memory_agent,
        AgentMetadata(
            name="memory_agent",
            description="Counts human messages in session memory",
            routing_keywords=["remember"],
            model_name="gpt4o",
        ),
    )

    o = orch(memory="session_mem")

    first = o.invoke("remember first", context={"session_id": "s1"})
    assert _last_ai(first["messages"]) == "humans=1"

    second = o.invoke("remember second", context={"session_id": "s1"})
    assert _last_ai(second["messages"]) == "humans=2"

    stored = backend.load_messages("s1")
    stored_human = [message.content for message in stored if isinstance(message, HumanMessage)]
    assert stored_human == ["remember first", "remember second"]
    assert any(message.content == "humans=2" for message in stored if isinstance(message, AIMessage))


def test_chat_uses_session_memory_without_context_boilerplate():
    backend = InMemoryBackend()
    memory_registry.register("chat_mem", lambda: backend)

    def create_memory_agent():
        class MemoryAgent:
            def invoke(self, state):
                human_count = sum(
                    1 for message in state.get("messages", [])
                    if isinstance(message, HumanMessage)
                )
                return {"messages": [AIMessage(content=f"humans={human_count}")]}

        return MemoryAgent()

    agent_registry.register(
        "chat_memory_agent",
        create_memory_agent,
        AgentMetadata(
            name="chat_memory_agent",
            description="Counts human messages in session memory",
            routing_keywords=["chatmem"],
            model_name="gpt4o",
        ),
    )

    o = orch(memory="chat_mem")

    first = o.chat("chatmem first", session_id="cli")
    second = o.chat("chatmem second", session_id="cli")

    assert _last_ai(first["messages"]) == "humans=1"
    assert _last_ai(second["messages"]) == "humans=2"


def test_invoke_dict_without_supported_keys_has_clear_error():
    o = orch()

    try:
        o.invoke({"unexpected": "value"})
        assert False, "Should reject unsupported dict input"
    except Exception as exc:
        assert "without 'messages', 'input', 'user_input', or 'content'" in str(exc)


def test_invoke_updates_process_snapshot_by_process_id():
    process_manager = ProcessManager()
    process = process_manager.create("chat")
    o = orch(process_manager=process_manager)

    result = o.invoke("你好", context={"process_id": process.pid})

    stored = process_manager.get_process(process.pid)
    assert stored is not None
    assert stored.state == ProcessState.ACTIVE
    assert stored.snapshot["task_context"]["process_id"] == process.pid
    assert stored.snapshot["agent_results"] == result["agent_results"]


def test_waiting_confirmation_sets_process_awaiting_human():
    process_manager = ProcessManager()
    process = process_manager.create("approval")
    o = orch(routing_strategy=ToPlanner(), process_manager=process_manager)
    plan = [
        {
            "id": "needs_approval",
            "agent": "test_echo",
            "depends_on": [],
            "status": "pending",
            "requires_confirmation": True,
        }
    ]

    result = o.invoke(
        "needs approval",
        context={"process_id": process.pid},
        workflow_plan=plan,
    )

    stored = process_manager.get_process(process.pid)
    assert result["workflow_plan"][0]["status"] == "waiting_confirmation"
    assert stored is not None
    assert stored.state == ProcessState.AWAITING_HUMAN
    assert stored.snapshot["workflow_plan"][0]["status"] == "waiting_confirmation"


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


def test_ainvoke_accepts_langgraph_style_messages_state():
    o = orch()

    async def run():
        return await o.ainvoke({"messages": [HumanMessage(content="你好")]})

    result = asyncio.run(run())
    assert len(_last_ai(result["messages"])) > 5


def test_ainvoke_prefers_graph_async_api():
    """ainvoke should use the graph's async entrypoint when it exists."""

    class AsyncGraph:
        def __init__(self):
            self.initial = None

        def invoke(self, initial):
            raise AssertionError("sync invoke should not be called")

        async def ainvoke(self, initial):
            self.initial = initial
            return {"messages": [AIMessage(content="async result")]}

    o = orch()
    graph = AsyncGraph()
    o._graph = graph  # type: ignore

    async def run():
        return await o.ainvoke("async boundary")

    result = asyncio.run(run())
    assert _last_ai(result["messages"]) == "async result"
    assert graph.initial["messages"][0].content == "async boundary"


def test_astream():
    o = orch()
    async def run():
        nodes = set()
        async for chunk in o.astream("搜索 news"):
            for node_name in chunk:
                nodes.add(node_name)
        assert len(nodes) >= 1
    asyncio.run(run())


def test_astream_prefers_graph_async_stream_api():
    """astream should consume the graph's async stream when it exists."""

    class AsyncStreamGraph:
        def stream(self, initial, **kwargs):
            raise AssertionError("sync stream should not be called")

        async def astream(self, initial, **kwargs):
            yield {"first": {"messages": [AIMessage(content=initial["messages"][0].content)]}}
            yield {"second": {"messages": [AIMessage(content=str(kwargs.get("mode")))]}}

    o = orch()
    o._graph = AsyncStreamGraph()  # type: ignore

    async def run():
        chunks = []
        async for chunk in o.astream("stream boundary", mode="values"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    assert chunks[0]["first"]["messages"][0].content == "stream boundary"
    assert chunks[1]["second"]["messages"][0].content == "values"


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
