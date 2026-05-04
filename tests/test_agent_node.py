"""Unit tests for agent_node factory (make_agent_node) and _extract."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage

from langdeep.core.orchestrator.agent_node import make_agent_node, _extract
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata

from conftest import clean_registries


def setup_function():
    clean_registries()


def _register_dummy_agent(name: str = "dummy"):
    def factory():
        class Agent:
            def __init__(self):
                self.call_count = 0
            def invoke(self, state):
                self.call_count += 1
                return {"messages": [AIMessage(content=f"response from {name}")]}
        return Agent()
    agent_registry.register(name, factory, AgentMetadata(
        name=name, description="dummy", capabilities=["test"],
        routing_keywords=[], model_name="gpt4o",
    ))
    return factory  # return for state inspection


def test_extract_from_dict_with_messages():
    resp = {"messages": [AIMessage(content="hello"), HumanMessage(content="x")]}
    assert _extract(resp) == "hello"


def test_extract_from_string():
    assert _extract("plain text") == "plain text"


def test_extract_from_dict_no_messages():
    assert _extract({"key": "val"}) == str({"key": "val"})


def test_extract_empty():
    assert _extract(None) == "None"
    assert _extract("") == ""


def test_extract_list_content():
    """Handle content that is a list of dicts (e.g. multi-modal)."""
    resp = {"messages": [AIMessage(content=[{"text": "hello "}, {"text": "world"}])]}
    assert _extract(resp) == "hello world"


def test_node_returns_aimessage():
    fn = make_agent_node("dummy")
    state = {"messages": [HumanMessage(content="test")], "task_context": {}}
    result = fn(state)
    assert "messages" in result
    assert "agent_results" in result
    assert "dummy" in result["agent_results"]


def test_node_result_content():
    _register_dummy_agent("echo")
    fn = make_agent_node("echo")
    state = {"messages": [HumanMessage(content="hello")], "task_context": {}}
    result = fn(state)
    assert "echo" in result["messages"][0].content


def test_node_retry_on_failure():
    """Node should retry if agent fails initially."""
    class FlakyAgent:
        def __init__(self):
            self.tries = 0
        def invoke(self, state):
            self.tries += 1
            if self.tries < 2:
                raise RuntimeError("not yet")
            return {"messages": [AIMessage(content="success on retry")]}

    agent_registry.register("flaky", lambda: FlakyAgent(), AgentMetadata(
        name="flaky", description="flaky", capabilities=[],
        routing_keywords=[], model_name="gpt4o",
    ))
    fn = make_agent_node("flaky", max_retries=3)
    state = {"messages": [HumanMessage(content="go")], "task_context": {}}
    result = fn(state)
    assert "success on retry" in result["messages"][0].content


def test_node_exhausts_retries():
    """When max_retries exhausted, return error message not crash."""
    class AlwaysFail:
        def invoke(self, state):
            raise RuntimeError("always fail")

    agent_registry.register("bad", lambda: AlwaysFail(), AgentMetadata(
        name="bad", description="bad", capabilities=[],
        routing_keywords=[], model_name="gpt4o",
    ))
    fn = make_agent_node("bad", max_retries=2)
    state = {"messages": [HumanMessage(content="go")], "task_context": {}}
    result = fn(state)
    assert "exhausted" in result["messages"][0].content.lower()
    assert "always fail" in result["messages"][0].content


def test_clean_messages_fn_integration():
    """If clean_messages_fn is passed, it should filter messages."""
    from langdeep.core.orchestrator.executor import _clean_messages

    def factory():
        class Agent:
            def invoke(self, state):
                msgs = state.get("messages", [])
                return {"messages": [AIMessage(content=f"got {len(msgs)} messages")]}
        return Agent()

    agent_registry.register("counter", factory, AgentMetadata(
        name="counter", description="counter", capabilities=[],
        routing_keywords=[], model_name="gpt4o",
    ))

    # Without cleaning
    fn_no_clean = make_agent_node("counter")
    noisy_state = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="")],
        "task_context": {},
    }
    result_no = fn_no_clean(noisy_state)
    # The agent receives all messages including empty ones since clean_messages_fn is None
    # With cleaning
    fn_clean = make_agent_node("counter", clean_messages_fn=_clean_messages)
    result_clean = fn_clean(noisy_state)
    # Both should work (cleaning removes the empty AIMessages)
    assert "got" in result_clean["messages"][0].content
