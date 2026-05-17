"""Unit tests for KeywordRoutingStrategy — word-boundary matching."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langdeep.core.orchestrator.router import DefaultRouter, KeywordRoutingStrategy, RoutingStrategy
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata
from langdeep.core.registry.model_registry import ModelConfig, model_registry

from conftest import clean_registries


def setup_function():
    clean_registries()
    _register_test_agents()


def _register_test_agents():
    agent_registry.register("code_agent", lambda: object(), AgentMetadata(
        name="code_agent", description="Writes code",
        routing_keywords=["write", "code", "script", "create", "implement"],
    ))
    agent_registry.register("chat_agent", lambda: object(), AgentMetadata(
        name="chat_agent", description="General chat",
        routing_keywords=["hello", "hi", "chat", "talk", "你好"],
    ))
    agent_registry.register("search_agent", lambda: object(), AgentMetadata(
        name="search_agent", description="Searches web",
        routing_keywords=["search", "新闻", "web", "find"],
    ))
    agent_registry.register("data_agent", lambda: object(), AgentMetadata(
        name="data_agent", description="Analyzes data",
        routing_keywords=["analyze", "分析", "统计", "data", "report"],
    ))


def _make_agents():
    """Build available_agents list in the format the router expects."""
    return [
        {"name": name}
        for name in agent_registry.list_agents()
    ]


router = KeywordRoutingStrategy()


# ── ASCII word-boundary tests ─────────────────────────────────────────

def test_exact_match():
    assert router.route("write", _make_agents()) == "code_agent"


def test_word_boundary_no_false_match():
    """'encode' should NOT match keyword 'code' (word boundary)."""
    result = router.route("encode the data", _make_agents())
    assert result != "code_agent", "encode should not match code"


def test_word_boundary_match():
    """'write a script' should match both 'write' and 'script' → first match."""
    result = router.route("write a script", _make_agents())
    assert result == "code_agent"


def test_search_keyword():
    result = router.route("search the web", _make_agents())
    assert result == "search_agent"


def test_analyze_keyword():
    result = router.route("analyze this data", _make_agents())
    assert result == "data_agent"


def test_chat_keyword():
    result = router.route("hello world", _make_agents())
    assert result == "chat_agent"


def test_partial_word_no_match():
    """'written' should NOT match 'write'."""
    result = router.route("written document", _make_agents())
    assert result != "code_agent"


# ── CJK tests ─────────────────────────────────────────────────────────

def test_chinese_keyword_match():
    result = router.route("搜索最近的新闻", _make_agents())
    assert result == "search_agent"


def test_chinese_analysis():
    result = router.route("分析销售数据", _make_agents())
    assert result == "data_agent"


def test_chinese_greeting():
    result = router.route("你好，今天天气怎么样", _make_agents())
    assert result == "chat_agent"


def test_single_char_cjk_no_false_match():
    """Single CJK characters should not cause false positives."""
    # '写' is not in any keyword list, but even if it were short keywords (<2) are rejected
    assert router.route("写小说", _make_agents()) is None


# ── No match → None ───────────────────────────────────────────────────

def test_no_match_returns_none():
    result = router.route("something completely unrelated", _make_agents())
    assert result is None


def test_empty_input():
    result = router.route("", _make_agents())
    assert result is None


def test_keywords_len_1_rejected():
    """Single-char keywords are rejected by _match_keyword."""
    # Direct test of the static method
    assert router._match_keyword("a b c", "a") is False
    assert router._match_keyword("x y z", "x") is False


def test_default_router_uses_response_cache_for_llm_path():
    class NoFastRoute(RoutingStrategy):
        def route(self, user_input, available_agents):
            return None

    class CountingRouterLLM(BaseChatModel):
        calls: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            tool_call = {
                "name": "route_to_node",
                "args": {"next_node": "chat_agent"},
                "id": "route_1",
            }
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="", tool_calls=[tool_call])
            )])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "counting-router"

    llm = CountingRouterLLM()
    model_registry.register("router_model", ModelConfig(provider="mock", model_name="router"))
    model_registry.set_model_instance("router_model", llm)
    model_registry.enable_response_cache(ttl=300, max_entries=10)

    default_router = DefaultRouter(
        "router_model",
        routing_strategy=NoFastRoute(),
        valid_targets=["chat_agent", "planner", "end"],
    )
    state = {"messages": [HumanMessage(content="unmatched request")]}
    available = [{"name": "chat_agent", "description": "General chat"}]

    first = default_router.route(state, available)
    second = default_router.route(state, available)

    assert first["next"] == "chat_agent"
    assert second["next"] == "chat_agent"
    assert llm.calls == 1
