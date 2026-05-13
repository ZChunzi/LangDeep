"""Shared test fixtures and helpers for all LangDeep unit tests.

Usage:
    from tests.conftest import SmartMockLLM, clean_registries, orch, _last_ai, _last_human
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, os.path.abspath(_project_root))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from langdeep import FlowOrchestrator, ExecutionPolicy, RoutingStrategy
from langdeep.core.registry.model_registry import model_registry, provider_registry, ModelConfig
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata
from langdeep.core.registry.tool_registry import tool_registry
from langdeep.core.memory.registry import memory_registry
from langdeep.core.cache.registry import cache_registry
from langdeep.core.im.registry import im_channel_registry
from langdeep.core.sandbox.registry import sandbox_registry
from langdeep.core.secrets.manager import secrets_manager


def clean_registries():
    """Reset all singleton registries to pristine state (for test isolation)."""
    agent_registry.reset()
    tool_registry.reset()
    model_registry.reset()
    provider_registry.reset()
    memory_registry.clear()
    cache_registry.clear()
    im_channel_registry.clear()
    sandbox_registry.clear()  # clear() re-registers the built-in subprocess backend
    secrets_manager.clear()


def populate_minimal_registries():
    """Register the minimal set of models, tools, and agents needed for unit tests."""
    from langdeep.core.registry.tool_registry import ToolMetadata
    from langchain_core.tools import tool as lc_tool

    # Register mock model
    if "gpt4o" not in model_registry.list_models():
        model_registry.register("gpt4o", ModelConfig(provider="mock", model_name="gpt4o"))
        model_registry.set_model_instance("gpt4o", SmartMockLLM(model_name="gpt4o"))

    if "deepseek_chat" not in model_registry.list_models():
        model_registry.register("deepseek_chat", ModelConfig(provider="mock", model_name="deepseek-chat"))
        model_registry.set_model_instance("deepseek_chat", SmartMockLLM(model_name="deepseek-chat"))

    # Register a test tool
    if "test_tool" not in tool_registry.list_tools():
        @lc_tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"result for {query}"
        tool_registry.register(test_tool, ToolMetadata(name="test_tool", description="A test tool"))

    # Register test agents
    _register_default_test_agents()


def _register_default_test_agents():
    """Register agents used across multiple unit tests."""

    def _create_echo():
        class Agent:
            def invoke(self, state):
                last = _last_human_text(state.get("messages", []))
                return {"messages": [AIMessage(content=f"[echo] {last[:60]}")]}
            async def ainvoke(self, state):
                return self.invoke(state)
        return Agent()

    if "test_echo" not in agent_registry.list_agents():
        agent_registry.register("test_echo", _create_echo, AgentMetadata(
            name="test_echo", description="Echo agent",
            capabilities=["test"], routing_keywords=["echo", "回显", "你好", "test"],
            model_name="gpt4o", priority=1,
        ))

    def _create_web():
        class Agent:
            def invoke(self, state):
                return {"messages": [AIMessage(content="[web] search done")]}
            async def ainvoke(self, state):
                return self.invoke(state)
        return Agent()

    if "web_agent" not in agent_registry.list_agents():
        agent_registry.register("web_agent", _create_web, AgentMetadata(
            name="web_agent", description="Web search agent",
            capabilities=["search"], routing_keywords=["搜索", "search", "web"],
            model_name="gpt4o", priority=1,
        ))


# ═══════════════════════════════════════════════════════════════════════
# SmartMockLLM
# ═══════════════════════════════════════════════════════════════════════

class SmartMockLLM(BaseChatModel):
    """Mock LLM that adapts its response based on call context.

    Detects Supervisor / Planner / Aggregator / Agent roles from the
    prompt content and returns role-appropriate replies.
    """

    model_name: str = "smart-mock"
    temperature: float = 0.7
    _bound_tools: List[Any] = []
    _tool_choice: str = ""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        all_text = " ".join(str(getattr(m, "content", "")) for m in messages)

        if self._bound_tools and self._tool_choice == "required":
            return self._route(messages, all_text)
        if any(kw in all_text.lower() for kw in ["json", "tasks", "plan", "计划", "工作流"]):
            return self._plan(messages, all_text)
        if any(kw in all_text.lower() for kw in ["synthesise", "synthesize", "整合", "aggregat", "汇总", "合成"]):
            return self._aggregate(messages, all_text)
        return self._chat(messages)

    def _route(self, messages, all_text) -> ChatResult:
        user_text = _last_human_text(messages)
        tool_name = self._bound_tools[0].name if self._bound_tools else "route_to_node"
        target = "planner" if any(w in user_text for w in ["分析", "多步", "复杂", "报告"]) else "test_echo"
        tool_call = {"name": tool_name, "args": {"next_node": target}, "id": "mock_001"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tool_call]))])

    def _plan(self, messages, all_text) -> ChatResult:
        agents = [a for a in ["web_agent", "test_echo"] if a in all_text] or ["test_echo"]
        plan = json.dumps([
            {"id": "t1", "name": "step1", "agent": agents[0], "depends_on": [], "status": "pending"},
        ], ensure_ascii=False)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=plan))])

    def _aggregate(self, messages, all_text) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="Synthesised: combined result from all agents."),
        )])

    def _chat(self, messages) -> ChatResult:
        user_text = _last_human_text(messages)
        replies = {
            "你好": "你好！有什么可以帮助您的？",
            "谢谢": "不客气！",
            "再见": "再见！",
        }
        for key, reply in replies.items():
            if key in user_text:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=f"Received: 「{user_text[:80]}」, done."),
        )])

    def bind_tools(self, tools, **kwargs):
        import copy
        new = copy.copy(self)
        new._bound_tools = list(tools)
        new._tool_choice = kwargs.get("tool_choice", "")
        return new

    @property
    def _llm_type(self) -> str:
        return "smart-mock"


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator factory helpers
# ═══════════════════════════════════════════════════════════════════════

_mock_singleton: Optional[SmartMockLLM] = None


def _mock() -> SmartMockLLM:
    global _mock_singleton
    if _mock_singleton is None:
        _mock_singleton = SmartMockLLM(model_name="smart-mock")
    return _mock_singleton


def orch(**kw) -> FlowOrchestrator:
    kw.setdefault("enable_checkpoint", False)
    o = FlowOrchestrator(**kw)
    for name in [kw.get("supervisor_model", "gpt4o"), "deepseek_chat"]:
        if name not in model_registry.list_models():
            model_registry.register(name, ModelConfig(provider="mock", model_name=name))
        model_registry.set_model_instance(name, _mock())
    return o


# ═══════════════════════════════════════════════════════════════════════
# Response extraction helpers
# ═══════════════════════════════════════════════════════════════════════

def _last_ai(msgs: List) -> str:
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content:
            return str(m.content)
    return ""


def _last_human_text(msgs: List) -> str:
    for m in reversed(msgs):
        if isinstance(m, HumanMessage) and m.content:
            return str(m.content)
    return ""


def _build_messages(*contents: str) -> List[BaseMessage]:
    """Build a message sequence: Human, AIMessage(tool_calls), ToolMessage, Human, ..."""
    msgs = []
    for i, c in enumerate(contents):
        if i == 0:
            msgs.append(HumanMessage(content=c))
        elif i == 1 and c == "__tool_call__":
            msgs.append(AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {"query": "x"}, "id": "call_1"}]))
        elif i == 1:
            msgs.append(AIMessage(content=c))
        elif i == 2 and c.startswith("__tool_result__"):
            msgs.append(AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {"query": "x"}, "id": "call_1"}]))
        else:
            msgs.append(AIMessage(content=c))
    return msgs


class ToPlanner(RoutingStrategy):
    """Routing strategy that always routes to 'planner' for executor tests."""
    def route(self, user_input, agents):
        return "planner"
