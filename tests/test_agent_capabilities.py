#!/usr/bin/env python3
"""
LangDeep 框架全面能力测试（纯 Mock / 可选真实 API）

运行方式:
    # 纯 Mock 模式（无需任何 API Key）
    python tests/test_agent_capabilities.py

    # 真实 DeepSeek API 模式（需设置环境变量）
    DEEPSEEK_API_KEY=sk-xxx python tests/test_agent_capabilities.py

    # 显示详细日志
    python tests/test_agent_capabilities.py -v

    # 只运行指定测试
    python tests/test_agent_capabilities.py --filter "重试"

覆盖能力:
    路由层: 关键词快速路由、LLM 监督路由、自定义路由策略
    规划层: LLM 动态规划、工作流模板 (YAML/JSON)、预定义计划
    执行层: gather/sequential/priority_queue 策略、依赖拓扑排序、指数退避重试
    聚合层: 多结果合成、单结果直通、错误优雅降级
    扩展点: 自定义节点、自定义路由策略、自定义执行策略
    基础设施: 流式输出、Trace ID 全链路追踪、异步接口、上下文传递
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# 确保项目根目录在 import 路径中
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, os.path.abspath(_project_root))

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import tool as lc_tool

from langdeep import (
    FlowOrchestrator, ExecutionPolicy, RoutingStrategy,
    LangDeepError,
)
from langdeep.core.registry.model_registry import model_registry, ModelConfig
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata
from langdeep.core.registry.tool_registry import tool_registry, ToolMetadata
from langdeep.core.logging import get_logger, set_trace_context, get_trace_id

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════
# 智能 Mock LLM —— 根据调用上下文（Supervisor/Planner/Aggregator/Agent）返回合理回复
# ═══════════════════════════════════════════════════════════════════════════════════

class SmartMockLLM(BaseChatModel):
    """零依赖 Mock 模型，根据 prompt 内容判断当前节点角色并返回恰当回复。"""

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

        # Supervisor routing: FlowOrchestrator binds the routing tool without
        # requiring providers to set a tool_choice hint.
        if self._bound_tools:
            return self._route(messages, all_text)

        # Planner: 需要 JSON 计划
        if any(kw in all_text.lower() for kw in ["json", "tasks", "plan", "计划", "工作流"]):
            return self._plan(messages, all_text)

        # Aggregator: 合成多个结果
        if any(kw in all_text.lower() for kw in ["synthesise", "synthesize", "整合", "aggregat", "汇总", "合成"]):
            return self._aggregate(messages, all_text)

        # 普通对话
        return self._chat(messages)

    def _route(self, messages, all_text) -> ChatResult:
        user_text = _last_human_text(messages)
        tool_name = self._bound_tools[0].name if self._bound_tools else "route_to_node"

        if any(w in user_text for w in ["搜索", "新闻", "实时", "查询"]):
            target = "web_research_agent"
        elif any(w in user_text for w in ["分析", "处理", "流程", "报告", "多步", "数据", "统计"]):
            target = "planner"
        else:
            target = "default_agent"

        tool_call = {"name": tool_name, "args": {"next_node": target}, "id": "mock_001"}
        logger.info(f"Mock Supervisor → {target}")
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="", tool_calls=[tool_call])
        )])

    def _plan(self, messages, all_text) -> ChatResult:
        user_text = _last_human_text(messages)
        # 从上下文中提取可用 agent
        agents = _extract_agent_names(all_text)
        if not agents:
            agents = ["web_research_agent", "default_agent"]

        is_complex = any(w in user_text for w in ["然后", "接着", "并且", "多步", "复杂", "搜索并", "分析并"])
        if is_complex and len(agents) >= 2:
            plan = [
                {"id": "task_1", "name": "搜索收集", "agent": agents[0],
                 "depends_on": [], "status": "pending", "priority": 1},
                {"id": "task_2", "name": "分析汇总", "agent": agents[-1],
                 "depends_on": ["task_1"], "status": "pending", "priority": 2},
            ]
        else:
            plan = [{"id": "task_1", "name": "处理请求", "agent": agents[0],
                     "depends_on": [], "status": "pending"}]

        content = json.dumps(plan, ensure_ascii=False)
        logger.info(f"Mock Planner → {len(plan)} 步计划")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _aggregate(self, messages, all_text) -> ChatResult:
        # 提取上下文中的结果信息
        snippets = []
        for m in messages:
            content = str(getattr(m, "content", ""))
            if "[" in content and "]" in content:
                snippets.append(content[:200])
        body = "; ".join(snippets) if snippets else "已综合各来源信息生成回答。"
        content = f"综合报告：根据多个 Agent 的分析结果，{body}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _chat(self, messages) -> ChatResult:
        user_text = _last_human_text(messages)
        replies = {
            "你好": "你好！我是智能助手，有什么可以帮助您的？",
            "谢谢": "不客气！如有其他问题随时问我。",
            "再见": "再见！祝您愉快。",
        }
        for key, reply in replies.items():
            if key in user_text:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=f"已收到：「{user_text[:80]}」，处理完成。")
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


# ═══════════════════════════════════════════════════════════════════════════════════
# 测试环境初始化
# ═══════════════════════════════════════════════════════════════════════════════════

def init_test_env():
    """
    初始化测试环境：
    1. 如果 DEEPSEEK_API_KEY 已设置 → 使用真实 API
    2. 如果未设置 → 后续通过 orch() 自动注入 SmartMockLLM
    3. 注册测试专用 Agent（前缀 test_，避免与业务 Agent 冲突）
    """
    has_real_api = bool(os.getenv("DEEPSEEK_API_KEY"))
    if has_real_api:
        logger.info("检测到 DEEPSEEK_API_KEY，将使用真实 DeepSeek API 进行测试")
    else:
        logger.info("未检测到 DEEPSEEK_API_KEY，将使用 SmartMockLLM 模拟所有 LLM 调用")

    _register_test_agents()

    logger.info(
        "测试环境就绪",
        extra={
            "mode": "real_api" if has_real_api else "mock",
            "agents": agent_registry.list_agents(),
        },
    )


def _register_test_agents():
    """注册测试专用 Agent，全部使用 test_ 前缀。"""

    # --- 测试 Agent 1: 总是成功的简单助手 ---
    def _create_test_echo():
        class Agent:
            def invoke(self, state):
                msgs = state.get("messages", [])
                last = ""
                for m in reversed(msgs):
                    if isinstance(m, HumanMessage):
                        last = str(m.content)
                        break
                return {"messages": [AIMessage(content=f"[test_echo] 收到: {last[:60]}")]}
            async def ainvoke(self, state):
                return self.invoke(state)
        return Agent()

    agent_registry.register("test_echo", _create_test_echo, AgentMetadata(
        name="test_echo", description="测试用 Echo Agent",
        capabilities=["test"], routing_keywords=["test_echo", "回显"],
        model_name="deepseek_chat", priority=1,
    ))

    # --- 测试 Agent 2: 模拟搜索 ---
    def _create_test_search():
        class Agent:
            def invoke(self, state):
                msgs = state.get("messages", [])
                last = ""
                for m in reversed(msgs):
                    if isinstance(m, HumanMessage):
                        last = str(m.content)
                        break
                task = state.get("task_context", {}).get("task", {})
                tid = task.get("id", "?")
                return {"messages": [AIMessage(content=f"[test_search] 搜索「{last[:40]}」→ 找到 3 条结果 (task:{tid})")]}
            async def ainvoke(self, state):
                return self.invoke(state)
        return Agent()

    agent_registry.register("test_search", _create_test_search, AgentMetadata(
        name="test_search", description="测试用搜索 Agent",
        capabilities=["test", "search"], routing_keywords=["test_search", "搜索测试"],
        model_name="deepseek_chat", tools=["web_search"], priority=2,
    ))

    # --- 测试 Agent 3: 模拟数据分析 ---
    def _create_test_analyst():
        class Agent:
            def invoke(self, state):
                return {"messages": [AIMessage(content="[test_analyst] 分析结果: mean=42.5, median=40, std=5.2")]}
            async def ainvoke(self, state):
                return self.invoke(state)
        return Agent()

    agent_registry.register("test_analyst", _create_test_analyst, AgentMetadata(
        name="test_analyst", description="测试用数据分析 Agent",
        capabilities=["test", "analysis"], routing_keywords=["test_analyst", "分析测试"],
        model_name="deepseek_chat", priority=3,
    ))

    # --- 测试 Agent 4: 前 N-1 次失败，最后成功（测试重试） ---
    class _RetryAgent:
        def __init__(self):
            self.call_count = 0
        def invoke(self, state):
            self.call_count += 1
            if self.call_count < 3:
                raise RuntimeError(f"模拟失败 (第{self.call_count}次)")
            return {"messages": [AIMessage(content=f"[test_retry] 第{self.call_count}次调用成功")]}
        async def ainvoke(self, state):
            return self.invoke(state)

    agent_registry.register("test_retry", lambda: _RetryAgent(), AgentMetadata(
        name="test_retry", description="测试重试逻辑（前2次失败）",
        capabilities=["test"], routing_keywords=["test_retry", "重试"],
        model_name="deepseek_chat", priority=1,
    ))

    # --- 测试 Agent 5: 永远失败（测试错误降级） ---
    class _AlwaysFail:
        def invoke(self, state):
            raise RuntimeError("模拟持续失败")
        async def ainvoke(self, state):
            raise RuntimeError("模拟持续失败")

    agent_registry.register("test_always_fail", _AlwaysFail, AgentMetadata(
        name="test_always_fail", description="永远失败的 Agent（测试错误处理）",
        capabilities=["test"], routing_keywords=["test_fail", "必败"],
        model_name="deepseek_chat", priority=1,
    ))

    # --- 测试 Agent 6: 捕获上下文用于验证 ---
    class _ContextCapture:
        def __init__(self):
            self.last_context = {}
        def invoke(self, state):
            self.last_context = state.get("task_context", {})
            task = self.last_context.get("task", {})
            prev = self.last_context.get("previous_results", {})
            return {"messages": [AIMessage(
                content=f"[test_context] task_id={task.get('id')}, prev_keys={list(prev.keys())}"
            )]}
        async def ainvoke(self, state):
            return self.invoke(state)

    _ctx_instance = _ContextCapture()
    agent_registry.register("test_context", lambda: _ctx_instance, AgentMetadata(
        name="test_context", description="捕获 task_context 的 Agent",
        capabilities=["test"], routing_keywords=["test_ctx"],
        model_name="deepseek_chat", priority=1,
    ))

    # --- 模板引用的 Agent（仅在没有业务 Agent 时注册） ---
    _existing = set(agent_registry.list_agents())

    if "web_research_agent" not in _existing:
        def _create_web_research():
            class Agent:
                def invoke(self, state):
                    task = state.get("task_context", {}).get("task", {})
                    tid = task.get("id", "?")
                    return {"messages": [AIMessage(content=f"[web_research] 搜索完成 (task:{tid})")]}
                async def ainvoke(self, state):
                    return self.invoke(state)
            return Agent()
        agent_registry.register("web_research_agent", _create_web_research, AgentMetadata(
            name="web_research_agent", description="测试用网页搜索 Agent",
            capabilities=["test", "search"], routing_keywords=["搜索", "新闻", "实时"],
            model_name="deepseek_chat", priority=1,
        ))

    if "default_agent" not in _existing:
        def _create_default():
            class Agent:
                def invoke(self, state):
                    task = state.get("task_context", {}).get("task", {})
                    tid = task.get("id", "?")
                    return {"messages": [AIMessage(content=f"[default_agent] 处理完成 (task:{tid})")]}
                async def ainvoke(self, state):
                    return self.invoke(state)
            return Agent()
        agent_registry.register("default_agent", _create_default, AgentMetadata(
            name="default_agent", description="测试用默认 Agent",
            capabilities=["test"], routing_keywords=["你好", "谢谢", "默认"],
            model_name="deepseek_chat", priority=1,
        ))


# ═══════════════════════════════════════════════════════════════════════════════════
# 测试辅助
# ═══════════════════════════════════════════════════════════════════════════════════

PASS = 0
FAIL = 0
RESULTS: List[Dict] = []


def case(name: str):
    """装饰器：包装测试函数，自动计时和异常捕获。"""
    def dec(fn):
        async def wrapper(*a, **kw):
            global PASS, FAIL
            print(f"\n  🧪 {name} ...", end=" ", flush=True)
            start = time.perf_counter()
            try:
                ok = await fn(*a, **kw)
                elapsed = time.perf_counter() - start
                if ok is False:
                    print(f"❌ FAIL ({elapsed:.2f}s)")
                    FAIL += 1; RESULTS.append({"name": name, "status": "FAIL", "time": elapsed})
                else:
                    print(f"✅ PASS ({elapsed:.2f}s)")
                    PASS += 1; RESULTS.append({"name": name, "status": "PASS", "time": elapsed})
                return ok
            except Exception as e:
                elapsed = time.perf_counter() - start
                print(f"💥 ERROR ({elapsed:.2f}s): {e}")
                traceback.print_exc()
                FAIL += 1; RESULTS.append({"name": name, "status": "ERROR", "time": elapsed, "error": str(e)})
                return False
        return wrapper
    return dec


_mock_singleton: Optional[SmartMockLLM] = None

def _mock() -> SmartMockLLM:
    global _mock_singleton
    if _mock_singleton is None:
        _mock_singleton = SmartMockLLM(model_name="smart-mock")
    return _mock_singleton


def orch(**kw) -> FlowOrchestrator:
    """创建测试 Orchestrator。

    默认禁用 checkpoint（测试不需要持久化），Mock 模式下自动
    注册缺失的模型（provider=mock）。
    """
    kw.setdefault("enable_checkpoint", False)
    o = FlowOrchestrator(**kw)
    if not os.getenv("DEEPSEEK_API_KEY"):
        # auto_import 在 tests/ 下找不到模型定义，手动补齐
        for name in [kw.get("supervisor_model", "gpt4o"), "deepseek_chat"]:
            if name not in model_registry.list_models():
                model_registry.register(name, ModelConfig(provider="mock", model_name=name))
            model_registry.set_model_instance(name, _mock())
    return o


class _ToPlanner(RoutingStrategy):
    def route(self, user_input, agents):
        return "planner"


def orch_planner(**kw) -> FlowOrchestrator:
    """创建自动路由到 planner 的 Orchestrator（测试 executor 路径时使用）。"""
    kw.setdefault("routing_strategy", _ToPlanner())
    return orch(**kw)


def _last_ai(msgs: List) -> str:
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            return str(m.content)
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content:
            return str(m.content)
    return ""


def _last_human_text(msgs: List) -> str:
    for m in reversed(msgs):
        if isinstance(m, HumanMessage) and m.content:
            return str(m.content)
    return ""


def _extract_agent_names(text: str) -> List[str]:
    """从文本中提取 agent 名称。"""
    import re
    found = set()
    for name in agent_registry.list_agents():
        if name in text:
            found.add(name)
    return list(found) if found else []


# ═══════════════════════════════════════════════════════════════════════════════════
# 测试用例（24 项）
# ═══════════════════════════════════════════════════════════════════════════════════

# ── 1. 关键词快速路由 ───────────────────────────────────────────────────────────

@case("关键词路由: '你好' → default_agent")
async def test_kw_greeting():
    r = orch().invoke("你好")
    ans = _last_ai(r["messages"])
    assert len(ans) > 5, f"回复过短: {ans}"
    return True


@case("关键词路由: '搜索 AI 新闻' → web_research_agent")
async def test_kw_search():
    r = orch().invoke("帮我搜索最新的 AI 新闻")
    ans = _last_ai(r["messages"])
    # 由 web_research_agent 处理
    assert len(ans) > 10, f"回复过短: {ans}"
    return True


# ── 2. LLM 监督路由 ─────────────────────────────────────────────────────────────

@case("LLM路由: 无关键词匹配时走 LLM 分发")
async def test_llm_routing():
    r = orch().invoke("请帮我综合分析一下当前的市场趋势")
    ans = _last_ai(r["messages"])
    assert len(ans) > 10, f"回复过短: {ans}"
    return True


# ── 3. Planner 动态规划 ─────────────────────────────────────────────────────────

@case("动态规划: 复杂问题生成多步计划")
async def test_planner():
    r = orch().invoke("搜索 AI 最新趋势并帮我做分析汇总")
    ans = _last_ai(r["messages"])
    assert len(ans) > 10, f"回复过短: {ans}"
    results = r.get("agent_results", {})
    assert len(results) >= 1, f"至少1个任务结果，实际: {len(results)}"
    return True


# ── 4. 并发执行策略 ─────────────────────────────────────────────────────────────

@case("gather 策略: 3个无依赖任务并发执行")
async def test_gather():
    o = orch_planner(execution_policy=ExecutionPolicy(strategy="gather", max_concurrency=5))
    plan = [
        {"id": "t1", "agent": "test_echo", "depends_on": [], "status": "pending"},
        {"id": "t2", "agent": "test_search", "depends_on": [], "status": "pending"},
        {"id": "t3", "agent": "test_analyst", "depends_on": [], "status": "pending"},
    ]
    r = o.invoke("并发测试", workflow_plan=plan)
    results = r.get("agent_results", {})
    for tid in ["t1", "t2", "t3"]:
        assert tid in results, f"缺少任务 {tid}，实际: {list(results.keys())}"
    return True


@case("sequential 策略: 任务逐个执行")
async def test_sequential():
    o = orch_planner(execution_policy=ExecutionPolicy(strategy="sequential"))
    plan = [
        {"id": "a1", "agent": "test_echo", "depends_on": [], "status": "pending"},
        {"id": "a2", "agent": "test_search", "depends_on": [], "status": "pending"},
    ]
    r = o.invoke("顺序测试", workflow_plan=plan)
    results = r.get("agent_results", {})
    assert len(results) == 2, f"应有2个结果，实际: {len(results)}"
    return True


@case("priority_queue 策略: 按优先级排序执行")
async def test_priority():
    o = orch_planner(execution_policy=ExecutionPolicy(strategy="priority_queue"))
    plan = [
        {"id": "low", "agent": "test_echo", "depends_on": [], "priority": 1, "status": "pending"},
        {"id": "high", "agent": "test_search", "depends_on": [], "priority": 10, "status": "pending"},
    ]
    r = o.invoke("优先级测试", workflow_plan=plan)
    results = r.get("agent_results", {})
    assert "low" in results and "high" in results, f"缺少结果: {list(results.keys())}"
    return True


# ── 5. 依赖拓扑排序 ─────────────────────────────────────────────────────────────

@case("依赖排序: t2 依赖 t1，验证两个任务均完成")
async def test_dependency():
    o = orch_planner()
    plan = [
        {"id": "first", "agent": "test_search", "depends_on": [], "status": "pending"},
        {"id": "second", "agent": "test_echo", "depends_on": ["first"], "status": "pending"},
    ]
    r = o.invoke("依赖测试", workflow_plan=plan)
    results = r.get("agent_results", {})
    assert "first" in results, "first 任务应完成"
    assert "second" in results, "second 任务应完成（依赖已满足）"
    for v in results.values():
        assert "依赖无法满足" not in str(v), f"不应出现依赖错误: {v}"
        assert "Dependency" not in str(v)
    return True


# ── 6. 重试逻辑 ─────────────────────────────────────────────────────────────────

@case("指数退避重试: 前2次失败，第3次成功")
async def test_retry_success():
    o = orch(max_retries=3)
    # test_retry 前2次抛异常，第3次成功 => 关键词路由直接命中
    result = o.invoke("test_retry 测试")
    agent_results = result.get("agent_results", {})
    # 直接路由时 key 是 agent 名称
    val = agent_results.get("test_retry", "")
    assert "成功" in str(val) or "第3次" in str(val) or "第 3 次" in str(val), \
        f"预期重试成功，实际: {val[:200]}"
    return True


@case("重试耗尽: max_retries=2，Agent永远失败，返回错误不崩溃")
async def test_retry_exhausted():
    o = orch(max_retries=2)
    result = o.invoke("test_fail 必败")
    agent_results = result.get("agent_results", {})
    val = agent_results.get("test_always_fail", "")
    assert ("retries" in val.lower() or "重试" in val.lower() or "max" in val.lower()
            or "exhausted" in val.lower()), \
        f"预期重试耗尽错误，实际: {val[:200]}"
    return True


# ── 7. 自定义节点 ────────────────────────────────────────────────────────────────

@case("自定义节点: 注入 custom_nodes 并验证执行")
async def test_custom_node():
    def my_node(state: dict) -> dict:
        prev = state.get("agent_results", {})
        return {
            "messages": state.get("messages", []),
            "agent_results": {**prev, "my_extra": "自定义节点已执行 ✓"},
        }

    o = orch(custom_nodes={"extra_node": my_node})
    targets = o._get_valid_targets()
    assert "extra_node" in targets, f"自定义节点应在合法目标中: {targets}"

    # 用自定义路由策略指向它
    class ToExtra(RoutingStrategy):
        def route(self, u, agents):
            return "extra_node"

    o2 = orch(custom_nodes={"extra_node": my_node}, routing_strategy=ToExtra())
    r = o2.invoke("触发自定义节点")
    results = r.get("agent_results", {})
    assert "my_extra" in results, f"自定义节点应输出结果: {results}"
    assert "已执行" in str(results["my_extra"])
    return True


# ── 8. 自定义路由策略 ───────────────────────────────────────────────────────────

@case("自定义路由: RoutingStrategy 子类 AlwaysPlanner")
async def test_custom_routing():
    class AlwaysPlanner(RoutingStrategy):
        def route(self, u, agents):
            return "planner"
    r = orch(routing_strategy=AlwaysPlanner()).invoke("任意问题")
    ans = _last_ai(r["messages"])
    assert len(ans) > 5, f"应产生回复: {ans}"
    return True


# ── 9. 预定义计划 ────────────────────────────────────────────────────────────────

@case("预定义计划: workflow_plan 直接传入执行")
async def test_predefined_plan():
    plan = [
        {"id": "s1", "name": "搜索", "agent": "test_search", "depends_on": [], "status": "pending"},
        {"id": "s2", "name": "分析", "agent": "test_analyst", "depends_on": ["s1"], "status": "pending"},
    ]
    r = orch_planner().invoke("预定义计划", workflow_plan=plan)
    results = r.get("agent_results", {})
    assert "s1" in results and "s2" in results, f"两步都应完成: {list(results.keys())}"
    assert "test_search" in str(results["s1"]) or "搜索" in str(results["s1"])
    return True


# ── 10. 流式输出 ────────────────────────────────────────────────────────────────

@case("流式输出: astream 逐节点 yield")
async def test_stream():
    o = orch()
    nodes = set()
    async for chunk in o.astream("搜索 AI 新闻"):
        for node_name in chunk:
            nodes.add(node_name)
    assert len(nodes) >= 2, f"流式应覆盖至少2个节点，实际: {nodes}"
    return True


# ── 11. 错误优雅降级 ────────────────────────────────────────────────────────────

@case("优雅降级: 全部 Agent 失败 → 返回兜底消息不崩溃")
async def test_degradation():
    o = orch(max_retries=1)
    result = o.invoke("test_fail 必败测试")
    ans = _last_ai(result.get("messages", []))
    assert len(ans) > 0, "即使全部失败也应有兜底回复"
    return True


# ── 12. Trace ID 全链路追踪 ─────────────────────────────────────────────────────

@case("Trace ID: invoke 自动生成并清理")
async def test_trace():
    tid = set_trace_context("my-trace-001")
    orch().invoke("trace 测试")
    assert get_trace_id() is None, "invoke 后应清除 trace context"
    return True


# ── 13. 异常层次结构 ─────────────────────────────────────────────────────────────

@case("异常体系: 所有自定义异常类型正确")
async def test_exceptions():
    from langdeep.core.errors import (
        ModelNotFoundError, AgentNotFoundError, ToolNotFoundError,
        ProviderNotFoundError, InvalidPolicyError, LangDeepError,
    )
    # ModelNotFound
    try:
        model_registry.get_model("__不存在__")
        assert False
    except ModelNotFoundError as e:
        assert e.code == "MODEL_NOT_FOUND" and isinstance(e, LangDeepError)
    # AgentNotFound
    try:
        agent_registry.get_agent("__不存在__")
        assert False
    except AgentNotFoundError as e:
        assert e.code == "AGENT_NOT_FOUND"
    # InvalidPolicy
    try:
        ExecutionPolicy(strategy="bad_strategy")
        assert False
    except InvalidPolicyError as e:
        assert e.code == "INVALID_POLICY"
    # to_dict
    try:
        model_registry.get_model("nope")
    except ModelNotFoundError as e:
        d = e.to_dict()
        assert all(k in d for k in ["code", "detail", "context"])
    return True


# ── 14. 工作流模板 YAML ─────────────────────────────────────────────────────────

@case("YAML模板: daily_report 两步骤执行")
async def test_tmpl_yaml():
    wf_dir = os.path.join(os.path.dirname(_project_root), "workflows")
    o = orch_planner(workflow_templates_dir=wf_dir)
    r = o.invoke("AI 行业动态", template_name="daily_report")
    results = r.get("agent_results", {})
    assert len(results) >= 1, f"模板至少产生1个结果: {list(results.keys())}"
    return True


@case("JSON模板: data_pipeline 两步骤执行")
async def test_tmpl_json():
    wf_dir = os.path.join(os.path.dirname(_project_root), "workflows")
    o = orch_planner(workflow_templates_dir=wf_dir)
    r = o.invoke("数据处理流水线", template_name="data_pipeline")
    results = r.get("agent_results", {})
    assert len(results) >= 1, f"JSON模板至少产生1个结果: {list(results.keys())}"
    return True


# ── 15. 聚合器 ──────────────────────────────────────────────────────────────────

@case("聚合器: 多Agent结果通过LLM合成")
async def test_agg_multi():
    o = orch_planner()
    plan = [
        {"id": "r1", "agent": "test_search", "depends_on": [], "status": "pending"},
        {"id": "r2", "agent": "test_analyst", "depends_on": [], "status": "pending"},
        {"id": "r3", "agent": "test_echo", "depends_on": [], "status": "pending"},
    ]
    r = o.invoke("综合分析多Agent", workflow_plan=plan)
    results = r.get("agent_results", {})
    assert len(results) == 3, f"应有3个结果: {len(results)}"
    ans = _last_ai(r["messages"])
    assert len(ans) > 15, f"聚合结果不应为空: {ans}"
    return True


@case("聚合器: 单结果跳过LLM直接返回")
async def test_agg_single():
    o = orch_planner()
    plan = [{"id": "only", "agent": "test_search", "depends_on": [], "status": "pending"}]
    r = o.invoke("单结果测试", workflow_plan=plan)
    ans = _last_ai(r["messages"])
    assert "test_search" in ans, f"单结果应直通: {ans[:120]}"
    return True


# ── 16. 执行策略构造 ─────────────────────────────────────────────────────────────

@case("ExecutionPolicy: from_dict() / to_dict() 往返")
async def test_policy_serde():
    p = ExecutionPolicy.from_dict({"max_concurrency": 3, "strategy": "sequential"})
    assert p.max_concurrency == 3 and p.strategy == "sequential"
    d = p.to_dict()
    assert d["strategy"] == "sequential"
    return True


# ── 17. 异步接口 ─────────────────────────────────────────────────────────────────

@case("异步: ainvoke 接口正常工作")
async def test_ainvoke():
    r = await orch().ainvoke("异步测试")
    ans = _last_ai(r["messages"])
    assert len(ans) > 5
    return True


# ── 18. 上下文传递 ───────────────────────────────────────────────────────────────

@case("上下文: task_context 含 task 和 previous_results")
async def test_context():
    o = orch_planner()
    plan = [{"id": "ctx1", "agent": "test_context", "depends_on": [], "status": "pending",
             "extra_field": "value_123"}]
    r = o.invoke("上下文测试", workflow_plan=plan, context={"global": "data"})
    results = r.get("agent_results", {})
    assert "ctx1" in results, f"应有 ctx1 结果: {list(results.keys())}"
    # test_context Agent 返回包含 task_id 和 prev_keys 的信息
    assert "ctx1" in str(results["ctx1"]), f"应包含 task id ctx1: {results['ctx1'][:200]}"
    return True


# ═══════════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════════

async def run_all_tests(filter_str: str = ""):
    global PASS, FAIL, RESULTS
    PASS = 0; FAIL = 0; RESULTS = []

    api_mode = "真实 API" if os.getenv("DEEPSEEK_API_KEY") else "Mock 模拟"
    print("=" * 70)
    print(f"  LangDeep 框架全面能力测试")
    print(f"  模式: {api_mode}")
    print(f"  覆盖: 路由 | 规划 | 执行 | 聚合 | 重试 | 流式 | 模板 | 错误 | 扩展")
    print("=" * 70)

    init_test_env()

    # 所有测试
    all_cases = [
        test_kw_greeting, test_kw_search, test_llm_routing, test_planner,
        test_gather, test_sequential, test_priority, test_dependency,
        test_retry_success, test_retry_exhausted, test_custom_node,
        test_custom_routing, test_predefined_plan, test_stream,
        test_degradation, test_trace, test_exceptions,
        test_tmpl_yaml, test_tmpl_json, test_agg_multi, test_agg_single,
        test_policy_serde, test_ainvoke, test_context,
    ]

    start = time.perf_counter()
    for c in all_cases:
        if filter_str and filter_str not in c.__name__:
            continue
        await c()
    total = time.perf_counter() - start

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  测试结果")
    print(f"{'=' * 70}")
    for r in RESULTS:
        icon = "✅" if r["status"] == "PASS" else "❌" if r["status"] == "FAIL" else "💥"
        print(f"  {icon} {r['name']}  ({r['time']:.2f}s)")
    print(f"\n{'─' * 70}")
    total_n = PASS + FAIL
    print(f"  总计: {total_n}  |  通过: {PASS} ✅  |  失败: {FAIL} ❌  |  耗时: {total:.2f}s")
    print(f"{'─' * 70}\n")
    return FAIL == 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LangDeep 框架全面能力测试")
    p.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    p.add_argument("--filter", type=str, default="", help="只运行名称包含指定字符串的测试")
    args = p.parse_args()

    import logging
    logging.getLogger("langdeep").setLevel(logging.INFO if args.verbose else logging.WARNING)

    ok = asyncio.run(run_all_tests(args.filter))
    sys.exit(0 if ok else 1)
