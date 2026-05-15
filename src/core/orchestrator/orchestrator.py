"""Flow orchestrator — modular, extensible coordinator built on LangGraph."""

import importlib
import inspect
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from ..logging import get_logger, set_trace_context, clear_trace_context
from ..errors import ConfigurationError, LangDeepError, OrchestrationError
from ..registry.agent_registry import agent_registry
from ..execution.execution_policy import ExecutionPolicy

from .router import DefaultRouter, RoutingStrategy
from .planner import Planner, PlanGenerator, TemplateLoader, FallbackPlanGenerator
from .executor import Executor, TaskRunner, _clean_messages
from .aggregator import Aggregator, ResultMerger
from .agent_node import make_agent_node

if TYPE_CHECKING:
    from ..process import ProcessManager
    from ..observability import MetricsCollector

logger = get_logger(__name__)

DEFAULT_COMPONENT_DIRS = ["models", "tools", "agents"]

# ── State definition ─────────────────────────────────────────────────────────────

class OrchestratorState(dict):
    """Typed state used by the LangGraph state graph.

    Because LangGraph needs resolve_parallel_state to handle concurrent updates,
    we keep this as a plain dict and rely on the state graph's reducer annotations.
    """

    pass


def _build_state_schema() -> type:
    """Build the TypedDict schema for the graph."""
    from typing import TypedDict, Annotated

    class Schema(TypedDict, total=False):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        next: str
        current_task: str
        task_context: Dict[str, Any]
        agent_results: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]
        workflow_plan: Optional[List[Dict[str, Any]]]
        memory_session_id: Optional[str]
        memory_start_len: int
        error_count: int
        max_retries: int
        aggregation_done: bool

    return Schema


# ── Orchestrator ─────────────────────────────────────────────────────────────────

class FlowOrchestrator:
    """Modular, extensible workflow orchestrator.

    Extension points (all injectable at construction time):
        - ``routing_strategy`` — custom fast-path routing (subclass ``RoutingStrategy``).
        - ``plan_generator`` — custom plan generation (subclass ``PlanGenerator``).
        - ``task_runner`` — custom task execution/retry logic (subclass ``TaskRunner``).
        - ``result_merger`` — custom result synthesis (subclass ``ResultMerger``).
        - ``custom_nodes`` — dict of ``{name: callable}`` for additional graph nodes.

    Parameters:
        supervisor_model: Name of the registered model for supervisor/planning/aggregation.
        max_retries: Default retry count for agent and task execution.
        enable_checkpoint: Persist graph state via MemorySaver (or pass ``checkpointer``).
        prompt_dir: External directory for prompt markdown files.
        component_dirs: Directories to scan for ``@agent``/``@model``/``@tool`` modules.
        llm_timeout: Timeout in seconds for LLM calls.
        checkpointer: External LangGraph checkpointer instance.
        routing_strategy: Custom ``RoutingStrategy`` instance.
        workflow_templates_dir: Directory with YAML/JSON workflow templates.
        execution_policy: Concurrency and execution strategy control.
        custom_nodes: Dict mapping node names to callables for user-defined graph nodes.
        plan_generator: Custom ``PlanGenerator`` instance.
        task_runner: Custom ``TaskRunner`` instance.
        result_merger: Custom ``ResultMerger`` instance.
    """

    # Graph node names (exposed for reference / custom edge logic)
    NODE_SUPERVISOR = "supervisor"
    NODE_PLANNER = "planner"
    NODE_EXECUTOR = "executor"
    NODE_AGGREGATOR = "aggregator"

    def __init__(
        self,
        supervisor_model: str = "gpt4o",
        max_retries: int = 3,
        enable_checkpoint: bool = True,
        prompt_dir: Optional[str] = None,
        component_dirs: Optional[List[str]] = None,
        llm_timeout: float = 30.0,
        checkpointer=None,
        routing_strategy: Optional[RoutingStrategy] = None,
        workflow_templates_dir: Optional[str] = None,
        execution_policy: Optional[ExecutionPolicy] = None,
        custom_nodes: Optional[Dict[str, Callable]] = None,
        # New extension points
        plan_generator: Optional[PlanGenerator] = None,
        task_runner: Optional[TaskRunner] = None,
        result_merger: Optional[ResultMerger] = None,
        # Memory backend (name registered via @memory decorator)
        memory: Optional[str] = None,
        # Process manager (optional, for long-running workflow lifecycle)
        process_manager: Optional["ProcessManager"] = None,
        strict_component_import: bool = False,
        metrics_collector: Optional["MetricsCollector"] = None,
    ):
        self._supervisor_model = supervisor_model
        self._max_retries = max_retries
        self._llm_timeout = llm_timeout
        self._policy = execution_policy or ExecutionPolicy()
        from ..observability import MetricsCollector
        self._metrics = metrics_collector or MetricsCollector()

        # Prompt loading
        from ..prompt.prompt_loader import MarkdownPromptLoader
        self._prompt_loader = MarkdownPromptLoader(prompt_dir)

        # Checkpointer
        if checkpointer is not None:
            self._checkpointer = checkpointer
        elif enable_checkpoint:
            self._checkpointer = MemorySaver()
        else:
            self._checkpointer = None

        # Memory backend (optional, for conversation persistence)
        self._memory_backend: Optional[Any] = None
        if memory is not None:
            try:
                from ..memory.registry import memory_registry
                self._memory_backend = memory_registry.get_backend(memory)
                logger.info(
                    "Memory backend configured",
                    extra={"memory_backend": memory},
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load memory backend",
                    extra={"memory": memory, "error": str(exc)},
                )

        # Auto-import component modules
        self._auto_import(
            dirs=component_dirs or DEFAULT_COMPONENT_DIRS,
            caller_file=inspect.stack()[1].filename,
            strict=strict_component_import,
        )

        from ..registry.tool_registry import tool_registry
        tool_registry.set_metrics_collector(self._metrics)

        # Cached registrations
        self._cached_agents: Optional[List[Dict]] = None
        self._cached_targets: Optional[List[str]] = None
        self._custom_nodes = custom_nodes or {}

        # Template support
        self._templates: Optional[TemplateLoader] = None
        if workflow_templates_dir:
            self._templates = TemplateLoader(workflow_templates_dir)

        # ── Wire extension points ─────────────────────────────────────────────
        self._router = DefaultRouter(
            model_name=supervisor_model,
            routing_strategy=routing_strategy,
            valid_targets=self._get_valid_targets(),
            metrics_collector=self._metrics,
        )

        self._planner = Planner(
            model_name=supervisor_model,
            plan_generator=plan_generator,
            prompt_loader=self._prompt_loader,
            metrics_collector=self._metrics,
        )

        self._executor = Executor(
            task_runner=task_runner,
            policy=self._policy,
            max_retries=max_retries,
            timeout=llm_timeout,
            metrics_collector=self._metrics,
        )

        self._aggregator = Aggregator(
            model_name=supervisor_model,
            merger=result_merger,
            prompt_loader=self._prompt_loader,
            metrics_collector=self._metrics,
        )

        # Process manager (optional)
        self._process_manager = process_manager

        self._graph = self._build_graph()

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(
        self,
        user_input: Any,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow synchronously and return the final state.

        Accepted inputs:
        - ``str``: normal chat/workflow request.
        - ``BaseMessage``: a single LangChain message.
        - ``Sequence[BaseMessage]``: explicit conversation history.
        - ``{"messages": [...]}``: LangGraph-style state input.
        """
        initial = self._prepare_initial_state(user_input, context, workflow_plan, template_name)
        input_preview = _input_preview(user_input)
        trace_id = set_trace_context()
        started = time.monotonic()
        status = "success"
        self._metrics.counter("orchestrator.invocations", tags={"mode": "sync"})
        logger.info("Orchestrator invoke start", extra={"input_preview": input_preview})
        try:
            result = self._graph.invoke(initial)
            result = self._after_execution(result)
            logger.info("Orchestrator invoke complete")
            return result
        except Exception as exc:
            status = "failure"
            self._metrics.counter("orchestrator.errors", tags={"mode": "sync"})
            logger.error("Orchestrator invoke failed", extra={"error": str(exc)}, exc_info=True)
            raise OrchestrationError(
                "Workflow execution failed",
                context={"user_input": input_preview[:200], "trace_id": trace_id},
                cause=exc,
            ) from exc
        finally:
            self._record_orchestrator_duration(started, mode="sync", status=status)
            clear_trace_context()

    async def ainvoke(
        self,
        user_input: Any,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow asynchronously and return the final state."""
        initial = self._prepare_initial_state(user_input, context, workflow_plan, template_name)
        input_preview = _input_preview(user_input)
        trace_id = set_trace_context()
        started = time.monotonic()
        status = "success"
        self._metrics.counter("orchestrator.invocations", tags={"mode": "async"})
        logger.info("Orchestrator ainvoke start", extra={"input_preview": input_preview})
        try:
            graph_ainvoke = getattr(self._graph, "ainvoke", None)
            if callable(graph_ainvoke) and _should_use_native_async(self._graph):
                result = await graph_ainvoke(initial)
            elif _is_langgraph_compiled_state_graph(self._graph):
                result = await self._arun_compiled_graph_equivalent(initial)
            else:
                result = self._graph.invoke(initial)
            result = self._after_execution(result)
            logger.info("Orchestrator ainvoke complete")
            return result
        except Exception as exc:
            status = "failure"
            self._metrics.counter("orchestrator.errors", tags={"mode": "async"})
            logger.error("Orchestrator ainvoke failed", extra={"error": str(exc)}, exc_info=True)
            raise OrchestrationError(
                "Workflow execution failed",
                context={"user_input": input_preview[:200], "trace_id": trace_id},
                cause=exc,
            ) from exc
        finally:
            self._record_orchestrator_duration(started, mode="async", status=status)
            clear_trace_context()

    async def astream(self, user_input: Any, context: Optional[Dict] = None, **kwargs):
        """Execute the workflow as a stream, yielding each node's output."""
        initial = self._prepare_initial_state(
            user_input, context,
            workflow_plan=kwargs.pop("workflow_plan", None),
            template_name=kwargs.pop("template_name", None),
        )
        input_preview = _input_preview(user_input)
        set_trace_context()
        started = time.monotonic()
        status = "success"
        chunk_count = 0
        self._metrics.counter("orchestrator.streams", tags={"mode": "async"})
        logger.info("Orchestrator astream start", extra={"input_preview": input_preview})
        try:
            graph_astream = getattr(self._graph, "astream", None)
            if callable(graph_astream) and _should_use_native_async(self._graph):
                async for chunk in graph_astream(initial, **kwargs):
                    chunk_count += 1
                    yield chunk
                return

            if _is_langgraph_compiled_state_graph(self._graph):
                async for chunk in self._astream_compiled_graph_equivalent(initial):
                    chunk_count += 1
                    yield chunk
                return

            for chunk in self._graph.stream(initial, **kwargs):
                chunk_count += 1
                yield chunk
        except Exception as exc:
            status = "failure"
            self._metrics.counter("orchestrator.errors", tags={"mode": "stream"})
            logger.error("Orchestrator astream failed", extra={"error": str(exc)}, exc_info=True)
            raise
        finally:
            self._metrics.histogram(
                "orchestrator.stream_chunks",
                float(chunk_count),
                tags={"status": status},
            )
            self._record_orchestrator_duration(started, mode="stream", status=status)
            clear_trace_context()

    def chat(
        self,
        user_input: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience API for multi-turn chat.

        When ``session_id`` is supplied and a memory backend is configured,
        LangDeep loads and stores conversation history automatically.
        """
        chat_context = dict(context or {})
        if session_id is not None:
            chat_context["session_id"] = session_id
        return self.invoke(
            user_input,
            context=chat_context,
            workflow_plan=workflow_plan,
            template_name=template_name,
        )

    async def achat(
        self,
        user_input: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async convenience API for multi-turn chat."""
        chat_context = dict(context or {})
        if session_id is not None:
            chat_context["session_id"] = session_id
        return await self.ainvoke(
            user_input,
            context=chat_context,
            workflow_plan=workflow_plan,
            template_name=template_name,
        )

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute with explicit LangChain message history."""
        return self.invoke(
            list(messages),
            context=context,
            workflow_plan=workflow_plan,
            template_name=template_name,
        )

    async def ainvoke_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async execution with explicit LangChain message history."""
        return await self.ainvoke(
            list(messages),
            context=context,
            workflow_plan=workflow_plan,
            template_name=template_name,
        )

    def invoke_state(self, state: Dict[str, Any], *, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute with an explicit LangGraph-style state dictionary."""
        return self.invoke(state, context=context)

    async def ainvoke_state(self, state: Dict[str, Any], *, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Async execution with an explicit LangGraph-style state dictionary."""
        return await self.ainvoke(state, context=context)

    def health(self) -> Dict[str, Any]:
        """Return a health-check summary of the orchestrator and its components."""
        from ..observability import HealthChecker
        checker = HealthChecker(version=self.__class__.__module__)
        status = checker.check_all()
        return {
            "status": status.status,
            "checks": status.checks,
            "timestamp": status.timestamp.isoformat(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return in-process metrics from the built-in collector."""
        return self._metrics.get_metrics()

    def clear_metrics(self) -> None:
        """Clear in-process metrics for this orchestrator."""
        self._metrics.clear()

    def _record_orchestrator_duration(self, started: float, *, mode: str, status: str) -> None:
        self._metrics.histogram(
            "orchestrator.duration_ms",
            (time.monotonic() - started) * 1000,
            tags={"mode": mode, "status": status},
        )

    @property
    def graph(self):
        """The compiled LangGraph graph (for debugging / visualisation)."""
        return self._graph

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        schema = _build_state_schema()
        graph = StateGraph(schema)

        # Core nodes
        graph.add_node(self.NODE_SUPERVISOR, self._supervisor_node)
        graph.add_node(self.NODE_PLANNER, self._planner_node)
        graph.add_node(self.NODE_EXECUTOR, self._executor_node)
        graph.add_node(self.NODE_AGGREGATOR, self._aggregator_node)

        # Agent nodes
        self._agent_nodes: Dict[str, Callable] = {}
        for name in agent_registry.list_agents():
            node_fn = make_agent_node(
                name,
                max_retries=self._max_retries,
                clean_messages_fn=_clean_messages,
            )
            self._agent_nodes[name] = node_fn
            graph.add_node(name, node_fn)

        # Custom nodes
        custom_names = []
        for node_name, node_fn in self._custom_nodes.items():
            graph.add_node(node_name, node_fn)
            custom_names.append(node_name)
            logger.info("Custom node registered", extra={"node_name": node_name})

        # Edges
        graph.add_edge(START, self.NODE_SUPERVISOR)

        agent_names = list(self._agent_nodes.keys())
        all_targets = {self.NODE_PLANNER, "end", *agent_names, *custom_names}

        graph.add_conditional_edges(
            self.NODE_SUPERVISOR,
            self._route_from_supervisor,
            {t: (t if t != "end" else END) for t in all_targets},
        )

        graph.add_edge(self.NODE_PLANNER, self.NODE_EXECUTOR)
        graph.add_edge(self.NODE_EXECUTOR, self.NODE_AGGREGATOR)
        graph.add_edge(self.NODE_AGGREGATOR, END)

        for name in agent_names:
            graph.add_edge(name, self.NODE_AGGREGATOR)
        for name in custom_names:
            graph.add_edge(name, self.NODE_AGGREGATOR)

        if self._checkpointer:
            return graph.compile(checkpointer=self._checkpointer)
        return graph.compile()

    # ── Node: Supervisor ──────────────────────────────────────────────────────

    def _supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_core_node(self.NODE_SUPERVISOR, self._run_supervisor_node, state)

    def _run_supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agents = self._get_available_agents()
        # Keep valid targets up-to-date
        self._router.set_valid_targets(self._get_valid_targets())
        return self._router.route(state, agents)

    # ── Node: Planner ─────────────────────────────────────────────────────────

    def _planner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_core_node(self.NODE_PLANNER, self._planner.plan, state)

    # ── Node: Executor ────────────────────────────────────────────────────────

    def _executor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_core_node(self.NODE_EXECUTOR, self._executor.execute, state)

    # ── Node: Aggregator ──────────────────────────────────────────────────────

    def _aggregator_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_core_node(self.NODE_AGGREGATOR, self._aggregator.aggregate, state)

    def _run_core_node(self, node_name: str, node_fn: Callable, state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.monotonic()
        status = "success"
        self._metrics.counter("orchestrator.node.calls", tags={"node": node_name})
        try:
            return node_fn(state)
        except Exception:
            status = "failure"
            self._metrics.counter("orchestrator.node.errors", tags={"node": node_name})
            raise
        finally:
            self._metrics.histogram(
                "orchestrator.node.duration_ms",
                (time.monotonic() - started) * 1000,
                tags={"node": node_name, "status": status},
            )

    # ── Async graph equivalent ────────────────────────────────────────────────

    async def _arun_compiled_graph_equivalent(self, initial: Dict[str, Any]) -> Dict[str, Any]:
        """Run the orchestrator path asynchronously without LangGraph's async runner."""
        state = dict(initial)
        state, _ = await self._arun_node_update(self.NODE_SUPERVISOR, self._supervisor_node, state)
        next_node = self._route_from_supervisor(state)

        if next_node == "end":
            return state

        if next_node == self.NODE_PLANNER:
            for node_name, node_fn in (
                (self.NODE_PLANNER, self._planner_node),
                (self.NODE_EXECUTOR, self._executor_node),
                (self.NODE_AGGREGATOR, self._aggregator_node),
            ):
                state, _ = await self._arun_node_update(node_name, node_fn, state)
            return state

        node_fn = self._agent_nodes.get(next_node) or self._custom_nodes.get(next_node)
        if node_fn is None:
            return state

        state, _ = await self._arun_node_update(next_node, node_fn, state)
        state, _ = await self._arun_node_update(
            self.NODE_AGGREGATOR,
            self._aggregator_node,
            state,
        )
        return state

    async def _astream_compiled_graph_equivalent(self, initial: Dict[str, Any]):
        """Yield node updates asynchronously without LangGraph's async runner."""
        state = dict(initial)
        state, update = await self._arun_node_update(
            self.NODE_SUPERVISOR,
            self._supervisor_node,
            state,
        )
        yield {self.NODE_SUPERVISOR: update}

        next_node = self._route_from_supervisor(state)
        if next_node == "end":
            self._after_execution(state)
            return

        if next_node == self.NODE_PLANNER:
            for node_name, node_fn in (
                (self.NODE_PLANNER, self._planner_node),
                (self.NODE_EXECUTOR, self._executor_node),
                (self.NODE_AGGREGATOR, self._aggregator_node),
            ):
                state, update = await self._arun_node_update(node_name, node_fn, state)
                yield {node_name: update}
            self._after_execution(state)
            return

        node_fn = self._agent_nodes.get(next_node) or self._custom_nodes.get(next_node)
        if node_fn is None:
            self._after_execution(state)
            return

        state, update = await self._arun_node_update(next_node, node_fn, state)
        yield {next_node: update}
        state, update = await self._arun_node_update(
            self.NODE_AGGREGATOR,
            self._aggregator_node,
            state,
        )
        yield {self.NODE_AGGREGATOR: update}
        self._after_execution(state)

    async def _arun_node_update(
        self,
        node_name: str,
        node_fn: Callable,
        state: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        update = await _call_node_async(node_fn, state)
        if update is None:
            update = {}
        if not isinstance(update, dict):
            raise OrchestrationError(
                "Graph node returned a non-dict update",
                context={"node": node_name, "type": type(update).__name__},
            )
        return _merge_state_update(state, update), update

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_from_supervisor(self, state: Dict[str, Any]) -> str:
        next_node = state.get("next", "end")
        valid = {"planner", "end"} | set(agent_registry.list_agents()) | set(self._custom_nodes.keys())
        if next_node not in valid:
            logger.warning("Invalid routing target", extra={"target": next_node, "valid": list(valid)})
            return "end"
        return next_node

    # ── State helpers ─────────────────────────────────────────────────────────

    def _initial_state(
        self,
        user_input: str,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        plan = self._resolve_plan(user_input, workflow_plan, template_name)
        task_context = dict(context or {})
        session_id = _context_value(task_context, "session_id")
        memory_messages = self._load_memory_messages(session_id)
        messages = list(memory_messages) + [HumanMessage(content=user_input)]

        return {
            "messages": messages,
            "next": "",
            "current_task": "",
            "task_context": task_context,
            "agent_results": {},
            "workflow_plan": plan,
            "memory_session_id": session_id,
            "memory_start_len": len(memory_messages),
            "error_count": 0,
            "max_retries": self._max_retries,
            "aggregation_done": False,
        }

    def _prepare_initial_state(
        self,
        user_input: Any,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(user_input, str):
            return self._initial_state(user_input, context, workflow_plan, template_name)
        if isinstance(user_input, BaseMessage):
            return self._initial_state_from_messages(
                [user_input],
                context=context,
                workflow_plan=workflow_plan,
                template_name=template_name,
                include_memory=True,
            )
        if isinstance(user_input, dict):
            if "messages" in user_input:
                return self._initial_state_from_state(
                    user_input,
                    context=context,
                    workflow_plan=workflow_plan,
                    template_name=template_name,
                )
            for key in ("input", "user_input", "content"):
                if key in user_input:
                    value = user_input[key]
                    if not isinstance(value, str):
                        raise ConfigurationError(
                            f"FlowOrchestrator.invoke() expected dict['{key}'] to be a string",
                            context={"input_type": type(value).__name__},
                        )
                    return self._initial_state(value, context, workflow_plan, template_name)
            raise ConfigurationError(
                "FlowOrchestrator.invoke() received a dict without 'messages', 'input', 'user_input', or 'content'",
                context={"keys": sorted(str(key) for key in user_input.keys())},
            )
        if _is_message_sequence(user_input):
            return self._initial_state_from_messages(
                list(user_input),
                context=context,
                workflow_plan=workflow_plan,
                template_name=template_name,
                include_memory=False,
            )
        raise ConfigurationError(
            "FlowOrchestrator.invoke() expected a string, BaseMessage, sequence of BaseMessage, or state dict",
            context={"input_type": type(user_input).__name__},
        )

    def _initial_state_from_messages(
        self,
        messages: Sequence[BaseMessage],
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
        *,
        include_memory: bool,
    ) -> Dict[str, Any]:
        _validate_messages(messages)
        task_context = dict(context or {})
        session_id = _context_value(task_context, "session_id")
        memory_messages = self._load_memory_messages(session_id) if include_memory else []
        all_messages = list(memory_messages) + list(messages)
        user_text = _last_human_text(all_messages)
        plan = self._resolve_plan(user_text, workflow_plan, template_name)

        return {
            "messages": all_messages,
            "next": "",
            "current_task": "",
            "task_context": task_context,
            "agent_results": {},
            "workflow_plan": plan,
            "memory_session_id": session_id,
            "memory_start_len": len(memory_messages) if include_memory else len(all_messages),
            "error_count": 0,
            "max_retries": self._max_retries,
            "aggregation_done": False,
        }

    def _initial_state_from_state(
        self,
        state: Dict[str, Any],
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        _validate_messages(messages)

        task_context = dict(state.get("task_context") or {})
        task_context.update(context or {})
        session_id = state.get("memory_session_id") or _context_value(task_context, "session_id")
        user_text = _last_human_text(messages)
        plan = workflow_plan if workflow_plan is not None else state.get("workflow_plan")
        plan = self._resolve_plan(user_text, plan, template_name)

        initial = dict(state)
        initial.update({
            "messages": messages,
            "next": state.get("next", ""),
            "current_task": state.get("current_task", ""),
            "task_context": task_context,
            "agent_results": dict(state.get("agent_results") or {}),
            "workflow_plan": plan,
            "memory_session_id": session_id,
            "memory_start_len": int(state.get("memory_start_len", len(messages)) or 0),
            "error_count": int(state.get("error_count", 0) or 0),
            "max_retries": int(state.get("max_retries", self._max_retries) or self._max_retries),
            "aggregation_done": bool(state.get("aggregation_done", False)),
        })
        return initial

    def _resolve_plan(
        self,
        user_input: str,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Optional[List[Dict]]:
        if template_name and self._templates:
            return self._templates.apply(template_name, user_input)
        if workflow_plan:
            return workflow_plan
        return None

    def _load_memory_messages(self, session_id: Optional[str]) -> List[BaseMessage]:
        if not self._memory_backend or not session_id:
            return []
        try:
            messages = self._memory_backend.load_messages(session_id)
            logger.debug(
                "Memory messages loaded",
                extra={"session_id": session_id, "message_count": len(messages)},
            )
            return list(messages)
        except Exception as exc:
            logger.warning(
                "Failed to load memory messages",
                extra={"session_id": session_id, "error": str(exc)},
            )
            return []

    def _after_execution(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._store_memory_messages(state)
        self._sync_process_state(state)
        return state

    def _store_memory_messages(self, state: Dict[str, Any]) -> None:
        if not self._memory_backend:
            return
        session_id = state.get("memory_session_id") or _context_value(
            state.get("task_context", {}),
            "session_id",
        )
        if not session_id:
            return
        messages = list(state.get("messages") or [])
        start = int(state.get("memory_start_len") or 0)
        new_messages = messages[start:]
        if not new_messages:
            return
        try:
            self._memory_backend.store_messages(session_id, new_messages)
            logger.debug(
                "Memory messages stored",
                extra={"session_id": session_id, "message_count": len(new_messages)},
            )
        except Exception as exc:
            logger.warning(
                "Failed to store memory messages",
                extra={"session_id": session_id, "error": str(exc)},
            )

    def _sync_process_state(self, state: Dict[str, Any]) -> None:
        if self._process_manager is None:
            return
        task_context = state.get("task_context") or {}
        process_id = _context_value(task_context, "process_id")
        if not process_id:
            return

        snapshot = _build_process_snapshot(state)
        try:
            if _has_waiting_confirmation(state):
                process = self._process_manager.await_human(process_id, snapshot=snapshot)
            else:
                process = self._process_manager.update_snapshot(process_id, snapshot=snapshot)
            if process is None:
                logger.warning("Process not found", extra={"process_id": process_id})
        except Exception as exc:
            logger.warning(
                "Failed to sync process state",
                extra={"process_id": process_id, "error": str(exc)},
            )

    # ── Auto-import ───────────────────────────────────────────────────────────

    def _auto_import(self, dirs: List[str], caller_file: str, strict: bool = False) -> None:
        base_dir = os.path.dirname(os.path.abspath(caller_file))
        failures: List[Dict[str, str]] = []
        for folder in dirs:
            folder_path = (
                folder if os.path.isabs(folder)
                else os.path.join(base_dir, folder)
            )
            if not os.path.isdir(folder_path):
                logger.debug("Component directory not found, skipping", extra={"path": folder_path})
                continue
            parent = os.path.dirname(folder_path)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            package = os.path.basename(folder_path)
            for filename in sorted(os.listdir(folder_path)):
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue
                module_name = f"{package}.{filename[:-3]}"
                if module_name in sys.modules:
                    logger.debug("Module already loaded, skipping", extra={"component_module": module_name})
                    continue
                try:
                    importlib.import_module(module_name)
                    logger.info("Module loaded", extra={"component_module": module_name})
                except Exception as e:
                    logger.error("Module load failed", extra={"component_module": module_name, "error": str(e)}, exc_info=True)
                    failures.append({"module": module_name, "error": str(e)})

        if strict and failures:
            raise ConfigurationError(
                "Component auto-import failed in strict mode",
                context={"failures": failures},
            )

    # ── Cached lookups ────────────────────────────────────────────────────────

    def _get_available_agents(self) -> List[Dict]:
        if self._cached_agents is None:
            result = []
            for name in agent_registry.list_agents():
                meta = agent_registry.get_metadata(name)
                result.append({
                    "name": name,
                    "description": meta.description if meta else "",
                    "capabilities": meta.capabilities if meta else [],
                })
            self._cached_agents = result
        return self._cached_agents

    def _get_valid_targets(self) -> List[str]:
        if self._cached_targets is None:
            agents = [a["name"] for a in self._get_available_agents()]
            custom = list(self._custom_nodes.keys())
            self._cached_targets = ["planner", "end"] + agents + custom
        return self._cached_targets


def _is_langgraph_compiled_state_graph(graph: Any) -> bool:
    graph_type = type(graph)
    return (
        graph_type.__module__ == "langgraph.graph.state"
        and graph_type.__name__ == "CompiledStateGraph"
    )


def _should_use_native_async(graph: Any) -> bool:
    """Use custom async graph APIs, but avoid LangGraph sync-node async hangs."""
    return not _is_langgraph_compiled_state_graph(graph)


def _is_message_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(message, BaseMessage) for message in value)


def _validate_messages(messages: Sequence[BaseMessage]) -> None:
    invalid = [
        {"index": index, "type": type(message).__name__}
        for index, message in enumerate(messages)
        if not isinstance(message, BaseMessage)
    ]
    if invalid:
        raise ConfigurationError(
            "FlowOrchestrator message inputs must contain only LangChain BaseMessage objects",
            context={"invalid_messages": invalid},
        )


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    if messages:
        return str(messages[-1].content)
    return ""


def _input_preview(user_input: Any) -> str:
    if isinstance(user_input, str):
        return user_input[:120]
    if isinstance(user_input, BaseMessage):
        return f"{type(user_input).__name__}: {str(user_input.content)[:100]}"
    if isinstance(user_input, dict):
        if "messages" in user_input:
            return f"state(messages={len(user_input.get('messages') or [])})"
        return f"dict(keys={sorted(str(key) for key in user_input.keys())})"
    if _is_message_sequence(user_input):
        return f"messages(count={len(user_input)})"
    return f"{type(user_input).__name__}"


async def _call_node_async(node_fn: Callable, state: Dict[str, Any]) -> Dict[str, Any]:
    result = node_fn(state)
    if inspect.isawaitable(result):
        result = await result
    return result


def _merge_state_update(state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(state)
    for key, value in update.items():
        if key == "messages":
            existing = list(merged.get("messages") or [])
            incoming = list(value or [])
            merged[key] = existing + incoming
        elif key == "agent_results":
            merged[key] = {**(merged.get("agent_results") or {}), **(value or {})}
        else:
            merged[key] = value
    return merged


def _context_value(context: Dict[str, Any], key: str) -> Optional[str]:
    value = context.get(key) if isinstance(context, dict) else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_waiting_confirmation(state: Dict[str, Any]) -> bool:
    return any(
        isinstance(task, dict) and task.get("status") == "waiting_confirmation"
        for task in (state.get("workflow_plan") or [])
    )


def _build_process_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [_message_snapshot(message) for message in state.get("messages", [])],
        "task_context": dict(state.get("task_context") or {}),
        "agent_results": dict(state.get("agent_results") or {}),
        "workflow_plan": list(state.get("workflow_plan") or []),
        "aggregation_done": bool(state.get("aggregation_done", False)),
    }


def _message_snapshot(message: BaseMessage) -> Dict[str, str]:
    return {
        "type": getattr(message, "type", type(message).__name__),
        "content": str(getattr(message, "content", "")),
    }
