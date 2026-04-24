"""Flow orchestrator — modular, extensible coordinator built on LangGraph."""

import asyncio
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from ..logging import get_logger, set_trace_context, clear_trace_context
from ..errors import LangDeepError, OrchestrationError
from ..registry.agent_registry import agent_registry
from ..execution.execution_policy import ExecutionPolicy

from .router import DefaultRouter, RoutingStrategy
from .planner import Planner, PlanGenerator, TemplateLoader, FallbackPlanGenerator
from .executor import Executor, TaskRunner, _clean_messages
from .aggregator import Aggregator, ResultMerger
from .agent_node import make_agent_node

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
    ):
        self._supervisor_model = supervisor_model
        self._max_retries = max_retries
        self._llm_timeout = llm_timeout
        self._policy = execution_policy or ExecutionPolicy()

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

        # Auto-import component modules
        self._auto_import(
            dirs=component_dirs or DEFAULT_COMPONENT_DIRS,
            caller_file=inspect.stack()[1].filename,
        )

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
        )

        self._planner = Planner(
            model_name=supervisor_model,
            plan_generator=plan_generator,
            prompt_loader=self._prompt_loader,
        )

        self._executor = Executor(
            task_runner=task_runner,
            policy=self._policy,
            max_retries=max_retries,
            timeout=llm_timeout,
        )

        self._aggregator = Aggregator(
            model_name=supervisor_model,
            merger=result_merger,
            prompt_loader=self._prompt_loader,
        )

        self._graph = self._build_graph()

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(
        self,
        user_input: str,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow synchronously and return the final state."""
        initial = self._initial_state(user_input, context, workflow_plan, template_name)
        trace_id = set_trace_context()
        logger.info("Orchestrator invoke start", extra={"input_preview": user_input[:120]})
        try:
            result = self._graph.invoke(initial)
            logger.info("Orchestrator invoke complete")
            return result
        except Exception as exc:
            logger.error("Orchestrator invoke failed", extra={"error": str(exc)}, exc_info=True)
            raise OrchestrationError(
                "Workflow execution failed",
                context={"user_input": user_input[:200], "trace_id": trace_id},
                cause=exc,
            ) from exc
        finally:
            clear_trace_context()

    async def ainvoke(
        self,
        user_input: str,
        context: Optional[Dict] = None,
        workflow_plan: Optional[List[Dict]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow asynchronously and return the final state."""
        initial = self._initial_state(user_input, context, workflow_plan, template_name)
        trace_id = set_trace_context()
        logger.info("Orchestrator ainvoke start", extra={"input_preview": user_input[:120]})
        try:
            result = await self._graph.ainvoke(initial)
            logger.info("Orchestrator ainvoke complete")
            return result
        except Exception as exc:
            logger.error("Orchestrator ainvoke failed", extra={"error": str(exc)}, exc_info=True)
            raise OrchestrationError(
                "Async workflow execution failed",
                context={"user_input": user_input[:200], "trace_id": trace_id},
                cause=exc,
            ) from exc
        finally:
            clear_trace_context()

    async def astream(self, user_input: str, context: Optional[Dict] = None, **kwargs):
        """Execute the workflow as a stream, yielding each node's output."""
        initial = self._initial_state(
            user_input, context,
            workflow_plan=kwargs.pop("workflow_plan", None),
            template_name=kwargs.pop("template_name", None),
        )
        set_trace_context()
        logger.info("Orchestrator astream start", extra={"input_preview": user_input[:120]})
        try:
            async for chunk in self._graph.astream(initial, **kwargs):
                yield chunk
        except Exception as exc:
            logger.error("Orchestrator astream failed", extra={"error": str(exc)}, exc_info=True)
            raise
        finally:
            clear_trace_context()

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
        agents = self._get_available_agents()
        # Keep valid targets up-to-date
        self._router.set_valid_targets(self._get_valid_targets())
        result = self._router.route(state, agents)
        return result

    # ── Node: Planner ─────────────────────────────────────────────────────────

    def _planner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._planner.plan(state)

    # ── Node: Executor ────────────────────────────────────────────────────────

    def _executor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._executor.execute(state)

    # ── Node: Aggregator ──────────────────────────────────────────────────────

    def _aggregator_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._aggregator.aggregate(state)

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
        plan = None
        if template_name and self._templates:
            plan = self._templates.apply(template_name, user_input)
        elif workflow_plan:
            plan = workflow_plan

        return {
            "messages": [HumanMessage(content=user_input)],
            "next": "",
            "current_task": "",
            "task_context": context or {},
            "agent_results": {},
            "workflow_plan": plan,
            "error_count": 0,
            "max_retries": self._max_retries,
            "aggregation_done": False,
        }

    # ── Auto-import ───────────────────────────────────────────────────────────

    def _auto_import(self, dirs: List[str], caller_file: str) -> None:
        base_dir = os.path.dirname(os.path.abspath(caller_file))
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
