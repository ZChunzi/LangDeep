"""LangDeep — LangChain + LangGraph Agent Workflow Framework."""

from langdeep.core.orchestrator import FlowOrchestrator
from langdeep.core.orchestrator import (
    RoutingStrategy,
    KeywordRoutingStrategy,
    PlanGenerator,
    LLMPlanGenerator,
    FallbackPlanGenerator,
    TaskRunner,
    RetryTaskRunner,
    ResultMerger,
    LLMMerger,
    ConcatMerger,
    ok,
    err,
)
from langdeep.core.decorators.model import model
from langdeep.core.decorators.tool import regist_tool
from langdeep.core.decorators.agent import agent
from langdeep.core.decorators.provider import provider
from langdeep.core.execution.execution_policy import ExecutionPolicy
from langdeep.core.planner import WorkflowPlanner, WorkflowNode, NodeType
from langdeep.core.errors import (
    LangDeepError,
    ConfigurationError,
    ModelError,
    ModelNotFoundError,
    AgentError,
    AgentNotFoundError,
    ToolError,
    ToolNotFoundError,
    ExecutionError,
    OrchestrationError,
)
from langdeep.core.logging import get_logger, set_trace_context, get_trace_id

__all__ = [
    # Orchestrator
    "FlowOrchestrator",
    # Extension points
    "RoutingStrategy",
    "KeywordRoutingStrategy",
    "PlanGenerator",
    "LLMPlanGenerator",
    "FallbackPlanGenerator",
    "TaskRunner",
    "RetryTaskRunner",
    "ResultMerger",
    "LLMMerger",
    "ConcatMerger",
    "ok",
    "err",
    # Decorators
    "model",
    "regist_tool",
    "agent",
    "provider",
    # Execution
    "ExecutionPolicy",
    # Planner
    "WorkflowPlanner",
    "WorkflowNode",
    "NodeType",
    # Errors
    "LangDeepError",
    "ConfigurationError",
    "ModelError",
    "ModelNotFoundError",
    "AgentError",
    "AgentNotFoundError",
    "ToolError",
    "ToolNotFoundError",
    "ExecutionError",
    "OrchestrationError",
    # Logging
    "get_logger",
    "set_trace_context",
    "get_trace_id",
]
__version__ = "1.0.7"
