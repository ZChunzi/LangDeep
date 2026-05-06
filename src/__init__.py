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
from langdeep.core.decorators import memory, cache, im_channel
from langdeep.core.sandbox import BaseSandbox, SubprocessSandbox, sandbox_registry, sandbox
from langdeep.core.process import ProcessManager, ProcessState
from langdeep.core.secrets import secrets_manager, EnvSecretsProvider
from langdeep.core.observability import HealthChecker, MetricsCollector
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

__version__ = "1.2.1"

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
    "memory",
    "cache",
    "im_channel",
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
    # Sandbox
    "BaseSandbox",
    "SubprocessSandbox",
    "sandbox_registry",
    "sandbox",
    # Process
    "ProcessManager",
    "ProcessState",
    # Secrets
    "secrets_manager",
    "EnvSecretsProvider",
    # Observability
    "HealthChecker",
    "MetricsCollector",
]
