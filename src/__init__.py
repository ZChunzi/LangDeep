"""LangDeep — LangChain + LangGraph Agent Workflow Framework."""

from typing import Callable

from langchain_core.language_models import BaseChatModel

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
from langdeep.core.decorators.tool import register_tool, regist_tool
from langdeep.core.decorators.agent import agent
from langdeep.core.decorators.provider import provider
from langdeep.core.decorators import memory, cache, im_channel
from langdeep.core.memory import RedisMemoryBackend
from langdeep.core.registry.model_registry import ModelConfig, provider_registry
from langdeep.core.sandbox import BaseSandbox, SubprocessSandbox, sandbox_registry, sandbox
from langdeep.core.process import ProcessManager, ProcessState
from langdeep.core.secrets import secrets_manager, EnvSecretsProvider
from langdeep.core.observability import HealthChecker, MetricsCollector
from langdeep.core.diagnostics import (
    DiagnosticIssue,
    RuntimeDiagnostics,
    RuntimeValidator,
    validate_runtime,
)
from langdeep.core.execution.execution_policy import ExecutionPolicy
from langdeep.core.agent_builder import BaseAgentBuilder, ReActAgentBuilder, agent_builder_registry
from langdeep.core.adapters.deepseek import (
    DeepSeekChatModel,
    DeepSeekCompatibilityProfile,
    build_deepseek_payload_messages,
    configure_deepseek_v4,
    extract_reasoning_content,
    normalize_deepseek_messages,
)
from langdeep.core.planner import WorkflowPlanner, WorkflowNode, NodeType
from langdeep.core.tools import PolicyAwareTool, ToolAuditLog, ToolExecutionPolicy, ToolExecutionRecord
from langdeep.schemas import WorkflowPlan, WorkflowTask, validate_workflow_plan
from langdeep.core.errors import (
    LangDeepError,
    ConfigurationError,
    ModelError,
    ModelNotFoundError,
    AgentError,
    AgentBuildError,
    AgentNotFoundError,
    ToolError,
    ToolNotFoundError,
    ToolPolicyError,
    ToolConfirmationRequired,
    ToolWorkspaceError,
    ToolTimeoutError,
    ExecutionError,
    OrchestrationError,
)
from langdeep.core.logging import get_logger, set_trace_context, get_trace_id
from langdeep.messages import (
    AssistantMessage,
    Message,
    UserMessage,
    assistant_message,
    last_assistant_text,
    last_user_text,
    message_text,
    system_message,
    tool_message,
    user_message,
)

__version__ = "2.0.13"


def register_provider(name: str, factory: Callable[[ModelConfig], BaseChatModel]) -> Callable[[ModelConfig], BaseChatModel]:
    """Register a model provider factory and return it.

    This is the function-style companion to the ``@provider(name=...)`` decorator.
    It is useful when provider factories are created dynamically or when users prefer
    explicit registration in application bootstrap code.
    """
    provider_registry.register(name, factory)
    return factory


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
    # Decorators and provider helpers
    "model",
    "register_tool",
    "regist_tool",
    "agent",
    "provider",
    "register_provider",
    "ModelConfig",
    "AssistantMessage",
    "Message",
    "UserMessage",
    "assistant_message",
    "last_assistant_text",
    "last_user_text",
    "message_text",
    "system_message",
    "tool_message",
    "user_message",
    "memory",
    "RedisMemoryBackend",
    "cache",
    "im_channel",
    # Execution
    "ExecutionPolicy",
    "BaseAgentBuilder",
    "ReActAgentBuilder",
    "agent_builder_registry",
    # Planner
    "WorkflowPlanner",
    "WorkflowNode",
    "NodeType",
    "WorkflowPlan",
    "WorkflowTask",
    "validate_workflow_plan",
    # Errors
    "LangDeepError",
    "ConfigurationError",
    "ModelError",
    "ModelNotFoundError",
    "AgentError",
    "AgentBuildError",
    "AgentNotFoundError",
    "ToolError",
    "ToolNotFoundError",
    "ToolPolicyError",
    "ToolConfirmationRequired",
    "ToolWorkspaceError",
    "ToolTimeoutError",
    "ExecutionError",
    "OrchestrationError",
    # Logging
    "get_logger",
    "set_trace_context",
    "get_trace_id",
    # DeepSeek adapter
    "DeepSeekChatModel",
    "DeepSeekCompatibilityProfile",
    "build_deepseek_payload_messages",
    "configure_deepseek_v4",
    "extract_reasoning_content",
    "normalize_deepseek_messages",
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
    "DiagnosticIssue",
    "RuntimeDiagnostics",
    "RuntimeValidator",
    "validate_runtime",
    # Tool policy
    "PolicyAwareTool",
    "ToolAuditLog",
    "ToolExecutionPolicy",
    "ToolExecutionRecord",
]
