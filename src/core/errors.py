"""Structured exception hierarchy for LangDeep."""

from typing import Any, Dict, Optional


class LangDeepError(Exception):
    """Base exception for all LangDeep errors.

    Attributes:
        code: Machine-readable error code (e.g. "MODEL_NOT_FOUND").
        detail: Human-readable message.
        context: Optional dict with structured context (model name, agent name, etc.).
        cause: The original exception that triggered this error, if any.
    """

    code: str = "LANKDEEP_ERROR"

    def __init__(
        self,
        detail: str = "",
        *,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.detail = detail
        self.context = context or {}
        self.cause = cause
        super().__init__(self._format())

    def _format(self) -> str:
        parts = [f"[{self.code}] {self.detail}"]
        if self.context:
            parts.append(f"context={self.context}")
        if self.cause:
            parts.append(f"cause={self.cause!r}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }


# ── Configuration errors ─────────────────────────────────────────────────────────

class ConfigurationError(LangDeepError):
    """Invalid configuration detected at startup or registration time."""
    code = "CONFIG_ERROR"


class InvalidPolicyError(ConfigurationError):
    """Execution policy is misconfigured."""
    code = "INVALID_POLICY"


# ── Model errors ─────────────────────────────────────────────────────────────────

class ModelError(LangDeepError):
    """Base for model-related errors."""
    code = "MODEL_ERROR"


class ModelNotFoundError(ModelError):
    """Requested model is not registered."""
    code = "MODEL_NOT_FOUND"


class ProviderNotFoundError(ModelError):
    """Requested provider is not registered."""
    code = "PROVIDER_NOT_FOUND"


class ProviderImportError(ModelError):
    """Provider SDK package is not installed."""
    code = "PROVIDER_IMPORT_ERROR"


class ModelInvocationError(ModelError):
    """LLM API call failed."""
    code = "MODEL_INVOCATION_ERROR"


class ModelTimeoutError(ModelInvocationError):
    """LLM API call timed out."""
    code = "MODEL_TIMEOUT"


# ── Agent errors ─────────────────────────────────────────────────────────────────

class AgentError(LangDeepError):
    """Base for agent-related errors."""
    code = "AGENT_ERROR"


class AgentNotFoundError(AgentError):
    """Requested agent is not registered."""
    code = "AGENT_NOT_FOUND"


class AgentInvocationError(AgentError):
    """Agent execution failed."""
    code = "AGENT_INVOCATION_ERROR"


class AgentRetryExhaustedError(AgentInvocationError):
    """Agent failed after all retry attempts."""
    code = "AGENT_RETRY_EXHAUSTED"


# ── Tool errors ─────────────────────────────────────────────────────────────────

class ToolError(LangDeepError):
    """Base for tool-related errors."""
    code = "TOOL_ERROR"


class ToolNotFoundError(ToolError):
    """Requested tool is not registered."""
    code = "TOOL_NOT_FOUND"


# ── Execution errors ─────────────────────────────────────────────────────────────

class ExecutionError(LangDeepError):
    """Base for workflow execution errors."""
    code = "EXECUTION_ERROR"


class TaskExecutionError(ExecutionError):
    """A single task within a workflow failed."""
    code = "TASK_EXECUTION_ERROR"


class CircularDependencyError(ExecutionError):
    """Workflow plan contains circular dependencies."""
    code = "CIRCULAR_DEPENDENCY"


# ── Orchestration errors ─────────────────────────────────────────────────────────

class OrchestrationError(LangDeepError):
    """Base for orchestration-level errors."""
    code = "ORCHESTRATION_ERROR"


class RoutingError(OrchestrationError):
    """Supervisor routing failed to determine a valid next node."""
    code = "ROUTING_ERROR"


class PlannerError(OrchestrationError):
    """Workflow plan generation failed."""
    code = "PLANNER_ERROR"


class AggregatorError(OrchestrationError):
    """Result aggregation failed."""
    code = "AGGREGATOR_ERROR"


# ── Template & prompt errors ────────────────────────────────────────────────────

class TemplateError(LangDeepError):
    """Base for template/prompt errors."""
    code = "TEMPLATE_ERROR"


class TemplateNotFoundError(TemplateError):
    """Workflow template not found."""
    code = "TEMPLATE_NOT_FOUND"


class PromptNotFoundError(TemplateError):
    """Prompt file not found."""
    code = "PROMPT_NOT_FOUND"
