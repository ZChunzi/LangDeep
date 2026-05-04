"""Unit tests for LangDeep structured exception hierarchy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langdeep.core.errors import (
    LangDeepError,
    ConfigurationError, InvalidPolicyError,
    ModelError, ModelNotFoundError, ProviderNotFoundError, ProviderImportError,
    ModelInvocationError, ModelTimeoutError,
    AgentError, AgentNotFoundError, AgentInvocationError, AgentRetryExhaustedError,
    ToolError, ToolNotFoundError,
    ExecutionError, TaskExecutionError, CircularDependencyError,
    OrchestrationError, RoutingError, PlannerError, AggregatorError,
    TemplateError, TemplateNotFoundError, PromptNotFoundError,
)


def test_base_error():
    e = LangDeepError("test", context={"key": "val"}, cause=ValueError("inner"))
    assert e.code == "LANKDEEP_ERROR"
    assert e.detail == "test"
    assert e.context == {"key": "val"}
    assert isinstance(e.cause, ValueError)
    assert "LANKDEEP_ERROR" in str(e)
    d = e.to_dict()
    assert d["code"] == "LANKDEEP_ERROR"
    assert d["detail"] == "test"


def test_configuration_errors():
    assert issubclass(ConfigurationError, LangDeepError)
    assert issubclass(InvalidPolicyError, ConfigurationError)
    assert InvalidPolicyError().code == "INVALID_POLICY"


def test_model_errors():
    assert issubclass(ModelNotFoundError, ModelError)
    assert issubclass(ProviderNotFoundError, ModelError)
    assert issubclass(ProviderImportError, ModelError)
    assert issubclass(ModelInvocationError, ModelError)
    assert issubclass(ModelTimeoutError, ModelInvocationError)

    e = ModelNotFoundError("not found", context={"model": "gpt5"})
    assert e.code == "MODEL_NOT_FOUND"
    assert "gpt5" in str(e)


def test_agent_errors():
    assert issubclass(AgentNotFoundError, AgentError)
    assert issubclass(AgentInvocationError, AgentError)
    assert issubclass(AgentRetryExhaustedError, AgentInvocationError)

    e = AgentNotFoundError("missing agent", context={"available": ["a1"]})
    assert e.code == "AGENT_NOT_FOUND"
    assert "a1" in str(e)


def test_tool_errors():
    assert issubclass(ToolNotFoundError, ToolError)
    e = ToolNotFoundError("missing tool")
    assert e.code == "TOOL_NOT_FOUND"


def test_execution_errors():
    assert issubclass(TaskExecutionError, ExecutionError)
    assert issubclass(CircularDependencyError, ExecutionError)
    assert CircularDependencyError().code == "CIRCULAR_DEPENDENCY"


def test_orchestration_errors():
    assert issubclass(RoutingError, OrchestrationError)
    assert issubclass(PlannerError, OrchestrationError)
    assert issubclass(AggregatorError, OrchestrationError)
    assert RoutingError().code == "ROUTING_ERROR"


def test_template_errors():
    assert issubclass(TemplateNotFoundError, TemplateError)
    assert issubclass(PromptNotFoundError, TemplateError)
    assert TemplateNotFoundError().code == "TEMPLATE_NOT_FOUND"


def test_error_without_context():
    e = LangDeepError()
    assert e.detail == ""
    assert e.context == {}
    assert e.cause is None


def test_error_from_exception():
    try:
        1 / 0
    except ZeroDivisionError as exc:
        e = LangDeepError("math error", cause=exc)
        assert "ZeroDivisionError" in str(e)
        assert e.to_dict()["cause"] is not None
