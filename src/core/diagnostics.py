"""Runtime diagnostics for startup preflight checks."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from .errors import ConfigurationError
from .logging import get_logger
from .registry.agent_registry import agent_registry
from .registry.model_registry import model_registry, provider_registry
from .registry.tool_registry import tool_registry

logger = get_logger(__name__)

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class DiagnosticIssue:
    """A structured runtime configuration issue."""

    severity: Severity
    component: str
    name: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "component": self.component,
            "name": self.name,
            "message": self.message,
            "context": self.context,
        }


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """Result of a runtime preflight validation."""

    issues: List[DiagnosticIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def raise_for_errors(self) -> None:
        if self.ok:
            return
        raise ConfigurationError(
            "Runtime diagnostics found configuration errors",
            context=self.to_dict(),
        )


class RuntimeValidator:
    """Validate registry wiring before serving production traffic."""

    def validate(self, *, instantiate_agents: bool = False) -> RuntimeDiagnostics:
        issues: List[DiagnosticIssue] = []
        self._check_models(issues)
        self._check_agents(issues, instantiate_agents=instantiate_agents)
        self._check_tools(issues)
        diagnostics = RuntimeDiagnostics(issues=issues)
        if diagnostics.ok:
            logger.info("Runtime diagnostics passed", extra=diagnostics.to_dict())
        else:
            logger.warning("Runtime diagnostics found issues", extra=diagnostics.to_dict())
        return diagnostics

    @staticmethod
    def _check_models(issues: List[DiagnosticIssue]) -> None:
        providers = set(provider_registry.list_providers())
        for name in model_registry.list_models():
            try:
                config = model_registry.get_config(name)
            except Exception as exc:
                issues.append(_issue("error", "model", name, "model config is unreadable", error=str(exc)))
                continue

            if not name.strip():
                issues.append(_issue("error", "model", name, "model registry name must not be empty"))
            if not config.model_name or not str(config.model_name).strip():
                issues.append(_issue("error", "model", name, "model_name must not be empty"))
            if not config.provider or not str(config.provider).strip():
                issues.append(_issue("error", "model", name, "provider must not be empty"))
            elif config.provider not in providers:
                issues.append(
                    _issue(
                        "error",
                        "model",
                        name,
                        f"provider '{config.provider}' is not registered",
                        available=sorted(providers),
                    )
                )
            try:
                temperature = float(config.temperature)
            except (TypeError, ValueError):
                issues.append(
                    _issue(
                        "error",
                        "model",
                        name,
                        "temperature must be numeric",
                        temperature=config.temperature,
                    )
                )
                temperature = 0.7

            if not 0 <= temperature <= 2:
                issues.append(
                    _issue(
                        "warning",
                        "model",
                        name,
                        "temperature is outside the common provider range [0, 2]",
                        temperature=config.temperature,
                    )
                )
            if config.max_tokens is not None and config.max_tokens <= 0:
                issues.append(
                    _issue(
                        "error",
                        "model",
                        name,
                        "max_tokens must be positive when provided",
                        max_tokens=config.max_tokens,
                    )
                )

    @staticmethod
    def _check_agents(issues: List[DiagnosticIssue], *, instantiate_agents: bool) -> None:
        model_names = set(model_registry.list_models())
        tool_names = set(tool_registry.list_tools())
        for name in agent_registry.list_agents():
            meta = agent_registry.get_metadata(name)
            if meta is None:
                issues.append(_issue("error", "agent", name, "agent metadata is missing"))
                continue

            if meta.name != name:
                issues.append(
                    _issue(
                        "error",
                        "agent",
                        name,
                        "agent metadata name does not match registry key",
                        metadata_name=meta.name,
                    )
                )
            if not meta.description:
                issues.append(_issue("warning", "agent", name, "agent description is empty"))
            if meta.model_name and meta.model_name != "default" and meta.model_name not in model_names:
                issues.append(
                    _issue(
                        "error",
                        "agent",
                        name,
                        f"agent model '{meta.model_name}' is not registered",
                        available=sorted(model_names),
                    )
                )

            missing_tools = [tool for tool in (meta.tools or []) if tool not in tool_names]
            if missing_tools:
                issues.append(
                    _issue(
                        "error",
                        "agent",
                        name,
                        "agent references tools that are not registered",
                        missing_tools=missing_tools,
                    )
                )

            if instantiate_agents:
                try:
                    agent_registry.get_agent(name)
                except Exception as exc:
                    issues.append(
                        _issue(
                            "error",
                            "agent",
                            name,
                            "agent failed to instantiate",
                            error=str(exc),
                        )
                    )

    @staticmethod
    def _check_tools(issues: List[DiagnosticIssue]) -> None:
        for name in tool_registry.list_tools():
            try:
                tool = tool_registry.get_tool(name)
            except Exception as exc:
                issues.append(_issue("error", "tool", name, "tool is unreadable", error=str(exc)))
                continue

            meta = tool_registry.get_metadata(name)
            if meta is not None and meta.name != name:
                issues.append(
                    _issue(
                        "warning",
                        "tool",
                        name,
                        "tool metadata name does not match registry key",
                        metadata_name=meta.name,
                    )
                )
            if not getattr(tool, "description", "") and (meta is None or not meta.description):
                issues.append(_issue("warning", "tool", name, "tool description is empty"))


def validate_runtime(*, instantiate_agents: bool = False) -> RuntimeDiagnostics:
    """Run startup diagnostics against the global LangDeep registries."""
    return RuntimeValidator().validate(instantiate_agents=instantiate_agents)


def _issue(
    severity: Severity,
    component: str,
    name: str,
    message: str,
    **context: Any,
) -> DiagnosticIssue:
    return DiagnosticIssue(
        severity=severity,
        component=component,
        name=name,
        message=message,
        context={k: v for k, v in context.items() if v is not None},
    )
