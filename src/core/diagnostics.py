"""Runtime diagnostics for startup preflight checks."""

import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
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


def build_doctor_report(
    *,
    strict: bool = False,
    instantiate_agents: bool = True,
    timeout: int = 5,
    audit_sink: Any = None,
) -> Dict[str, Any]:
    """Build an environment, registry, dependency, health, and security report."""
    diagnostics = validate_runtime(instantiate_agents=instantiate_agents)
    health = _health_snapshot(timeout)
    registries = _registry_snapshot()
    dependencies, dependency_issues = _dependency_snapshot()
    security = _security_snapshot()
    audit = _audit_snapshot(audit_sink)
    environment_issues = _environment_issues()

    doctor_issues = environment_issues + dependency_issues + security["issues"] + audit["issues"]
    error_count = diagnostics.error_count + sum(
        1 for issue in doctor_issues if issue["severity"] == "error"
    )
    warning_count = diagnostics.warning_count + sum(
        1 for issue in doctor_issues if issue["severity"] == "warning"
    )
    if health.get("status") == "unhealthy":
        error_count += 1

    status = "error" if error_count else "warning" if warning_count else "ok"
    exit_ok = status != "error" and not (strict and warning_count)
    version = _langdeep_version()

    return {
        "status": status,
        "ok": exit_ok,
        "strict": strict,
        "version": version,
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_supported": sys.version_info >= (3, 9),
            "platform": platform.platform(),
        },
        "dependencies": dependencies,
        "registries": registries,
        "diagnostics": diagnostics.to_dict(),
        "health": health,
        "audit": {
            "schema_version": audit["schema_version"],
            "sink_type": audit["sink_type"],
            "configured": audit["configured"],
            "durable": audit["durable"],
            "path": audit["path"],
            "issues": audit["issues"],
        },
        "security": {
            "response_cache": security["response_cache"],
            "issues": security["issues"],
        },
        "issues": doctor_issues,
        "error_count": error_count,
        "warning_count": warning_count,
    }


def _audit_snapshot(audit_sink: Any = None) -> Dict[str, Any]:
    from .audit import AUDIT_SCHEMA_VERSION, AuditSink, InMemoryAuditSink, JsonlAuditSink

    issues = []
    sink_type = type(audit_sink).__name__ if audit_sink is not None else None
    configured = audit_sink is not None
    durable = False
    path = None

    if audit_sink is None:
        issues.append(
            _doctor_issue(
                "warning",
                "audit",
                "sink",
                "no audit sink was provided to doctor; production services should configure durable audit logging",
            )
        )
    elif isinstance(audit_sink, JsonlAuditSink):
        durable = True
        path = str(audit_sink.path)
    elif isinstance(audit_sink, InMemoryAuditSink):
        issues.append(
            _doctor_issue(
                "warning",
                "audit",
                "sink",
                "InMemoryAuditSink is process-local and should not be the production audit store",
                sink_type=sink_type,
            )
        )
    elif not isinstance(audit_sink, AuditSink) and not hasattr(audit_sink, "record"):
        issues.append(
            _doctor_issue(
                "warning",
                "audit",
                "sink",
                "audit sink does not expose record(); verify it can persist normalized AuditEvent objects",
                sink_type=sink_type,
            )
        )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "sink_type": sink_type,
        "configured": configured,
        "durable": durable,
        "path": path,
        "issues": issues,
    }


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


def _doctor_issue(
    severity: Severity,
    component: str,
    name: str,
    message: str,
    **context: Any,
) -> Dict[str, Any]:
    return _issue(severity, component, name, message, **context).to_dict()


def _health_snapshot(timeout: int) -> Dict[str, Any]:
    try:
        from .observability import HealthChecker

        status = HealthChecker(version=_langdeep_version()).check_all(timeout=timeout)
        return {
            "status": status.status,
            "checks": status.checks,
            "version": status.version,
            "timestamp": status.timestamp.isoformat(),
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "checks": {"doctor": {"status": "error", "detail": str(exc)}},
            "version": _langdeep_version(),
        }


def _registry_snapshot() -> Dict[str, Any]:
    from .cache.registry import cache_registry
    from .im.registry import im_channel_registry
    from .memory.registry import memory_registry
    from .sandbox.registry import sandbox_registry

    return {
        "models": model_registry.list_models(),
        "providers": provider_registry.list_providers(),
        "agents": agent_registry.list_agents(),
        "tools": tool_registry.list_tools(),
        "memory": memory_registry.list_backends(),
        "cache": cache_registry.list_backends(),
        "im": im_channel_registry.list_channels(),
        "sandbox": sandbox_registry.list_backends(),
    }


def _dependency_snapshot() -> tuple:
    provider_dependencies = {
        "openai": "langchain-openai",
        "azure_openai": "langchain-openai",
        "deepseek": "langchain-openai",
        "anthropic": "langchain-anthropic",
        "google_genai": "langchain-google-genai",
        "vertexai": "langchain-google-vertexai",
        "ollama": "langchain-ollama",
    }
    packages = {
        "langchain_core": "langchain-core",
        "langgraph": "langgraph",
        "pydantic": "pydantic",
        "langchain_openai": "langchain-openai",
        "langchain_anthropic": "langchain-anthropic",
        "langchain_google_genai": "langchain-google-genai",
        "langchain_google_vertexai": "langchain-google-vertexai",
        "langchain_ollama": "langchain-ollama",
    }
    dependencies = {
        logical_name: _package_version(package_name)
        for logical_name, package_name in packages.items()
    }
    issues = []
    registered_providers = set(provider_registry.list_providers())
    for provider_name, package_name in provider_dependencies.items():
        if provider_name in registered_providers and _package_version(package_name) is None:
            issues.append(
                _doctor_issue(
                    "warning",
                    "dependency",
                    provider_name,
                    f"provider optional dependency '{package_name}' is not installed",
                    package=package_name,
                )
            )
    return dependencies, issues


def _package_version(package_name: str) -> Any:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _langdeep_version() -> str:
    return str(_package_version("langdeep") or "")


def _environment_issues() -> List[Dict[str, Any]]:
    if sys.version_info >= (3, 9):
        return []
    return [
        _doctor_issue(
            "error",
            "environment",
            "python",
            "Python version is below the supported minimum 3.9",
            python=platform.python_version(),
        )
    ]


def _security_snapshot() -> Dict[str, Any]:
    from .sandbox.registry import sandbox_registry

    issues = []
    sandbox_backends = sandbox_registry.list_backends()
    if "subprocess" in sandbox_backends:
        issues.append(
            _doctor_issue(
                "warning",
                "sandbox",
                "subprocess",
                "SubprocessSandbox is registered; it is not a complete boundary for hostile code",
            )
        )

    file_tools = []
    for name in tool_registry.list_tools():
        meta = tool_registry.get_metadata(name)
        tags = getattr(meta, "tags", None) or []
        if getattr(meta, "category", None) == "file" or "file" in tags:
            file_tools.append(name)
    if file_tools and not tool_registry.get_policy().workspace_roots:
        issues.append(
            _doctor_issue(
                "warning",
                "tool_policy",
                "workspace_roots",
                "file tools are registered but workspace roots are not configured",
                tools=file_tools,
            )
        )

    hardcoded_keys = []
    for name, config in model_registry.list_model_configs().items():
        if config.api_key and not _looks_indirect_secret(config.api_key):
            hardcoded_keys.append(name)
    if hardcoded_keys:
        issues.append(
            _doctor_issue(
                "warning",
                "secrets",
                "model_api_key",
                "model configs include direct api_key values; prefer secrets providers or environment variables",
                models=hardcoded_keys,
            )
        )

    snapshot = model_registry.snapshot()
    response_cache_type = snapshot.get("response_cache_type")
    return {
        "response_cache": {
            "type": response_cache_type,
            "persistent": response_cache_type not in (None, "MemoryCache"),
        },
        "issues": issues,
    }


def _looks_indirect_secret(value: str) -> bool:
    text = str(value)
    return text.startswith("$") or text.startswith("${") or text.startswith("env:")
