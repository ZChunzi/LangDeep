"""Tool registry — singleton registry for LangChain tools."""

import copy
import threading
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool

from ..logging import get_logger
from ..errors import ConfigurationError, ToolNotFoundError
from ..observability.metrics import MetricsCollector
from ..tools import ToolAuditLog, ToolExecutionPolicy, wrap_tool

logger = get_logger(__name__)


@dataclass
class ToolMetadata:
    """Metadata describing a registered tool."""
    name: str
    description: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    timeout: Optional[int] = None


class ToolRegistry:
    """Namespace-aware registry for tools with metadata."""

    _instance = None
    _registries: Dict[str, "ToolRegistry"] = {}
    _class_lock = threading.Lock()

    def __new__(cls, namespace: str = "default"):
        namespace = namespace or "default"
        with cls._class_lock:
            if namespace not in cls._registries:
                instance = super().__new__(cls)
                instance._namespace = namespace
                instance._tools: Dict[str, BaseTool] = {}
                instance._metadata: Dict[str, ToolMetadata] = {}
                instance._policy = ToolExecutionPolicy()
                instance._audit_log = ToolAuditLog()
                instance._metrics_collector = None
                instance._tracing_adapter = None
                instance._registry_lock = threading.RLock()
                cls._registries[namespace] = instance
                if namespace == "default":
                    cls._instance = instance
            return cls._registries[namespace]

    @classmethod
    def for_namespace(cls, namespace: str) -> "ToolRegistry":
        """Return an isolated registry for a namespace."""
        return cls(namespace=namespace)

    @property
    def namespace(self) -> str:
        return self._namespace

    def register(
        self,
        tool_obj: BaseTool,
        metadata: Optional[ToolMetadata] = None,
        *,
        replace: bool = True,
    ) -> None:
        name = tool_obj.name
        with self._registry_lock:
            if name in self._tools and not replace:
                raise ConfigurationError(
                    f"Tool '{name}' is already registered",
                    context={"tool": name, "namespace": self._namespace},
                )
            self._tools[name] = tool_obj
            if metadata:
                self._metadata[name] = metadata
            else:
                self._metadata.pop(name, None)
        logger.info(
            "Tool registered",
            extra={
                "tool_name": name,
                "namespace": self._namespace,
                "category": metadata.category if metadata else "general",
            },
        )

    def get_tool(self, name: str, *, enforce_policy: bool = True) -> BaseTool:
        with self._registry_lock:
            if name not in self._tools:
                raise ToolNotFoundError(
                    f"Tool '{name}' is not registered",
                    context={"available": list(self._tools.keys())},
                )
            tool = self._tools[name]
            metadata = self._metadata.get(name)
            policy = self._policy
            audit_log = self._audit_log
            metrics_collector = self._metrics_collector
            tracing_adapter = self._tracing_adapter
        if not enforce_policy or not isinstance(tool, BaseTool):
            return tool
        return wrap_tool(
            tool,
            metadata,
            policy=policy,
            audit_log=audit_log,
            metrics_collector=metrics_collector,
            tracing_adapter=tracing_adapter,
        )

    def get_raw_tool(self, name: str) -> BaseTool:
        """Return a registered tool without LangDeep policy wrapping."""
        return self.get_tool(name, enforce_policy=False)

    def get_tools(
        self,
        names: Optional[List[str]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        enforce_policy: bool = True,
    ) -> List[BaseTool]:
        with self._registry_lock:
            entries = [
                (name, tool_obj, self._metadata.get(name))
                for name, tool_obj in self._tools.items()
            ]
            policy = self._policy
            audit_log = self._audit_log
            metrics_collector = self._metrics_collector
            tracing_adapter = self._tracing_adapter

        result = []
        for name, tool_obj, meta in entries:
            if names and name not in names:
                continue
            if category and (not meta or meta.category != category):
                continue
            if tags and (not meta or not all(t in meta.tags for t in tags)):
                continue
            if enforce_policy and isinstance(tool_obj, BaseTool):
                result.append(
                    wrap_tool(
                        tool_obj,
                        meta,
                        policy=policy,
                        audit_log=audit_log,
                        metrics_collector=metrics_collector,
                        tracing_adapter=tracing_adapter,
                    )
                )
            else:
                result.append(tool_obj)
        return result

    def list_tools(self) -> List[str]:
        with self._registry_lock:
            return list(self._tools.keys())

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        with self._registry_lock:
            return self._metadata.get(name)

    def set_policy(self, policy: ToolExecutionPolicy) -> None:
        """Replace the registry-level tool execution policy."""
        with self._registry_lock:
            self._policy = policy

    def get_policy(self) -> ToolExecutionPolicy:
        """Return the registry-level tool execution policy."""
        with self._registry_lock:
            return self._policy

    def reset_policy(self) -> None:
        """Reset policy and audit log to defaults."""
        with self._registry_lock:
            self._policy = ToolExecutionPolicy()
            self._audit_log.clear()

    def set_workspace_roots(self, roots: Sequence[str]) -> None:
        """Configure allowed workspace roots for file-category tools."""
        with self._registry_lock:
            self._policy = ToolExecutionPolicy(
                enforce_confirmation=self._policy.enforce_confirmation,
                workspace_roots=tuple(roots),
                path_argument_names=tuple(self._policy.path_argument_names),
            )

    def get_audit_log(self) -> ToolAuditLog:
        """Return the in-memory tool execution audit log."""
        with self._registry_lock:
            return self._audit_log

    def set_metrics_collector(self, metrics_collector: Optional[MetricsCollector]) -> None:
        """Attach a metrics collector used by policy-wrapped tools."""
        with self._registry_lock:
            self._metrics_collector = metrics_collector

    def get_metrics_collector(self) -> Optional[MetricsCollector]:
        """Return the metrics collector used by policy-wrapped tools."""
        with self._registry_lock:
            return self._metrics_collector

    def set_tracing_adapter(self, tracing_adapter: Optional[Any]) -> None:
        """Attach a tracing adapter used by policy-wrapped tools."""
        with self._registry_lock:
            self._tracing_adapter = tracing_adapter

    def get_tracing_adapter(self) -> Optional[Any]:
        """Return the tracing adapter used by policy-wrapped tools."""
        with self._registry_lock:
            return self._tracing_adapter

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow runtime snapshot with copied metadata and policy."""
        with self._registry_lock:
            return {
                "namespace": self._namespace,
                "tools": dict(self._tools),
                "metadata": copy.deepcopy(self._metadata),
                "policy": copy.deepcopy(self._policy),
                "audit_records": self._audit_log.list_records(),
                "metrics_enabled": self._metrics_collector is not None,
                "tracing_enabled": self._tracing_adapter is not None,
            }

    def reset(self) -> None:
        """Clear registered tools, metadata, policy, and audit records."""
        with self._registry_lock:
            self._tools.clear()
            self._metadata.clear()
            self._policy = ToolExecutionPolicy()
            self._audit_log.clear()
            self._metrics_collector = None
            self._tracing_adapter = None


tool_registry = ToolRegistry()
