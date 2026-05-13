"""Agent registry — singleton registry for agent factories and metadata."""

import copy
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from ..logging import get_logger
from ..errors import AgentBuildError, AgentNotFoundError, ConfigurationError

logger = get_logger(__name__)


@dataclass
class AgentMetadata:
    """Metadata describing a registered agent."""
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    routing_keywords: List[str] = field(default_factory=list)
    model_name: str = "default"
    tools: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    prompt_path: Optional[str] = None
    priority: int = 1
    auto_build: bool = False
    agent_type: str = "react"


class AgentRegistry:
    """Namespace-aware registry for agent registration and lazy instantiation."""

    _instance = None
    _registries: Dict[str, "AgentRegistry"] = {}
    _class_lock = threading.Lock()

    def __new__(cls, namespace: str = "default"):
        namespace = namespace or "default"
        with cls._class_lock:
            if namespace not in cls._registries:
                instance = super().__new__(cls)
                instance._namespace = namespace
                instance._agents: Dict[str, Any] = {}
                instance._metadata: Dict[str, AgentMetadata] = {}
                instance._factories: Dict[str, Callable] = {}
                instance._registry_lock = threading.RLock()
                cls._registries[namespace] = instance
                if namespace == "default":
                    cls._instance = instance
            return cls._registries[namespace]

    @classmethod
    def for_namespace(cls, namespace: str) -> "AgentRegistry":
        """Return an isolated registry for a namespace."""
        return cls(namespace=namespace)

    @property
    def namespace(self) -> str:
        return self._namespace

    def register(
        self,
        name: str,
        factory: Callable,
        metadata: AgentMetadata,
        *,
        replace: bool = True,
    ) -> None:
        with self._registry_lock:
            if name in self._factories and not replace:
                raise ConfigurationError(
                    f"Agent '{name}' is already registered",
                    context={"agent": name, "namespace": self._namespace},
                )
            self._factories[name] = factory
            self._metadata[name] = metadata
            self._agents.pop(name, None)
        logger.info(
            "Agent registered",
            extra={
                "agent_name": name,
                "namespace": self._namespace,
                "capabilities": metadata.capabilities,
                "routing_keywords": metadata.routing_keywords,
            },
        )

    def get_agent(self, name: str) -> Any:
        with self._registry_lock:
            if name not in self._factories:
                raise AgentNotFoundError(
                    f"Agent '{name}' is not registered",
                    context={"available": list(self._factories.keys())},
                )
            if name not in self._agents:
                meta = self._metadata[name]
                factory = self._factories[name]
                instance = factory()

                if instance is None and meta.auto_build:
                    from ..agent_builder import agent_builder_registry

                    instance = agent_builder_registry.build(meta)

                if instance is None:
                    raise AgentBuildError(
                        f"Agent '{name}' factory returned None. Use auto_build=True or return a runnable agent.",
                        context={"agent": name, "auto_build": meta.auto_build},
                    )

                from ..agent_builder import validate_agent_runnable

                validate_agent_runnable(instance)
                self._agents[name] = instance
                logger.debug(
                    "Agent instance created",
                    extra={"agent_name": name, "namespace": self._namespace},
                )
            return self._agents[name]

    def list_agents(self) -> List[str]:
        with self._registry_lock:
            return list(self._metadata.keys())

    def get_metadata(self, name: str) -> Optional[AgentMetadata]:
        with self._registry_lock:
            return self._metadata.get(name)

    def get_agents_by_capability(self, capability: str) -> List[str]:
        with self._registry_lock:
            return [
                name for name, meta in self._metadata.items()
                if capability in meta.capabilities
            ]

    def audit_tools(self) -> List[str]:
        """Verify every registered agent's tool list against the tool registry.

        Returns a list of warning messages (empty = all tools valid).
        """
        from ..registry.tool_registry import tool_registry

        registered_tools = set(tool_registry.list_tools())
        issues: List[str] = []

        with self._registry_lock:
            metadata_items = list(self._metadata.items())

        for name, meta in metadata_items:
            missing = [t for t in (meta.tools or []) if t not in registered_tools]
            if missing:
                msg = f"Agent '{name}': tools not registered -> {missing}"
                issues.append(msg)
                logger.warning(msg)

        if not issues:
            logger.info("Agent tool audit passed — all tools are registered")

        return issues

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow runtime snapshot with copied metadata."""
        with self._registry_lock:
            return {
                "namespace": self._namespace,
                "agents": dict(self._agents),
                "factories": dict(self._factories),
                "metadata": copy.deepcopy(self._metadata),
            }

    def reset(self) -> None:
        """Clear registered agents, metadata, factories, and cached instances."""
        with self._registry_lock:
            self._agents.clear()
            self._metadata.clear()
            self._factories.clear()


agent_registry = AgentRegistry()
