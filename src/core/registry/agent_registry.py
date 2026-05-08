"""Agent registry — singleton registry for agent factories and metadata."""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from ..logging import get_logger
from ..errors import AgentBuildError, AgentNotFoundError

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
    """Thread-safe-ish singleton for agent registration and lazy instantiation."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: Dict[str, Any] = {}
            cls._instance._metadata: Dict[str, AgentMetadata] = {}
            cls._instance._factories: Dict[str, Callable] = {}
        return cls._instance

    def register(self, name: str, factory: Callable, metadata: AgentMetadata) -> None:
        self._factories[name] = factory
        self._metadata[name] = metadata
        logger.info(
            "Agent registered",
            extra={
                "agent_name": name,
                "capabilities": metadata.capabilities,
                "routing_keywords": metadata.routing_keywords,
            },
        )

    def get_agent(self, name: str) -> Any:
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
            logger.debug("Agent instance created", extra={"agent_name": name})
        return self._agents[name]

    def list_agents(self) -> List[str]:
        return list(self._metadata.keys())

    def get_metadata(self, name: str) -> Optional[AgentMetadata]:
        return self._metadata.get(name)

    def get_agents_by_capability(self, capability: str) -> List[str]:
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

        for name, meta in self._metadata.items():
            missing = [t for t in (meta.tools or []) if t not in registered_tools]
            if missing:
                msg = f"Agent '{name}': tools not registered -> {missing}"
                issues.append(msg)
                logger.warning(msg)

        if not issues:
            logger.info("Agent tool audit passed — all tools are registered")

        return issues


agent_registry = AgentRegistry()
