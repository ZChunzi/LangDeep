"""Agent registry — singleton registry for agent factories and metadata."""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from ..logging import get_logger
from ..errors import AgentNotFoundError

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
    priority: int = 1


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
            self._agents[name] = self._factories[name]()
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


agent_registry = AgentRegistry()
