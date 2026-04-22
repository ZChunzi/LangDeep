"""Agent registry for agent management."""
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent as create_agent
from langgraph.graph import StateGraph


@dataclass
class AgentMetadata:
    """Agent metadata"""
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)  # Capability tags
    model_name: str = "default"       # Model name to use
    tools: List[str] = field(default_factory=list)  # List of dependent tool names
    system_prompt: Optional[str] = None
    priority: int = 1                 # Priority


class AgentRegistry:
    """Agent registry - centralized management of all agents"""

    _instance = None
    _agents: Dict[str, Any] = {}           # Agent instances
    _metadata: Dict[str, AgentMetadata] = {}  # Agent metadata
    _agent_factories: Dict[str, callable] = {}  # Agent factory functions

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(
        self,
        name: str,
        factory: callable,
        metadata: AgentMetadata
    ) -> None:
        """Register agent factory and metadata"""
        self._agent_factories[name] = factory
        self._metadata[name] = metadata

    def get_agent(self, name: str) -> Any:
        """Get agent instance (lazy loading)"""
        if name not in self._agent_factories:
            raise ValueError(f"Agent '{name}' not registered")

        if name not in self._agents:
            self._agents[name] = self._agent_factories[name]()

        return self._agents[name]

    def create_agent_with_tools(
        self,
        name: str,
        model: BaseChatModel,
        tools: List[Any],
        system_prompt: str
    ) -> Any:
        """Create agent instance using LangChain's create_agent"""
        return create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt
        )

    def get_agents_by_capability(self, capability: str) -> List[str]:
        """Get agent list by capability"""
        return [
            name for name, meta in self._metadata.items()
            if capability in meta.capabilities
        ]

    def list_agents(self) -> List[str]:
        """List all agents"""
        return list(self._metadata.keys())

    def get_metadata(self, name: str) -> Optional[AgentMetadata]:
        """Get agent metadata"""
        return self._metadata.get(name)


agent_registry = AgentRegistry()