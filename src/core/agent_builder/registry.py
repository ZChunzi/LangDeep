"""Registry for metadata-driven agent builders."""

from typing import Dict

from .base import BaseAgentBuilder
from .react_builder import ReActAgentBuilder
from ..errors import AgentBuildError
from ..registry.agent_registry import AgentMetadata


class AgentBuilderRegistry:
    """Maps agent_type names to builders."""

    def __init__(self):
        self._builders: Dict[str, BaseAgentBuilder] = {}
        self.register("react", ReActAgentBuilder())

    def register(self, agent_type: str, builder: BaseAgentBuilder) -> None:
        if not agent_type:
            raise AgentBuildError("agent_type cannot be empty")
        self._builders[agent_type] = builder

    def build(self, metadata: AgentMetadata):
        builder = self._builders.get(metadata.agent_type)
        if builder is None:
            raise AgentBuildError(
                f"Agent builder '{metadata.agent_type}' is not registered",
                context={"available": list(self._builders.keys()), "agent": metadata.name},
            )
        return builder.build(metadata)

    def list_builders(self):
        return list(self._builders.keys())


agent_builder_registry = AgentBuilderRegistry()
