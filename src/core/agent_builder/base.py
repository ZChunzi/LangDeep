"""Agent builder extension points."""

from abc import ABC, abstractmethod
from typing import Any

from ..registry.agent_registry import AgentMetadata


class BaseAgentBuilder(ABC):
    """Build a runnable agent from registered agent metadata."""

    @abstractmethod
    def build(self, metadata: AgentMetadata) -> Any:
        ...


def validate_agent_runnable(instance: Any) -> None:
    """Validate the minimal runnable contract used by LangDeep executors."""
    from ..errors import AgentBuildError

    if instance is None:
        raise AgentBuildError("Agent builder returned None")
    if not callable(getattr(instance, "invoke", None)):
        raise AgentBuildError(
            "Agent instance must expose an invoke(state) method",
            context={"agent_type": type(instance).__name__},
        )
