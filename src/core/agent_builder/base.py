"""Agent builder extension points."""

import asyncio
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
    if not _has_invoke(instance) and not _has_ainvoke(instance):
        raise AgentBuildError(
            "Agent instance must expose invoke(state), ainvoke(state), or both",
            context={"agent_type": type(instance).__name__},
        )


def invoke_agent_runnable(instance: Any, state: Any) -> Any:
    """Invoke an agent from synchronous code.

    Async-only agents can be bridged when no event loop is already running.
    If a loop is active, callers must use ``ainvoke_agent_runnable`` so the
    coroutine can be awaited instead of blocking the loop.
    """
    from ..errors import AgentBuildError

    invoke = getattr(instance, "invoke", None)
    if callable(invoke):
        return invoke(state)

    ainvoke = getattr(instance, "ainvoke", None)
    if callable(ainvoke):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(ainvoke(state))
        raise AgentBuildError(
            "Agent instance only exposes ainvoke(state); use an async execution path",
            context={"agent_type": type(instance).__name__},
        )

    validate_agent_runnable(instance)


async def ainvoke_agent_runnable(instance: Any, state: Any) -> Any:
    """Invoke an agent from asynchronous code, falling back to sync invoke."""
    ainvoke = getattr(instance, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(state)

    invoke = getattr(instance, "invoke", None)
    if callable(invoke):
        return invoke(state)

    validate_agent_runnable(instance)


def is_async_only_agent(instance: Any) -> bool:
    """Return True when the agent only supports the async runnable contract."""
    return _has_ainvoke(instance) and not _has_invoke(instance)


def _has_invoke(instance: Any) -> bool:
    return callable(getattr(instance, "invoke", None))


def _has_ainvoke(instance: Any) -> bool:
    return callable(getattr(instance, "ainvoke", None))
