"""@agent decorator — registers an agent factory with metadata."""

from functools import wraps
from typing import List, Optional

from ..registry.agent_registry import agent_registry, AgentMetadata


def agent(
    name: Optional[str] = None,
    description: str = "",
    capabilities: Optional[List[str]] = None,
    routing_keywords: Optional[List[str]] = None,
    model: str = "default",
    tools: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    prompt_path: Optional[str] = None,
    priority: int = 1,
    auto_build: bool = False,
    agent_type: str = "react",
):
    """Decorator that registers an agent factory function.

    Usage::

        @agent(
            name="math_agent",
            description="Solves math problems",
            capabilities=["math", "calculation"],
            routing_keywords=["calculate", "math", "equation"],
            model="gpt4o",
            tools=["calculator"],
            auto_build=True,
        )
        def create_math_agent():
            ...
    """

    def decorator(func):
        agent_name = name or func.__name__
        metadata = AgentMetadata(
            name=agent_name,
            description=description or func.__doc__ or "",
            capabilities=capabilities or [],
            routing_keywords=routing_keywords or [],
            model_name=model,
            tools=tools or [],
            system_prompt=system_prompt,
            prompt_path=prompt_path,
            priority=priority,
            auto_build=auto_build,
            agent_type=agent_type,
        )
        agent_registry.register(agent_name, func, metadata)
        return func

    return decorator
