"""Agent registration decorator."""
from functools import wraps
from typing import Optional, List
from ..registry.agent_registry import agent_registry, AgentMetadata


def agent(
    name: Optional[str] = None,
    description: str = "",
    capabilities: Optional[List[str]] = None,
    model: str = "default",
    tools: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    priority: int = 1
):
    """
    Agent registration decorator

    Usage:
        @agent(
            name="math_agent",
            description="Math calculation expert",
            capabilities=["math", "calculation"],
            model="gpt4o",
            tools=["calculate", "solve_equation"]
        )
        def create_math_agent():
            # Build and return agent instance
            pass
    """
    def decorator(func):
        agent_name = name or func.__name__

        metadata = AgentMetadata(
            name=agent_name,
            description=description or func.__doc__ or "",
            capabilities=capabilities or [],
            model_name=model,
            tools=tools or [],
            system_prompt=system_prompt,
            priority=priority
        )

        agent_registry.register(agent_name, func, metadata)
        return func

    return decorator