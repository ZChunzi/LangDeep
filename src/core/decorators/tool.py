"""Tool registration decorator."""
from functools import wraps
from typing import Optional, List, Callable
from langchain_core.tools import tool as langchain_tool
from ..registry.tool_registry import tool_registry, ToolMetadata


def regist_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    tags: Optional[List[str]] = None,
    requires_confirmation: bool = False,
    timeout: Optional[int] = None
):
    """
    Tool registration decorator, extending LangChain's native @tool decorator

    Usage:
        @regist_tool(name="weather", category="external", tags=["api"])
        def get_weather(city: str) -> str:
            '''Get weather information for specified city'''
            return f"{city} weather: Sunny, 22°C"
    """
    def decorator(func: Callable):
        # Use LangChain native decorator to create tool
        tool_obj = langchain_tool(func)

        # Override tool name if custom name specified
        if name:
            tool_obj.name = name

        # Override tool description if custom description specified
        if description:
            tool_obj.description = description

        # Register to tool registry
        metadata = ToolMetadata(
            name=tool_obj.name,
            description=tool_obj.description or func.__doc__ or "",
            category=category,
            tags=tags or [],
            requires_confirmation=requires_confirmation,
            timeout=timeout
        )
        tool_registry.register(tool_obj, metadata)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator