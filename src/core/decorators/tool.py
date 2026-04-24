"""@regist_tool decorator — registers a tool with metadata in the tool registry."""

from functools import wraps
from typing import Callable, List, Optional

from langchain_core.tools import tool as langchain_tool

from ..registry.tool_registry import tool_registry, ToolMetadata


def regist_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    tags: Optional[List[str]] = None,
    requires_confirmation: bool = False,
    timeout: Optional[int] = None,
):
    """Decorator that wraps a function as a LangChain tool and registers it.

    Usage::

        @regist_tool(name="weather", category="external", tags=["api"])
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"{city}: sunny, 22C"
    """

    def decorator(func: Callable):
        tool_obj = langchain_tool(func)
        if name:
            tool_obj.name = name
        if description:
            tool_obj.description = description

        metadata = ToolMetadata(
            name=tool_obj.name,
            description=tool_obj.description or func.__doc__ or "",
            category=category,
            tags=tags or [],
            requires_confirmation=requires_confirmation,
            timeout=timeout,
        )
        tool_registry.register(tool_obj, metadata)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
