"""Tool registry for tool management."""
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool, tool


@dataclass
class ToolMetadata:
    """Tool metadata"""
    name: str
    description: str
    category: str = "general"        # Category: search / compute / io / external
    tags: List[str] = field(default_factory=list)
    requires_confirmation: bool = False  # Whether manual confirmation is required
    timeout: Optional[int] = None    # Timeout in seconds


class ToolRegistry:
    """Tool registry - centralized management of all registered tools"""

    _instance = None
    _tools: Dict[str, BaseTool] = {}
    _metadata: Dict[str, ToolMetadata] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(
        self,
        tool_obj: BaseTool,
        metadata: Optional[ToolMetadata] = None
    ) -> None:
        """Register tool"""
        name = tool_obj.name
        self._tools[name] = tool_obj
        if metadata:
            self._metadata[name] = metadata

    def get_tool(self, name: str) -> BaseTool:
        """Get single tool"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not registered")
        return self._tools[name]

    def get_tools(
        self,
        names: Optional[List[str]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """Get tool list with filtering by name, category, or tags"""
        result = []
        for name, tool_obj in self._tools.items():
            if names and name not in names:
                continue
            if category:
                meta = self._metadata.get(name)
                if not meta or meta.category != category:
                    continue
            if tags:
                meta = self._metadata.get(name)
                if not meta or not all(t in meta.tags for t in tags):
                    continue
            result.append(tool_obj)
        return result

    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self._tools.keys())

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata"""
        return self._metadata.get(name)


tool_registry = ToolRegistry()