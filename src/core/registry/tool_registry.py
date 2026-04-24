"""Tool registry — singleton registry for LangChain tools."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool

from ..logging import get_logger
from ..errors import ToolNotFoundError

logger = get_logger(__name__)


@dataclass
class ToolMetadata:
    """Metadata describing a registered tool."""
    name: str
    description: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    timeout: Optional[int] = None


class ToolRegistry:
    """Singleton registry for tools with metadata."""

    _instance = None
    _tools: Dict[str, BaseTool] = {}
    _metadata: Dict[str, ToolMetadata] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, tool_obj: BaseTool, metadata: Optional[ToolMetadata] = None) -> None:
        name = tool_obj.name
        self._tools[name] = tool_obj
        if metadata:
            self._metadata[name] = metadata
        logger.info("Tool registered", extra={"tool_name": name, "category": metadata.category if metadata else "general"})

    def get_tool(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered",
                context={"available": list(self._tools.keys())},
            )
        return self._tools[name]

    def get_tools(
        self,
        names: Optional[List[str]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BaseTool]:
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
        return list(self._tools.keys())

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        return self._metadata.get(name)


tool_registry = ToolRegistry()
