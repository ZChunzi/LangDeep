"""Unit tests for ToolRegistry."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.tools import tool as lc_tool
from langdeep.core.registry.tool_registry import tool_registry, ToolRegistry, ToolMetadata
from langdeep.core.errors import ToolNotFoundError

# Import clean helper
from conftest import clean_registries


def setup_function():
    clean_registries()


def test_singleton():
    t1 = ToolRegistry()
    t2 = ToolRegistry()
    assert t1 is t2


def test_register_and_get():
    @lc_tool
    def my_tool(x: str) -> str:
        """My test tool."""
        return x

    tool_registry.register(my_tool, ToolMetadata(
        name="my_tool", description="Test tool", category="test", tags=["demo"],
    ))
    assert "my_tool" in tool_registry.list_tools()
    retrieved = tool_registry.get_tool("my_tool")
    assert retrieved.name == "my_tool"


def test_get_tool_not_found():
    try:
        tool_registry.get_tool("non_existent")
        assert False, "Should raise"
    except ToolNotFoundError:
        pass


def test_get_metadata():
    @lc_tool
    def md_tool(x: str) -> str:
        """MD test tool."""
        return x
    meta = ToolMetadata(name="md_tool", description="desc", category="cat", tags=["a", "b"])
    tool_registry.register(md_tool, meta)
    assert tool_registry.get_metadata("md_tool").category == "cat"


def test_get_tools_filter():
    @lc_tool
    def a_tool(x: str) -> str:
        """Alpha tool."""
        return x
    @lc_tool
    def b_tool(x: str) -> str:
        """Beta tool."""
        return x

    tool_registry.register(a_tool, ToolMetadata(name="a_tool", description="a", category="alpha", tags=["t1"]))
    tool_registry.register(b_tool, ToolMetadata(name="b_tool", description="b", category="beta", tags=["t2"]))

    alphas = tool_registry.get_tools(category="alpha")
    assert len(alphas) == 1
    assert alphas[0].name == "a_tool"

    tagged = tool_registry.get_tools(tags=["t2"])
    assert len(tagged) == 1
    assert tagged[0].name == "b_tool"

    named = tool_registry.get_tools(names=["a_tool", "b_tool"])
    assert len(named) == 2


def test_list_tools_empty():
    clean_registries()
    assert tool_registry.list_tools() == []


def test_register_without_metadata():
    @lc_tool
    def bare_tool(x: str) -> str:
        """Bare tool."""
        return x
    tool_registry.register(bare_tool)
    assert "bare_tool" in tool_registry.list_tools()
    assert tool_registry.get_metadata("bare_tool") is None
