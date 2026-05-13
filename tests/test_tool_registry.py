"""Unit tests for ToolRegistry."""

import asyncio
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.tools import StructuredTool, tool as lc_tool
from langdeep.core.registry.tool_registry import tool_registry, ToolRegistry, ToolMetadata
from langdeep.core.observability import MetricsCollector
from langdeep.core.errors import (
    ConfigurationError,
    ToolConfirmationRequired,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolWorkspaceError,
)
from langdeep.core.tools import (
    PolicyAwareTool,
    ToolAuditLog,
    ToolExecutionPolicy,
    ToolExecutionRecord,
    wrap_tool,
)

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


def test_get_tool_returns_policy_aware_wrapper():
    @lc_tool
    def wrapped_tool(x: str) -> str:
        """Wrapped tool."""
        return x

    tool_registry.register(wrapped_tool, ToolMetadata(name="wrapped_tool", description="wrapped"))
    wrapped = tool_registry.get_tool("wrapped_tool")
    raw = tool_registry.get_raw_tool("wrapped_tool")
    assert isinstance(wrapped, PolicyAwareTool)
    assert wrapped.name == "wrapped_tool"
    assert raw is wrapped_tool
    assert wrapped.invoke({"x": "ok"}) == "ok"


def test_requires_confirmation_blocks_until_confirmed():
    calls = []

    @lc_tool
    def dangerous_tool(x: str) -> str:
        """Dangerous tool."""
        calls.append(x)
        return f"done:{x}"

    tool_registry.register(
        dangerous_tool,
        ToolMetadata(
            name="dangerous_tool",
            description="dangerous",
            requires_confirmation=True,
        ),
    )

    wrapped = tool_registry.get_tool("dangerous_tool")
    try:
        wrapped.invoke({"x": "no"})
        assert False, "Should require confirmation"
    except ToolConfirmationRequired:
        pass
    assert calls == []

    result = wrapped.invoke(
        {"x": "yes"},
        config={"metadata": {"tool_confirmations": {"dangerous_tool": True}}},
    )
    assert result == "done:yes"
    assert calls == ["yes"]

    records = tool_registry.get_audit_log().list_records("dangerous_tool")
    assert len(records) == 2
    assert records[0].blocked is True
    assert records[1].success is True
    assert records[1].confirmed is True


def test_file_tool_workspace_roots_block_outside_paths(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    tool_registry.set_workspace_roots([str(allowed)])

    @lc_tool
    def read_file(filepath: str) -> str:
        """Read file."""
        return filepath

    tool_registry.register(
        read_file,
        ToolMetadata(
            name="read_file",
            description="read",
            category="file",
        ),
    )

    wrapped = tool_registry.get_tool("read_file")
    inside = allowed / "input.txt"
    assert wrapped.invoke({"filepath": str(inside)}) == str(inside)

    try:
        wrapped.invoke({"filepath": str(outside)})
        assert False, "Should block paths outside configured workspace"
    except ToolWorkspaceError as exc:
        assert "workspace_roots" in exc.context


def test_audit_log_limits_filters_and_clears_records():
    audit_log = ToolAuditLog(max_records=2)
    audit_log.record(ToolExecutionRecord(tool_name="first", success=True, duration_ms=1))
    audit_log.record(ToolExecutionRecord(tool_name="second", success=True, duration_ms=2))
    audit_log.record(ToolExecutionRecord(tool_name="second", success=False, duration_ms=3))

    records = audit_log.list_records()
    assert [record.tool_name for record in records] == ["second", "second"]
    assert len(audit_log.list_records("second")) == 2

    audit_log.clear()
    assert audit_log.list_records() == []


def test_registry_policy_configuration_and_unwrapped_list():
    @lc_tool
    def policy_tool(x: str) -> str:
        """Policy test tool."""
        return x

    tool_registry.register(policy_tool, ToolMetadata(name="policy_tool", description="policy"))

    policy = ToolExecutionPolicy(enforce_confirmation=False, workspace_roots=("/tmp",))
    tool_registry.set_policy(policy)
    assert tool_registry.get_policy() is policy

    raw_tools = tool_registry.get_tools(names=["policy_tool"], enforce_policy=False)
    assert raw_tools == [policy_tool]

    tool_registry.set_workspace_roots(["/var/tmp"])
    assert tuple(tool_registry.get_policy().workspace_roots) == ("/var/tmp",)

    tool_registry.reset_policy()
    assert tool_registry.get_policy().enforce_confirmation is True
    assert tool_registry.get_audit_log().list_records() == []


def test_policy_wrapper_run_paths_and_without_audit_log():
    @lc_tool
    def runnable_tool(x: str) -> str:
        """Runnable tool."""
        return f"run:{x}"

    async def async_runnable_tool(x: str) -> str:
        return f"arun:{x}"

    async_tool = StructuredTool.from_function(
        coroutine=async_runnable_tool,
        name="async_runnable_tool",
        description="Async runnable tool.",
    )

    tool_registry.register(runnable_tool, ToolMetadata(name="runnable_tool", description="run"))
    tool_registry.register(async_tool, ToolMetadata(name="async_runnable_tool", description="arun"))
    wrapped = tool_registry.get_tool("runnable_tool")
    async_wrapped = tool_registry.get_tool("async_runnable_tool")

    assert wrapped._run(x="sync") == "run:sync"
    assert asyncio.run(async_wrapped._arun(x="async")) == "arun:async"
    assert wrap_tool(wrapped) is wrapped

    no_audit = wrap_tool(runnable_tool, audit_log=None)
    assert no_audit.invoke({"x": "quiet"}) == "run:quiet"


def test_async_policy_wrapper_confirms_and_audits_failures():
    calls = []

    async def async_danger_tool(x: str) -> str:
        calls.append(x)
        return f"async:{x}"

    async_tool = StructuredTool.from_function(
        coroutine=async_danger_tool,
        name="async_danger_tool",
        description="Async danger tool.",
    )

    tool_registry.register(
        async_tool,
        ToolMetadata(
            name="async_danger_tool",
            description="async danger",
            requires_confirmation=True,
        ),
    )
    wrapped = tool_registry.get_tool("async_danger_tool")

    async def run_case():
        try:
            await wrapped.ainvoke({"x": "blocked"})
            assert False, "Should require confirmation"
        except ToolConfirmationRequired:
            pass

        result = await wrapped.ainvoke(
            {"x": "allowed"},
            config={"metadata": {"confirmed_tools": ["async_danger_tool"]}},
        )
        return result

    assert asyncio.run(run_case()) == "async:allowed"
    assert calls == ["allowed"]

    records = tool_registry.get_audit_log().list_records("async_danger_tool")
    assert len(records) == 2
    assert records[0].blocked is True
    assert records[1].confirmed is True


def test_workspace_policy_accepts_relative_paths_and_nested_args(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    tool_registry.set_workspace_roots([str(allowed)])
    monkeypatch.chdir(allowed)

    @lc_tool
    def tagged_file_tool(filepath: str) -> str:
        """Tagged file tool."""
        return filepath

    tool_registry.register(
        tagged_file_tool,
        ToolMetadata(
            name="tagged_file_tool",
            description="tagged file",
            tags=["file"],
        ),
    )

    wrapped = tool_registry.get_tool("tagged_file_tool")
    assert wrapped.invoke({"filepath": "relative.txt"}) == "relative.txt"

    nested = {"args": {"filepath": str(allowed / "nested.txt")}}
    assert wrapped._execute_with_policy(nested, confirmed=False, call=lambda: "nested-ok") == "nested-ok"


def test_non_dict_confirmation_metadata_is_ignored():
    calls = []

    @lc_tool
    def strict_confirm_tool(x: str) -> str:
        """Strict confirm tool."""
        calls.append(x)
        return x

    tool_registry.register(
        strict_confirm_tool,
        ToolMetadata(
            name="strict_confirm_tool",
            description="strict",
            requires_confirmation=True,
        ),
    )
    wrapped = tool_registry.get_tool("strict_confirm_tool")

    try:
        wrapped.invoke({"x": "bad"}, config={"metadata": "confirmed"})
        assert False, "Should require structured confirmation metadata"
    except ToolConfirmationRequired:
        pass

    assert calls == []


def test_unregistered_metadata_does_not_apply_file_policy(tmp_path):
    tool_registry.set_workspace_roots([str(tmp_path / "allowed")])

    @lc_tool
    def unknown_category_tool(path: str) -> str:
        """Unknown category tool."""
        return path

    tool_registry.register(unknown_category_tool)
    wrapped = tool_registry.get_tool("unknown_category_tool")

    assert wrapped.invoke({"path": str(tmp_path / "outside.txt")}).endswith("outside.txt")


def test_sync_tool_timeout_blocks_and_audits():
    @lc_tool
    def slow_tool(x: str) -> str:
        """Slow tool."""
        time.sleep(0.05)
        return x

    tool_registry.register(
        slow_tool,
        ToolMetadata(
            name="slow_tool",
            description="slow",
            timeout=0.001,
        ),
    )
    wrapped = tool_registry.get_tool("slow_tool")

    try:
        wrapped.invoke({"x": "late"})
        assert False, "Should time out"
    except ToolTimeoutError as exc:
        assert exc.context["tool"] == "slow_tool"

    records = tool_registry.get_audit_log().list_records("slow_tool")
    assert len(records) == 1
    assert records[0].blocked is True
    assert "exceeded timeout" in records[0].error


def test_async_tool_timeout_blocks_and_audits():
    async def async_slow_tool(x: str) -> str:
        await asyncio.sleep(0.05)
        return x

    async_tool = StructuredTool.from_function(
        coroutine=async_slow_tool,
        name="async_slow_tool",
        description="Async slow tool.",
    )
    tool_registry.register(
        async_tool,
        ToolMetadata(
            name="async_slow_tool",
            description="async slow",
            timeout=0.001,
        ),
    )
    wrapped = tool_registry.get_tool("async_slow_tool")

    async def run_case():
        try:
            await wrapped.ainvoke({"x": "late"})
            assert False, "Should time out"
        except ToolTimeoutError as exc:
            assert exc.context["tool"] == "async_slow_tool"

    asyncio.run(run_case())

    records = tool_registry.get_audit_log().list_records("async_slow_tool")
    assert len(records) == 1
    assert records[0].blocked is True


def test_tool_registry_lifecycle_snapshot_reset_and_duplicate_policy():
    @lc_tool
    def life_tool(x: str) -> str:
        """Lifecycle tool."""
        return x

    metadata = ToolMetadata(name="life_tool", description="life")
    tool_registry.register(life_tool, metadata)
    wrapped = tool_registry.get_tool("life_tool")
    wrapped.invoke({"x": "ok"})

    snapshot = tool_registry.snapshot()
    assert snapshot["namespace"] == "default"
    assert "life_tool" in snapshot["tools"]
    assert snapshot["metadata"]["life_tool"].description == "life"
    assert len(snapshot["audit_records"]) == 1

    try:
        tool_registry.register(life_tool, metadata, replace=False)
        assert False, "Should reject duplicate registration when replace=False"
    except ConfigurationError:
        pass

    tool_registry.reset()
    assert tool_registry.list_tools() == []
    assert tool_registry.get_audit_log().list_records() == []


def test_tool_registry_namespace_isolation_and_metadata_replacement():
    tenant = ToolRegistry.for_namespace("tenant-a")
    tenant.reset()

    @lc_tool
    def tenant_tool(x: str) -> str:
        """Tenant tool."""
        return x

    tenant.register(tenant_tool, ToolMetadata(name="tenant_tool", description="tenant"))
    assert tenant.namespace == "tenant-a"
    assert ToolRegistry("tenant-a") is tenant
    assert "tenant_tool" in tenant.list_tools()
    assert "tenant_tool" not in tool_registry.list_tools()

    tenant.register(tenant_tool)
    assert tenant.get_metadata("tenant_tool") is None
    tenant.reset()


def test_policy_wrapped_tool_records_metrics():
    metrics = MetricsCollector()
    tool_registry.set_metrics_collector(metrics)

    @lc_tool
    def metric_tool(x: str) -> str:
        """Metric tool."""
        return x

    tool_registry.register(metric_tool, ToolMetadata(name="metric_tool", description="metric"))
    wrapped = tool_registry.get_tool("metric_tool")

    assert wrapped.invoke({"x": "ok"}) == "ok"
    collected = metrics.get_metrics()
    assert collected["counters"]["tool.calls|status=success,tool=metric_tool"] == 1
    assert "tool.duration_ms|status=success,tool=metric_tool" in collected["histograms"]
