"""Policy-aware wrappers for registered LangChain tools."""

import asyncio
import concurrent.futures
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.tools import BaseTool
from pydantic import Field

from ..errors import ToolConfirmationRequired, ToolTimeoutError, ToolWorkspaceError
from ..logging import get_logger
from ..observability.metrics import MetricsCollector

logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """Runtime policy applied around registered tool execution."""

    enforce_confirmation: bool = True
    workspace_roots: Sequence[str] = field(default_factory=tuple)
    path_argument_names: Sequence[str] = (
        "filepath",
        "file_path",
        "path",
        "directory",
        "workspace_dir",
    )

    def resolved_workspace_roots(self) -> List[Path]:
        return [Path(root).expanduser().resolve() for root in self.workspace_roots]


@dataclass(frozen=True)
class ToolExecutionRecord:
    """Single tool execution audit record."""

    tool_name: str
    success: bool
    duration_ms: int
    requires_confirmation: bool = False
    confirmed: bool = False
    blocked: bool = False
    error: str = ""


class ToolAuditLog:
    """Thread-safe in-memory audit log for tool execution attempts."""

    def __init__(self, max_records: int = 1024):
        self._max_records = max_records
        self._records: List[ToolExecutionRecord] = []
        self._lock = threading.Lock()

    def record(self, entry: ToolExecutionRecord) -> None:
        with self._lock:
            self._records.append(entry)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

    def list_records(self, tool_name: Optional[str] = None) -> List[ToolExecutionRecord]:
        with self._lock:
            records = list(self._records)
        if tool_name is not None:
            return [entry for entry in records if entry.tool_name == tool_name]
        return records

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class PolicyAwareTool(BaseTool):
    """BaseTool proxy that checks LangDeep policy before invoking a tool."""

    original_tool: BaseTool = Field(exclude=True)
    tool_metadata: Optional[Any] = Field(default=None, exclude=True)
    policy: ToolExecutionPolicy = Field(default_factory=ToolExecutionPolicy, exclude=True)
    audit_log: Optional[ToolAuditLog] = Field(default=None, exclude=True)
    metrics_collector: Optional[MetricsCollector] = Field(default=None, exclude=True)
    args_schema: Any = None

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        confirmed = _confirmed_for_tool(self.name, config)
        return self._execute_with_policy(
            input,
            confirmed=confirmed,
            call=lambda: self.original_tool.invoke(input, config=config, **kwargs),
        )

    async def ainvoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        confirmed = _confirmed_for_tool(self.name, config)
        return await self._aexecute_with_policy(
            input,
            confirmed=confirmed,
            call=lambda: self.original_tool.ainvoke(input, config=config, **kwargs),
        )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        tool_input = kwargs if kwargs else args[0] if args else {}
        return self._execute_with_policy(
            tool_input,
            confirmed=False,
            call=lambda: self.original_tool.invoke(tool_input),
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        tool_input = kwargs if kwargs else args[0] if args else {}
        return await self._aexecute_with_policy(
            tool_input,
            confirmed=False,
            call=lambda: self.original_tool.ainvoke(tool_input),
        )

    def _execute_with_policy(self, tool_input: Any, *, confirmed: bool, call) -> Any:
        started = time.monotonic()
        try:
            _validate_tool_policy(self.name, self.tool_metadata, self.policy, tool_input, confirmed)
            result = _call_with_timeout(self.name, self.tool_metadata, call)
            self._audit(started, success=True, confirmed=confirmed)
            return result
        except Exception as exc:
            self._audit(started, success=False, confirmed=confirmed, error=str(exc), blocked=True)
            raise

    async def _aexecute_with_policy(self, tool_input: Any, *, confirmed: bool, call) -> Any:
        started = time.monotonic()
        try:
            _validate_tool_policy(self.name, self.tool_metadata, self.policy, tool_input, confirmed)
            result = await _acall_with_timeout(self.name, self.tool_metadata, call)
            self._audit(started, success=True, confirmed=confirmed)
            return result
        except Exception as exc:
            self._audit(started, success=False, confirmed=confirmed, error=str(exc), blocked=True)
            raise

    def _audit(
        self,
        started: float,
        *,
        success: bool,
        confirmed: bool,
        error: str = "",
        blocked: bool = False,
    ) -> None:
        requires_confirmation = bool(getattr(self.tool_metadata, "requires_confirmation", False))
        duration_ms = int((time.monotonic() - started) * 1000)
        blocked = blocked and not success
        if self.audit_log is not None:
            self.audit_log.record(
                ToolExecutionRecord(
                    tool_name=self.name,
                    success=success,
                    duration_ms=duration_ms,
                    requires_confirmation=requires_confirmation,
                    confirmed=confirmed,
                    blocked=blocked,
                    error=error,
                )
            )
        if self.metrics_collector is not None:
            status = "success" if success else "failure"
            tags = {"tool": self.name, "status": status}
            self.metrics_collector.counter("tool.calls", tags=tags)
            self.metrics_collector.histogram("tool.duration_ms", duration_ms, tags=tags)
            if blocked:
                self.metrics_collector.counter("tool.blocked", tags={"tool": self.name})
            if requires_confirmation:
                self.metrics_collector.counter(
                    "tool.confirmation_required",
                    tags={"tool": self.name, "confirmed": confirmed},
                )


def wrap_tool(
    tool: BaseTool,
    metadata: Optional[Any] = None,
    *,
    policy: Optional[ToolExecutionPolicy] = None,
    audit_log: Optional[ToolAuditLog] = None,
    metrics_collector: Optional[MetricsCollector] = None,
) -> PolicyAwareTool:
    """Return a policy-aware proxy for a registered tool."""
    if isinstance(tool, PolicyAwareTool):
        return tool
    return PolicyAwareTool(
        name=tool.name,
        description=getattr(tool, "description", "") or "",
        args_schema=getattr(tool, "args_schema", None),
        return_direct=getattr(tool, "return_direct", False),
        original_tool=tool,
        tool_metadata=metadata,
        policy=policy or ToolExecutionPolicy(),
        audit_log=audit_log,
        metrics_collector=metrics_collector,
    )


def _validate_tool_policy(
    tool_name: str,
    metadata: Optional[Any],
    policy: ToolExecutionPolicy,
    tool_input: Any,
    confirmed: bool,
) -> None:
    requires_confirmation = bool(getattr(metadata, "requires_confirmation", False))
    if policy.enforce_confirmation and requires_confirmation and not confirmed:
        raise ToolConfirmationRequired(
            f"Tool '{tool_name}' requires confirmation",
            context={"tool": tool_name},
        )

    roots = policy.resolved_workspace_roots()
    if roots and _is_file_tool(metadata):
        _validate_workspace(tool_name, tool_input, policy.path_argument_names, roots)


def _call_with_timeout(tool_name: str, metadata: Optional[Any], call) -> Any:
    timeout = _timeout_seconds(metadata)
    if timeout is None:
        return call()

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=f"langdeep-tool-{tool_name}",
    )
    future = executor.submit(call)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise ToolTimeoutError(
            f"Tool '{tool_name}' exceeded timeout of {timeout:g}s",
            context={"tool": tool_name, "timeout": timeout},
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


async def _acall_with_timeout(tool_name: str, metadata: Optional[Any], call) -> Any:
    timeout = _timeout_seconds(metadata)
    coroutine = call()
    if timeout is None:
        return await coroutine

    try:
        return await asyncio.wait_for(coroutine, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise ToolTimeoutError(
            f"Tool '{tool_name}' exceeded timeout of {timeout:g}s",
            context={"tool": tool_name, "timeout": timeout},
        ) from exc


def _timeout_seconds(metadata: Optional[Any]) -> Optional[float]:
    value = getattr(metadata, "timeout", None)
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _validate_workspace(
    tool_name: str,
    tool_input: Any,
    path_argument_names: Sequence[str],
    roots: Sequence[Path],
) -> None:
    args = _extract_tool_args(tool_input)
    for key in path_argument_names:
        value = args.get(key)
        if value is None:
            continue
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve(strict=False)
        if not any(_is_relative_to(resolved, root) for root in roots):
            raise ToolWorkspaceError(
                f"Tool '{tool_name}' path is outside allowed workspace roots",
                context={
                    "tool": tool_name,
                    "argument": key,
                    "path": str(resolved),
                    "workspace_roots": [str(root) for root in roots],
                },
            )


def _extract_tool_args(tool_input: Any) -> Dict[str, Any]:
    if isinstance(tool_input, dict):
        args = tool_input.get("args")
        if isinstance(args, dict):
            return args
        return tool_input
    return {}


def _is_file_tool(metadata: Optional[Any]) -> bool:
    if metadata is None:
        return False
    if getattr(metadata, "category", None) == "file":
        return True
    tags = getattr(metadata, "tags", None) or []
    return "file" in tags


def _confirmed_for_tool(tool_name: str, config: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(config, dict):
        return False
    metadata = config.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False
    if metadata.get("confirmed") is True:
        return True
    confirmations = metadata.get("tool_confirmations") or {}
    if isinstance(confirmations, dict) and confirmations.get(tool_name) is True:
        return True
    confirmed_tools = metadata.get("confirmed_tools") or []
    return isinstance(confirmed_tools, list) and tool_name in confirmed_tools


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
