"""Task executor with retry, concurrency control, and dependency ordering."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..logging import get_logger, get_trace_id
from ..agent_builder import ainvoke_agent_runnable, invoke_agent_runnable
from ..errors import TaskExecutionError, CircularDependencyError
from ..execution.execution_policy import ExecutionPolicy
from ..observability.metrics import MetricsCollector
from ..registry.agent_registry import agent_registry
from ..registry.tool_registry import tool_registry

logger = get_logger(__name__)

# ── Uniform result wrappers ─────────────────────────────────────────────────────

def ok(data: str) -> Dict[str, Any]:
    return {"success": True, "data": data, "error": ""}


def err(msg: str) -> Dict[str, Any]:
    return {"success": False, "data": "", "error": msg}


def skipped(msg: str) -> Dict[str, Any]:
    return {"success": False, "data": "", "error": msg, "status": "skipped"}


def waiting_confirmation(task: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "success": False,
        "data": "",
        "error": "",
        "status": "waiting_confirmation",
        "task_id": task.get("id", "unknown"),
        "agent": task.get("agent"),
        "tools": list(task.get("tools") or []),
        "input": dict(task.get("input") or {}),
        "reason": reason,
    }


# ── Extension point ──────────────────────────────────────────────────────────────

class TaskRunner(ABC):
    """Pluggable task runner — override to change retry/execution behaviour."""

    @abstractmethod
    def run(
        self,
        task: Dict[str, Any],
        clean_messages: List[BaseMessage],
        state: Dict[str, Any],
        previous_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def arun(
        self,
        task: Dict[str, Any],
        clean_messages: List[BaseMessage],
        state: Dict[str, Any],
        previous_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...


class RetryTaskRunner(TaskRunner):
    """Executes a task via the agent registry with exponential-backoff retry.

    Automatically injects ``previous_results`` as SystemMessage context
    so the agent can reference earlier task outputs.
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout: float = 30.0,
        retry_on: Optional[List[str]] = None,
        retry_backoff: str = "exponential",
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_on = retry_on or []
        self.retry_backoff = retry_backoff
        self._metrics = metrics_collector

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _inject_context(
        clean_messages: List[BaseMessage],
        previous_results: Dict[str, Any],
    ) -> List[BaseMessage]:
        """Prepend a summary of previous task results as SystemMessage."""
        if not previous_results:
            return clean_messages

        lines = ["Previous task results (for context):"]
        for task_id, result in previous_results.items():
            text = result.get("data", "") if isinstance(result, dict) else str(result)
            if text and not text.startswith("Agent ") and "error" not in str(text).lower():
                lines.append(f"  [{task_id}]: {str(text)[:300]}")
        if len(lines) == 1:
            return clean_messages

        return [SystemMessage(content="\n".join(lines))] + list(clean_messages)

    # ── sync ───────────────────────────────────────────────────────────

    def run(
        self,
        task: Dict[str, Any],
        clean_messages: List[BaseMessage],
        state: Dict[str, Any],
        previous_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_name = task.get("agent")
        task_id = task.get("id", "unknown")

        if not agent_name or agent_name not in agent_registry.list_agents():
            msg = f"Agent '{agent_name}' not registered"
            logger.warning("Task cannot run", extra={"task_id": task_id, "agent": agent_name})
            return err(msg)

        messages = self._inject_context(clean_messages, previous_results)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            attempt_started = time.monotonic()
            self._record_attempt(agent_name, "sync")
            try:
                agent_instance = agent_registry.get_agent(agent_name)
                resp = invoke_agent_runnable(agent_instance, {
                    "messages": messages,
                    "task_context": {
                        **state.get("task_context", {}),
                        "task": task,
                        "previous_results": previous_results,
                    },
                })
                content = _extract_agent_answer(resp)
                logger.info(
                    "Task succeeded",
                    extra={"task_id": task_id, "agent": agent_name, "attempt": attempt},
                )
                self._record_attempt_result(agent_name, "sync", "success", attempt_started)
                return ok(content)
            except Exception as exc:
                last_error = exc
                self._record_attempt_result(agent_name, "sync", "failure", attempt_started)
                should_retry = self._should_retry(exc)
                wait = self._retry_wait(attempt)
                logger.warning(
                    "Task failed",
                    extra={
                        "task_id": task_id, "agent": agent_name,
                        "attempt": attempt, "error": str(exc), "retry_wait_s": wait,
                        "will_retry": should_retry and attempt < self.max_retries,
                    },
                )
                if should_retry and attempt < self.max_retries:
                    self._record_retry(agent_name, "sync")
                    time.sleep(wait)
                else:
                    break

        return err(f"Max retries ({self.max_retries}) exhausted. Last error: {last_error}")

    # ── async ──────────────────────────────────────────────────────────

    async def arun(
        self,
        task: Dict[str, Any],
        clean_messages: List[BaseMessage],
        state: Dict[str, Any],
        previous_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_name = task.get("agent")
        task_id = task.get("id", "unknown")

        if not agent_name or agent_name not in agent_registry.list_agents():
            msg = f"Agent '{agent_name}' not registered"
            logger.warning("Task cannot run", extra={"task_id": task_id, "agent": agent_name})
            return err(msg)

        messages = self._inject_context(clean_messages, previous_results)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            attempt_started = time.monotonic()
            self._record_attempt(agent_name, "async")
            try:
                agent_instance = agent_registry.get_agent(agent_name)
                resp = await asyncio.wait_for(
                    ainvoke_agent_runnable(agent_instance, {
                        "messages": messages,
                        "task_context": {
                            **state.get("task_context", {}),
                            "task": task,
                            "previous_results": previous_results,
                        },
                    }),
                    timeout=self.timeout,
                )
                content = _extract_agent_answer(resp)
                logger.info(
                    "Task succeeded (async)",
                    extra={"task_id": task_id, "agent": agent_name, "attempt": attempt},
                )
                self._record_attempt_result(agent_name, "async", "success", attempt_started)
                return ok(content)
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Timeout ({self.timeout}s)")
                self._record_attempt_result(agent_name, "async", "timeout", attempt_started)
                logger.warning(
                    "Task timed out",
                    extra={"task_id": task_id, "agent": agent_name, "attempt": attempt, "timeout_s": self.timeout},
                )
            except Exception as exc:
                last_error = exc
                self._record_attempt_result(agent_name, "async", "failure", attempt_started)
                logger.warning(
                    "Task failed (async)",
                    extra={"task_id": task_id, "agent": agent_name, "attempt": attempt, "error": str(exc)},
                )
            if self._should_retry(last_error) and attempt < self.max_retries:
                self._record_retry(agent_name, "async")
                await asyncio.sleep(self._retry_wait(attempt))
            else:
                break

        return err(f"Max retries ({self.max_retries}) exhausted. Last error: {last_error}")

    def _should_retry(self, exc: Any) -> bool:
        if not self.retry_on:
            return True
        name = exc.__class__.__name__ if hasattr(exc, "__class__") else type(exc).__name__
        return name in self.retry_on

    def _retry_wait(self, attempt: int) -> int:
        if self.retry_backoff == "fixed":
            return 1
        return 2 ** (attempt - 1)

    def _record_attempt(self, agent_name: str, mode: str) -> None:
        if self._metrics is not None:
            self._metrics.counter(
                "execution.task_attempts",
                tags={"agent": agent_name, "mode": mode},
            )

    def _record_attempt_result(
        self,
        agent_name: str,
        mode: str,
        status: str,
        started: float,
    ) -> None:
        if self._metrics is None:
            return
        tags = {"agent": agent_name, "mode": mode, "status": status}
        self._metrics.counter("execution.task_attempt_results", tags=tags)
        self._metrics.histogram(
            "execution.task_attempt_duration_ms",
            (time.monotonic() - started) * 1000,
            tags=tags,
        )

    def _record_retry(self, agent_name: str, mode: str) -> None:
        if self._metrics is not None:
            self._metrics.counter("execution.retries", tags={"agent": agent_name, "mode": mode})


# ── Executor ─────────────────────────────────────────────────────────────────────

class Executor:
    """Executes workflow tasks respecting dependency order and concurrency policy."""

    def __init__(
        self,
        task_runner: Optional[TaskRunner] = None,
        policy: Optional[ExecutionPolicy] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self._policy = policy or ExecutionPolicy()
        self._metrics = metrics_collector
        self._runner = task_runner or RetryTaskRunner(
            max_retries=self._policy.max_retries if policy else max_retries,
            timeout=self._policy.timeout_seconds if policy else timeout,
            retry_on=self._policy.retry_on,
            retry_backoff=self._policy.retry_backoff,
            metrics_collector=metrics_collector,
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all pending tasks from the workflow plan."""
        started = time.monotonic()
        if self._metrics is not None:
            self._metrics.counter("execution.requests", tags={"strategy": self._policy.strategy})
        workflow_plan = state.get("workflow_plan") or []
        pending = [t for t in workflow_plan if t.get("status") != "completed"]
        if self._metrics is not None:
            self._metrics.histogram("execution.pending_tasks", float(len(pending)))

        if not pending:
            self._record_execution_complete(started, {}, status="no_pending")
            return {"messages": [AIMessage(content="All tasks completed")]}

        registered = set(agent_registry.list_agents())
        clean_msgs = _clean_messages(state["messages"])
        results: Dict[str, Any] = {}

        max_rounds = len(pending) + 1
        remaining = list(pending)

        for _ in range(max_rounds):
            if not remaining:
                break

            ready = [t for t in remaining if _dependencies_satisfied(t, results)]
            if not ready:
                for t in remaining:
                    tid = t.get("id", "unknown")
                    results[tid] = skipped(f"Dependency unsatisfied; skipping task {tid}")
                    logger.warning("Dependency skipped", extra={"task_id": tid})
                break

            batch_results = self._run_batch(ready, registered, clean_msgs, state, results)
            results.update(batch_results)
            remaining = [t for t in remaining if t.get("id") not in batch_results]
            if self._policy.fail_fast and any(not r.get("success") for r in batch_results.values()):
                for t in remaining:
                    tid = t.get("id", "unknown")
                    results[tid] = skipped(f"Fail-fast enabled; skipping task {tid}")
                break

        from .planner import update_plan_status

        flat = {
            k: (v["data"] if v.get("success") else v.get("error", ""))
            for k, v in results.items()
        }
        self._record_execution_complete(started, results, status="completed")
        return {
            "agent_results": flat,
            "workflow_plan": update_plan_status(workflow_plan, results),
        }

    def _run_batch(
        self,
        tasks: List[Dict],
        registered: Set[str],
        clean_msgs: List[BaseMessage],
        state: Dict[str, Any],
        previous: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._metrics is not None:
            self._metrics.counter(
                "execution.batches",
                tags={"strategy": self._policy.strategy},
            )
            self._metrics.histogram(
                "execution.batch_size",
                float(len(tasks)),
                tags={"strategy": self._policy.strategy},
            )
        if self._policy.strategy == "sequential":
            results = {}
            for task in tasks:
                tid = task.get("id", "unknown")
                confirmation = _confirmation_required(task)
                if confirmation:
                    results[tid] = confirmation
                    self._record_confirmation_wait(task)
                    continue
                try:
                    results[tid] = self._runner.run(task, clean_msgs, state, previous)
                except Exception as exc:
                    results[tid] = err(str(exc))
            return results

        sorted_tasks = (
            sorted(tasks, key=lambda t: t.get("priority", 0), reverse=True)
            if self._policy.strategy == "priority_queue"
            else list(tasks)
        )

        # If already inside a running event loop, offload sync work to a thread pool.
        # Otherwise use asyncio.run() for concurrent I/O-bound execution.
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._policy.max_concurrency
            ) as pool:
                return _threaded_batch(
                    pool, sorted_tasks, self._runner, clean_msgs, state, previous,
                    self._policy.max_concurrency,
                )
        except RuntimeError:
            pass

        return asyncio.run(
            _async_batch(sorted_tasks, self._runner, clean_msgs, state, previous, self._policy.max_concurrency)
        )

    def _record_execution_complete(
        self,
        started: float,
        results: Dict[str, Any],
        *,
        status: str,
    ) -> None:
        if self._metrics is None:
            return
        self._metrics.histogram(
            "execution.duration_ms",
            (time.monotonic() - started) * 1000,
            tags={"status": status, "strategy": self._policy.strategy},
        )
        for result in results.values():
            result_status = _result_status(result)
            self._metrics.counter("execution.task_results", tags={"status": result_status})
            if result_status == "waiting_confirmation":
                self._metrics.counter("execution.confirmation_waits")

    def _record_confirmation_wait(self, task: Dict[str, Any]) -> None:
        if self._metrics is not None:
            self._metrics.counter(
                "execution.confirmation_waits",
                tags={"agent": task.get("agent", "unknown")},
            )


# ── Batch execution helpers ──────────────────────────────────────────────────────

def _threaded_batch(
    pool, tasks, runner, clean_msgs, state, previous, max_concurrency,
) -> Dict[str, Any]:
    import concurrent.futures
    futures = {}
    results = {}
    for task in tasks:
        confirmation = _confirmation_required(task)
        if confirmation:
            results[task.get("id", "unknown")] = confirmation
            continue
        f = pool.submit(runner.run, task, clean_msgs, state, previous)
        futures[f] = task.get("id", "unknown")
    for f, tid in futures.items():
        try:
            results[tid] = f.result(timeout=120)
        except Exception as exc:
            results[tid] = err(str(exc))
    return results


async def _async_batch(
    tasks, runner, clean_msgs, state, previous, max_concurrency,
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(max_concurrency)

    async def bounded(task):
        async with sem:
            confirmation = _confirmation_required(task)
            if confirmation:
                return confirmation
            return await runner.arun(task, clean_msgs, state, previous)

    gathered = await asyncio.gather(*[bounded(t) for t in tasks], return_exceptions=True)
    results = {}
    for task, result in zip(tasks, gathered):
        tid = task.get("id", "unknown")
        results[tid] = err(str(result)) if isinstance(result, Exception) else result
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _clean_messages(messages, max_messages: int = 80) -> List[BaseMessage]:
    """Clean messages while preserving tool-call chain integrity.

    Unlike the original (which deleted ALL AIMessages with tool_calls),
    this version keeps every message that carries signal:
      - AIMessage with tool_calls  → needed by ReAct for multi-turn loops
      - ToolMessage                → tool invocation results
      - AIMessage with content     → final / intermediate LLM responses
      - HumanMessage / SystemMessage → conversation context

    Only genuinely empty AIMessages (no content, no tool_calls) are dropped.
    """
    cleaned: List[BaseMessage] = []
    for m in messages:
        # ReAct loop invariants — never strip these
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            cleaned.append(m)
            continue
        if isinstance(m, ToolMessage):
            cleaned.append(m)
            continue
        # Content-carrying messages
        if isinstance(m, AIMessage) and m.content:
            cleaned.append(m)
            continue
        if isinstance(m, (HumanMessage, SystemMessage)):
            cleaned.append(m)
            continue

    # Bound context window to prevent token-limit overflow
    if len(cleaned) > max_messages:
        cleaned = cleaned[:2] + cleaned[-(max_messages - 2):]

    return cleaned


def _dependencies_satisfied(task: Dict, results: Dict[str, Any]) -> bool:
    return all(dep in results and results[dep].get("success", False) for dep in (task.get("depends_on") or []))


def _confirmation_required(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if task.get("requires_confirmation"):
        return waiting_confirmation(task, "task_requires_confirmation")

    for tool_name in task.get("tools") or []:
        meta = tool_registry.get_metadata(tool_name)
        if meta and meta.requires_confirmation:
            return waiting_confirmation(task, f"tool_requires_confirmation:{tool_name}")
    return None


def _result_status(result: Any) -> str:
    if not isinstance(result, dict):
        return "unknown"
    status = result.get("status")
    if status:
        return str(status)
    return "success" if result.get("success") else "failure"


def _extract_agent_answer(response: Any) -> str:
    if isinstance(response, dict) and "messages" in response:
        msgs = response["messages"]
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                return m.content
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content:
                return m.content
    if isinstance(response, str):
        return response
    return str(response)
