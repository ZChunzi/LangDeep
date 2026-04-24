"""Task executor with retry, concurrency control, and dependency ordering."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, BaseMessage

from ..logging import get_logger, get_trace_id
from ..errors import TaskExecutionError, CircularDependencyError
from ..execution.execution_policy import ExecutionPolicy
from ..registry.agent_registry import agent_registry

logger = get_logger(__name__)

# ── Uniform result wrappers ─────────────────────────────────────────────────────

def ok(data: str) -> Dict[str, Any]:
    return {"success": True, "data": data, "error": ""}


def err(msg: str) -> Dict[str, Any]:
    return {"success": False, "data": "", "error": msg}


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
    """Executes a task via the agent registry with exponential-backoff retry."""

    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout

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

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                agent_instance = agent_registry.get_agent(agent_name)
                resp = agent_instance.invoke({
                    "messages": clean_messages,
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
                return ok(content)
            except Exception as exc:
                last_error = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Task failed",
                    extra={
                        "task_id": task_id, "agent": agent_name,
                        "attempt": attempt, "error": str(exc), "retry_wait_s": wait,
                    },
                )
                if attempt < self.max_retries:
                    time.sleep(wait)

        return err(f"Max retries ({self.max_retries}) exhausted. Last error: {last_error}")

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

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                agent_instance = agent_registry.get_agent(agent_name)
                resp = await asyncio.wait_for(
                    agent_instance.ainvoke({
                        "messages": clean_messages,
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
                return ok(content)
            except asyncio.TimeoutError:
                last_error = f"Timeout ({self.timeout}s)"
                logger.warning(
                    "Task timed out",
                    extra={"task_id": task_id, "agent": agent_name, "attempt": attempt, "timeout_s": self.timeout},
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Task failed (async)",
                    extra={"task_id": task_id, "agent": agent_name, "attempt": attempt, "error": str(exc)},
                )
            if attempt < self.max_retries:
                await asyncio.sleep(2 ** (attempt - 1))

        return err(f"Max retries ({self.max_retries}) exhausted. Last error: {last_error}")


# ── Executor ─────────────────────────────────────────────────────────────────────

class Executor:
    """Executes workflow tasks respecting dependency order and concurrency policy."""

    def __init__(
        self,
        task_runner: Optional[TaskRunner] = None,
        policy: Optional[ExecutionPolicy] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self._runner = task_runner or RetryTaskRunner(max_retries, timeout)
        self._policy = policy or ExecutionPolicy()

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all pending tasks from the workflow plan."""
        workflow_plan = state.get("workflow_plan") or []
        pending = [t for t in workflow_plan if t.get("status") != "completed"]

        if not pending:
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
                    results[tid] = err(f"Dependency unsatisfied; skipping task {tid}")
                    logger.warning("Dependency skipped", extra={"task_id": tid})
                break

            batch_results = self._run_batch(ready, registered, clean_msgs, state, results)
            results.update(batch_results)
            remaining = [t for t in remaining if t.get("id") not in batch_results]

        from .planner import update_plan_status

        flat = {
            k: (v["data"] if v.get("success") else v.get("error", ""))
            for k, v in results.items()
        }
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
        if self._policy.strategy == "sequential":
            results = {}
            for task in tasks:
                tid = task.get("id", "unknown")
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

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return _threaded_batch(
                        pool, sorted_tasks, self._runner, clean_msgs, state, previous,
                        self._policy.max_concurrency,
                    )
        except RuntimeError:
            pass

        return asyncio.run(
            _async_batch(sorted_tasks, self._runner, clean_msgs, state, previous, self._policy.max_concurrency)
        )


# ── Batch execution helpers ──────────────────────────────────────────────────────

def _threaded_batch(
    pool, tasks, runner, clean_msgs, state, previous, max_concurrency,
) -> Dict[str, Any]:
    import concurrent.futures
    futures = {}
    for task in tasks:
        f = pool.submit(runner.run, task, clean_msgs, state, previous)
        futures[f] = task.get("id", "unknown")
    results = {}
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
            return await runner.arun(task, clean_msgs, state, previous)

    gathered = await asyncio.gather(*[bounded(t) for t in tasks], return_exceptions=True)
    results = {}
    for task, result in zip(tasks, gathered):
        tid = task.get("id", "unknown")
        results[tid] = err(str(result)) if isinstance(result, Exception) else result
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _clean_messages(messages) -> List[BaseMessage]:
    return [
        m for m in messages
        if not (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
    ]


def _dependencies_satisfied(task: Dict, results: Dict[str, Any]) -> bool:
    return all(dep in results for dep in (task.get("depends_on") or []))


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
