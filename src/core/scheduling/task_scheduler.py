"""Task scheduler with cron/interval/condition trigger support."""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from croniter import croniter

from ..logging import get_logger, get_trace_id
from ..errors import ConfigurationError
from .models import ScheduledTask, TriggerType, ConditionContext
from .worker_pool import WorkerPool
from .persistent_store import TaskStore
from .audit_log import AuditLog

if TYPE_CHECKING:
    from ..orchestrator import FlowOrchestrator

logger = get_logger(__name__)


class TaskScheduler:
    """Schedules and executes workflows on cron/intervals/conditions.

    Requires a FlowOrchestrator instance for workflow execution.
    """

    def __init__(
        self,
        orchestrator: "FlowOrchestrator",
        task_store: Optional[TaskStore] = None,
        worker_pool: Optional[WorkerPool] = None,
        audit_log: Optional[AuditLog] = None,
        auto_recover: bool = True,
        graceful_timeout: float = 10.0,
    ):
        self._orchestrator = orchestrator
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._condition_checkers: Dict[str, Callable] = {}
        # New components
        self._task_store = task_store or TaskStore()
        self._worker_pool = worker_pool or WorkerPool(max_workers=4)
        self._audit_log = audit_log or AuditLog()
        self._auto_recover = auto_recover
        self._graceful_timeout = graceful_timeout

    def register_task(self, task: ScheduledTask) -> None:
        self._tasks[task.id] = task
        self._calculate_next_run(task)
        self._task_store.save_task(task)
        logger.info(
            "Scheduled task registered",
            extra={
                "task_id": task.id,
                "task_name": task.name,
                "trigger_type": task.trigger_type.value,
                "next_run": str(task.next_run) if task.next_run else None,
            },
        )

    def list_tasks(self) -> List[ScheduledTask]:
        return list(self._tasks.values())

    def register_condition_checker(self, name: str, checker: Callable) -> None:
        self._condition_checkers[name] = checker
        logger.debug("Condition checker registered", extra={"checker_name": name})

    def start(self) -> None:
        if self._auto_recover:
            self._recover_tasks()
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Scheduler started",
            extra={
                "task_count": len(self._tasks),
                "recovered": self._auto_recover,
            },
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._graceful_timeout)
        self._worker_pool.shutdown(wait=True, timeout=self._graceful_timeout)
        logger.info("Scheduler stopped")

    def _recover_tasks(self) -> None:
        """Load persisted tasks from TaskStore and re-register them."""
        persisted = self._task_store.load_all()
        for task in persisted:
            self._tasks[task.id] = task
            self._calculate_next_run(task)
            logger.info(
                "Recovered scheduled task",
                extra={
                    "task_id": task.id,
                    "task_name": task.name,
                    "trigger_type": task.trigger_type.value,
                },
            )
        if persisted:
            logger.info(
                "Task recovery complete",
                extra={"recovered_count": len(persisted)},
            )

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        while self._running:
            now = datetime.now()
            for task in list(self._tasks.values()):
                if not task.enabled:
                    continue
                if self._should_run(task, now):
                    self._execute_task_async(task)
                    self._calculate_next_run(task)
            time.sleep(1)

    def _should_run(self, task: ScheduledTask, now: datetime) -> bool:
        if task.trigger_type in (TriggerType.CRON, TriggerType.INTERVAL, TriggerType.ONCE):
            return task.next_run is not None and task.next_run <= now
        if task.trigger_type == TriggerType.CONDITION:
            checker_name = task.trigger_config.get("condition")
            checker = self._condition_checkers.get(checker_name)
            if checker:
                ctx = ConditionContext(variables=task.trigger_config.get("variables", {}))
                return checker(ctx)
        return False

    def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a task synchronously (for testing or immediate execution)."""
        task.last_run = datetime.now()
        execution_id = self._audit_log.log_start(
            task.id, task.name, trace_id=get_trace_id() or "",
        )
        self._run_sync(task, execution_id)

    def _execute_task_async(self, task: ScheduledTask) -> None:
        """Submit task execution to the worker pool (non-blocking, for scheduler loop)."""
        task.last_run = datetime.now()
        execution_id = self._audit_log.log_start(
            task.id, task.name, trace_id=get_trace_id() or "",
        )
        self._worker_pool.submit(task.id, self._run_sync, task, execution_id)

    def _run_sync(self, task: ScheduledTask, execution_id: int) -> Optional[Any]:
        """Core execution logic with error handling and audit logging.

        Errors are handled internally (retry logic, audit log, task persistence).
        Returns the result on success, None on failure.
        """
        try:
            result = self._orchestrator.invoke(
                user_input=task.params.get("user_input", ""),
                context={
                    "task_id": task.id,
                    "task_name": task.name,
                    "scheduled": True,
                    **task.params.get("context", {}),
                },
            )
            duration = int((datetime.now() - task.last_run).total_seconds() * 1000)
            self._audit_log.log_complete(execution_id, str(result)[:200], duration)
            self._task_store.save_task(task)
            logger.info(
                "Scheduled task completed",
                extra={"task_id": task.id, "task_name": task.name, "duration_ms": duration},
            )
            return result
        except Exception as exc:
            duration = int((datetime.now() - task.last_run).total_seconds() * 1000)
            self._audit_log.log_failure(execution_id, str(exc)[:200], duration)
            logger.error(
                "Scheduled task failed",
                extra={"task_id": task.id, "task_name": task.name, "error": str(exc)},
                exc_info=True,
            )
            self._handle_error(task, exc)
            self._task_store.save_task(task)
            return None  # error handled internally, do not propagate

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task and remove it from the schedule."""
        self._worker_pool.cancel(task_id)
        self._tasks.pop(task_id, None)
        self._task_store.delete_task(task_id)
        return True

    def get_execution_history(self, task_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution history for a task."""
        return self._audit_log.get_history(task_id, limit)

    def get_scheduler_stats(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics."""
        return self._audit_log.get_stats(task_id)

    def _handle_error(self, task: ScheduledTask, error: Exception) -> None:
        task.retry_count -= 1
        if task.retry_count > 0:
            task.next_run = datetime.now() + timedelta(seconds=task.retry_delay)
            logger.warning(
                "Scheduling task retry",
                extra={
                    "task_id": task.id,
                    "remaining_retries": task.retry_count,
                    "retry_at": str(task.next_run),
                },
            )
        else:
            task.enabled = False
            logger.error(
                "Task disabled after exhausting retries",
                extra={"task_id": task.id, "task_name": task.name},
            )

    def _calculate_next_run(self, task: ScheduledTask) -> None:
        if task.trigger_type == TriggerType.CRON:
            cron = croniter(task.trigger_config["expression"], datetime.now())
            task.next_run = cron.get_next(datetime)
        elif task.trigger_type == TriggerType.INTERVAL:
            interval = task.trigger_config.get("seconds", 3600)
            task.next_run = datetime.now() + timedelta(seconds=interval)
        elif task.trigger_type == TriggerType.ONCE:
            task.next_run = datetime.fromisoformat(task.trigger_config["at"])
        logger.debug(
            "Next run calculated",
            extra={"task_id": task.id, "next_run": str(task.next_run)},
        )

    async def aexecute_now(self, task_id: str) -> Any:
        task = self._tasks.get(task_id)
        if not task:
            raise ConfigurationError(
                f"Task not found: {task_id}",
                context={"available": list(self._tasks.keys())},
            )
        logger.info("Manual task execution triggered", extra={"task_id": task_id})
        return await self._orchestrator.ainvoke(
            user_input=task.params.get("user_input", ""),
            context={
                "task_id": task.id,
                "task_name": task.name,
                "manual_trigger": True,
                **task.params.get("context", {}),
            },
        )
