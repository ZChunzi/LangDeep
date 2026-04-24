"""Task scheduler with cron/interval/condition trigger support."""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from croniter import croniter

from ..logging import get_logger
from ..errors import ConfigurationError

logger = get_logger(__name__)


class TriggerType(Enum):
    CRON = "cron"
    INTERVAL = "interval"
    CONDITION = "condition"
    EVENT = "event"
    ONCE = "once"


@dataclass
class ScheduledTask:
    id: str
    name: str
    trigger_type: TriggerType
    trigger_config: Dict[str, Any]
    workflow: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    timeout: int = 300
    retry_count: int = 3
    retry_delay: int = 60


@dataclass
class ConditionContext:
    variables: Dict[str, Any]
    last_result: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


class TaskScheduler:
    """Schedules and executes workflows on cron/intervals/conditions.

    Requires a FlowOrchestrator instance for workflow execution.
    """

    def __init__(self, orchestrator: "FlowOrchestrator"):
        self._orchestrator = orchestrator
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._condition_checkers: Dict[str, Callable] = {}

    def register_task(self, task: ScheduledTask) -> None:
        self._tasks[task.id] = task
        self._calculate_next_run(task)
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
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started", extra={"task_count": len(self._tasks)})

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        while self._running:
            now = datetime.now()
            for task in list(self._tasks.values()):
                if not task.enabled:
                    continue
                if self._should_run(task, now):
                    self._execute_task(task)
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
        task.last_run = datetime.now()
        logger.info("Scheduled task executing", extra={"task_id": task.id, "task_name": task.name})
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
            logger.info(
                "Scheduled task completed",
                extra={"task_id": task.id, "task_name": task.name},
            )
        except Exception as exc:
            logger.error(
                "Scheduled task failed",
                extra={"task_id": task.id, "task_name": task.name, "error": str(exc)},
                exc_info=True,
            )
            self._handle_error(task, exc)

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
