"""Task scheduler with cron and condition support."""
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from croniter import croniter
import asyncio
import threading
import json
import time


class TriggerType(Enum):
    """Trigger type"""
    CRON = "cron"              # Cron expression trigger
    INTERVAL = "interval"      # Fixed interval trigger
    CONDITION = "condition"    # Condition trigger
    EVENT = "event"            # Event trigger
    ONCE = "once"              # One-time schedule


@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    id: str
    name: str
    trigger_type: TriggerType
    trigger_config: Dict[str, Any]  # cron expression/interval/condition
    workflow: str                    # Workflow to execute
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    timeout: int = 300               # Timeout in seconds
    retry_count: int = 3
    retry_delay: int = 60


@dataclass
class ConditionContext:
    """Condition context"""
    variables: Dict[str, Any]
    last_result: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


class TaskScheduler:
    """
    Task scheduler

    Supports:
    1. Cron expression scheduling (LangGraph native support)
    2. Condition-based scheduling
    3. Integration with flow orchestrator
    """

    def __init__(self, orchestrator: "FlowOrchestrator"):
        self.orchestrator = orchestrator
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._condition_checkers: Dict[str, Callable] = {}

    def register_task(self, task: ScheduledTask) -> None:
        """Register scheduled task"""
        self._tasks[task.id] = task
        self._calculate_next_run(task)

    def list_tasks(self) -> List[ScheduledTask]:
        """List all scheduled tasks"""
        return list(self._tasks.values())

    def register_condition_checker(
        self,
        name: str,
        checker: Callable[[ConditionContext], bool]
    ) -> None:
        """Register condition checker"""
        self._condition_checkers[name] = checker

    def _calculate_next_run(self, task: ScheduledTask) -> None:
        """Calculate next run time"""
        if task.trigger_type == TriggerType.CRON:
            cron = croniter(task.trigger_config["expression"], datetime.now())
            task.next_run = cron.get_next(datetime)
        elif task.trigger_type == TriggerType.INTERVAL:
            interval = task.trigger_config.get("seconds", 3600)
            task.next_run = datetime.now() + timedelta(seconds=interval)
        elif task.trigger_type == TriggerType.ONCE:
            task.next_run = datetime.fromisoformat(task.trigger_config["at"])

    def start(self) -> None:
        """Start scheduler"""
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop scheduler"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _scheduler_loop(self) -> None:
        """Scheduler loop"""
        while self._running:
            now = datetime.now()

            for task_id, task in list(self._tasks.items()):
                if not task.enabled:
                    continue

                should_run = False

                if task.trigger_type in [TriggerType.CRON, TriggerType.INTERVAL, TriggerType.ONCE]:
                    if task.next_run and task.next_run <= now:
                        should_run = True

                elif task.trigger_type == TriggerType.CONDITION:
                    condition_name = task.trigger_config.get("condition")
                    context = ConditionContext(
                        variables=task.trigger_config.get("variables", {})
                    )
                    checker = self._condition_checkers.get(condition_name)
                    if checker and checker(context):
                        should_run = True

                if should_run:
                    self._execute_task(task)
                    self._calculate_next_run(task)

            time.sleep(1)  # Check every second

    def _execute_task(self, task: ScheduledTask) -> None:
        """Execute task"""
        try:
            task.last_run = datetime.now()

            # Call flow orchestrator to execute workflow
            result = self.orchestrator.invoke(
                user_input=task.params.get("user_input", ""),
                context={
                    "task_id": task.id,
                    "task_name": task.name,
                    "scheduled": True,
                    **task.params.get("context", {})
                }
            )

            # Record execution result
            self._log_execution(task, result)

        except Exception as e:
            # Error handling and retry
            self._handle_error(task, e)

    def _handle_error(self, task: ScheduledTask, error: Exception) -> None:
        """Handle execution error"""
        task.retry_count -= 1
        if task.retry_count > 0:
            # Schedule retry
            task.next_run = datetime.now() + timedelta(seconds=task.retry_delay)
        else:
            task.enabled = False
            self._log_error(task, error)

    def _log_execution(self, task: ScheduledTask, result: Any) -> None:
        """Log execution"""
        # Implement logging
        pass

    def _log_error(self, task: ScheduledTask, error: Exception) -> None:
        """Log error"""
        # Implement error logging
        pass

    async def aexecute_now(self, task_id: str) -> Any:
        """Execute specified task immediately (async)"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        return await self.orchestrator.ainvoke(
            user_input=task.params.get("user_input", ""),
            context={
                "task_id": task.id,
                "task_name": task.name,
                "manual_trigger": True,
                **task.params.get("context", {})
            }
        )