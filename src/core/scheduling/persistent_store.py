"""Persistent store for scheduled tasks.

Uses a CacheBackend for storage, allowing in-memory (default) or
custom backends registered with the @cache decorator.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..cache import BaseCacheBackend, MemoryCache
from .models import ScheduledTask, TriggerType


def _serialize_task(task: ScheduledTask) -> dict:
    """Serialize a ScheduledTask to a JSON-compatible dict."""
    return {
        "id": task.id,
        "name": task.name,
        "trigger_type": task.trigger_type.value,
        "trigger_config": json.dumps(task.trigger_config, default=str),
        "workflow": task.workflow,
        "params": json.dumps(task.params, default=str),
        "enabled": task.enabled,
        "last_run": task.last_run.isoformat() if task.last_run else None,
        "next_run": task.next_run.isoformat() if task.next_run else None,
        "timeout": task.timeout,
        "retry_count": task.retry_count,
        "retry_delay": task.retry_delay,
    }


def _deserialize_task(data: dict) -> ScheduledTask:
    """Reconstruct a ScheduledTask from a dict."""
    trigger_config_raw = data.get("trigger_config", "{}")
    if isinstance(trigger_config_raw, str):
        trigger_config = json.loads(trigger_config_raw)
    else:
        trigger_config = trigger_config_raw

    params_raw = data.get("params", "{}")
    if isinstance(params_raw, str):
        params = json.loads(params_raw)
    else:
        params = params_raw

    return ScheduledTask(
        id=data["id"],
        name=data["name"],
        trigger_type=TriggerType(data["trigger_type"]),
        trigger_config=trigger_config,
        workflow=data["workflow"],
        params=params,
        enabled=bool(data.get("enabled", True)),
        last_run=_parse_dt(data.get("last_run")),
        next_run=_parse_dt(data.get("next_run")),
        timeout=int(data.get("timeout", 300)),
        retry_count=int(data.get("retry_count", 3)),
        retry_delay=int(data.get("retry_delay", 60)),
    )


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value:
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            pass
    return None


class TaskStore:
    """Persistent task storage using a CacheBackend.

    Defaults to in-memory storage. Pass any ``BaseCacheBackend``
    instance (e.g. one registered via ``@cache``) for persistent storage.
    """

    def __init__(self, backend: Optional[BaseCacheBackend] = None):
        self._backend = backend or MemoryCache()
        self._task_list_key = "_scheduled_tasks"

    def save_task(self, task: ScheduledTask) -> None:
        """Persist a single task."""
        key = f"task:{task.id}"
        self._backend.set(key, _serialize_task(task))

    def delete_task(self, task_id: str) -> None:
        """Remove a task from storage."""
        self._backend.delete(f"task:{task_id}")

    def load_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Load a single task by ID."""
        data = self._backend.get(f"task:{task_id}")
        if data is None:
            return None
        if isinstance(data, dict):
            return _deserialize_task(data)
        return None

    def load_all(self) -> List[ScheduledTask]:
        """Load all persisted tasks."""
        # In a production backend this would use a proper scan/query.
        # For the MemoryCache, we use a stored list of keys.
        keys = self._backend.get(self._task_list_key)
        if keys is None:
            return []
        tasks = []
        for task_id in keys:
            task = self.load_task(task_id)
            if task is not None:
                tasks.append(task)
        return tasks

    def save_all(self, tasks: List[ScheduledTask]) -> None:
        """Bulk save all tasks."""
        keys = []
        for task in tasks:
            self.save_task(task)
            keys.append(task.id)
        self._backend.set(self._task_list_key, keys)

    def count(self) -> int:
        keys = self._backend.get(self._task_list_key)
        return len(keys) if keys else 0

    def clear(self) -> None:
        keys = self._backend.get(self._task_list_key) or []
        for task_id in keys:
            self._backend.delete(f"task:{task_id}")
        self._backend.delete(self._task_list_key)
