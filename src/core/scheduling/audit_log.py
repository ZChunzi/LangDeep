"""Execution audit log for scheduled tasks.

Records start, completion, and failure of task executions.
Supports statistics and automatic purging of old records.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..cache import BaseCacheBackend, MemoryCache


class AuditLog:
    """Execution history for scheduled tasks.

    Uses a CacheBackend for storage. In-memory by default.
    """

    def __init__(
        self,
        backend: Optional[BaseCacheBackend] = None,
        max_records_per_task: int = 100,
    ):
        self._backend = backend if backend is not None else MemoryCache()
        self._max_records_per_task = max_records_per_task
        self._counter = 0

    def log_start(self, task_id: str, task_name: str, trace_id: str = "") -> int:
        """Record the start of a task execution. Returns execution_id."""
        self._counter += 1
        execution_id = self._counter
        record = {
            "execution_id": execution_id,
            "task_id": task_id,
            "task_name": task_name,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "success": None,
            "result": None,
            "error": None,
            "duration_ms": None,
            "trace_id": trace_id,
        }
        self._backend.set(f"exec:{execution_id}", record)
        self._append_to_history(task_id, execution_id)
        return execution_id

    def log_complete(self, execution_id: int, result: str, duration_ms: int) -> None:
        """Record successful completion."""
        record = self._backend.get(f"exec:{execution_id}")
        if record:
            record["finished_at"] = datetime.now().isoformat()
            record["success"] = True
            record["result"] = str(result)[:500]
            record["duration_ms"] = duration_ms
            self._backend.set(f"exec:{execution_id}", record)

    def log_failure(self, execution_id: int, error: str, duration_ms: int) -> None:
        """Record execution failure."""
        record = self._backend.get(f"exec:{execution_id}")
        if record:
            record["finished_at"] = datetime.now().isoformat()
            record["success"] = False
            record["error"] = str(error)[:500]
            record["duration_ms"] = duration_ms
            self._backend.set(f"exec:{execution_id}", record)

    def get_history(self, task_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution history for a task."""
        history_keys = self._backend.get(f"hist:{task_id}") or []
        records = []
        for eid in history_keys[-limit:]:
            rec = self._backend.get(f"exec:{eid}")
            if rec:
                records.append(rec)
        return records

    def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent executions across all tasks."""
        # Scan all exec: keys (limited for in-memory)
        all_keys = []
        for i in range(1, self._counter + 1):
            rec = self._backend.get(f"exec:{i}")
            if rec:
                all_keys.append(rec)
        all_keys.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return all_keys[:limit]

    def get_stats(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics."""
        if task_id:
            records = self.get_history(task_id, limit=10000)
        else:
            records = self.get_recent(limit=10000)

        total = len(records)
        successes = sum(1 for r in records if r.get("success") is True)
        failures = sum(1 for r in records if r.get("success") is False)
        in_progress = sum(1 for r in records if r.get("success") is None)

        durations = [r["duration_ms"] for r in records if r.get("duration_ms") is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total": total,
            "successes": successes,
            "failures": failures,
            "in_progress": in_progress,
            "success_rate": (successes / total * 100) if total > 0 else 0.0,
            "avg_duration_ms": avg_duration,
        }

    def purge_old(self, task_id: str, max_records: int = 100) -> None:
        """Delete oldest records exceeding max_records for a task."""
        history_keys = self._backend.get(f"hist:{task_id}") or []
        if len(history_keys) > max_records:
            to_delete = history_keys[:-max_records]
            for eid in to_delete:
                self._backend.delete(f"exec:{eid}")
            self._backend.set(f"hist:{task_id}", history_keys[-max_records:])

    def _append_to_history(self, task_id: str, execution_id: int) -> None:
        key = f"hist:{task_id}"
        history = self._backend.get(key) or []
        history.append(execution_id)
        if len(history) > self._max_records_per_task:
            history = history[-self._max_records_per_task:]
        self._backend.set(key, history)

    def close(self) -> None:
        pass
