"""Tests for scheduling support infrastructure."""

import json
import time
from datetime import datetime

from langdeep.core.cache import MemoryCache
from langdeep.core.scheduling.audit_log import AuditLog
from langdeep.core.scheduling.models import ScheduledTask, TriggerType
from langdeep.core.scheduling.persistent_store import TaskStore
from langdeep.core.scheduling.worker_pool import WorkerPool


def test_audit_log_records_completion_failure_stats_and_purge():
    log = AuditLog(max_records_per_task=2)

    first = log.log_start("task1", "Task One", trace_id="trace-1")
    log.log_complete(first, "x" * 600, 25)
    second = log.log_start("task1", "Task One")
    log.log_failure(second, "boom", 10)
    third = log.log_start("task1", "Task One")

    history = log.get_history("task1")
    assert [r["execution_id"] for r in history] == [second, third]
    assert history[0]["success"] is False
    assert history[1]["success"] is None

    stats = log.get_stats("task1")
    assert stats["total"] == 2
    assert stats["failures"] == 1
    assert stats["in_progress"] == 1

    all_recent = log.get_recent()
    assert all_recent[0]["started_at"] >= all_recent[-1]["started_at"]

    log.purge_old("task1", max_records=1)
    assert len(log.get_history("task1")) == 1
    log.close()


def test_task_store_handles_json_encoded_backend_values():
    backend = MemoryCache()
    store = TaskStore(backend=backend)
    task_data = {
        "id": "json-task",
        "name": "JSON Task",
        "trigger_type": "interval",
        "trigger_config": json.dumps({"seconds": 5}),
        "workflow": "wf",
        "params": json.dumps({"user_input": "hi"}),
        "enabled": True,
        "last_run": datetime.now().isoformat(),
        "next_run": "not-a-date",
        "timeout": 30,
        "retry_count": 2,
        "retry_delay": 1,
    }
    backend.set("task:json-task", json.dumps(task_data))
    backend.set("_scheduled_tasks", json.dumps(["json-task"]))

    loaded = store.load_task("json-task")

    assert loaded is not None
    assert loaded.id == "json-task"
    assert loaded.params["user_input"] == "hi"
    assert loaded.last_run is not None
    assert loaded.next_run is None
    assert store.load_all()[0].id == "json-task"
    assert store.count() == 1


def test_task_store_save_all_clear_and_invalid_index():
    backend = MemoryCache()
    store = TaskStore(backend=backend)
    backend.set("_scheduled_tasks", 123)
    assert store._load_task_ids() == []

    task = ScheduledTask(
        id="bulk",
        name="Bulk",
        trigger_type=TriggerType.ONCE,
        trigger_config={"at": datetime.now().isoformat()},
        workflow="wf",
    )
    store.save_all([task])
    assert store.count() == 1
    store.clear()
    assert store.count() == 0
    assert store.load_all() == []


def test_worker_pool_submit_completion_cancel_and_shutdown():
    pool = WorkerPool(max_workers=1, name_prefix="test-sched")
    future = pool.submit("t1", lambda: "done")
    assert future.result(timeout=5) == "done"

    for _ in range(20):
        if pool.running_count() == 0:
            break
        time.sleep(0.01)
    assert pool.running_count() == 0
    assert pool.cancel("missing") is False

    slow_future = pool.submit("slow", time.sleep, 0.05)
    pool.shutdown(wait=True, timeout=1)
    assert slow_future.done() or slow_future.running() is False
    assert pool.running_count() == 0
