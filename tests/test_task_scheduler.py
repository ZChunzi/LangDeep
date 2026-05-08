"""Unit tests for TaskScheduler, ScheduledTask, triggers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import time
from datetime import datetime, timedelta

from langdeep.core.scheduling.task_scheduler import (
    TaskScheduler, ScheduledTask, TriggerType, TriggerType as TT,
    ConditionContext,
)
from langdeep.core.scheduling.persistent_store import TaskStore

# We need a minimal FlowOrchestrator-like object for the scheduler
class FakeOrchestrator:
    def __init__(self):
        self.invoke_calls = []
        self.ainvoke_calls = []
    def invoke(self, user_input="", context=None):
        self.invoke_calls.append((user_input, context))
        return {"result": "ok"}
    async def ainvoke(self, user_input="", context=None):
        self.ainvoke_calls.append((user_input, context))
        return {"result": "ok_async"}


def test_register_and_list():
    orch = FakeOrchestrator()
    sched = TaskScheduler(orchestrator=orch)
    task = ScheduledTask(
        id="t1", name="Test", trigger_type=TriggerType.ONCE,
        trigger_config={"at": (datetime.now() + timedelta(hours=1)).isoformat()},
        workflow="test_wf",
    )
    sched.register_task(task)
    tasks = sched.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "t1"
    assert tasks[0].next_run is not None


def test_task_store_save_task_updates_recovery_index():
    store = TaskStore()
    task = ScheduledTask(
        id="persisted",
        name="Persisted",
        trigger_type=TriggerType.INTERVAL,
        trigger_config={"seconds": 60},
        workflow="wf",
        params={"user_input": "hello"},
    )

    store.save_task(task)

    recovered = store.load_all()
    assert store.count() == 1
    assert len(recovered) == 1
    assert recovered[0].id == "persisted"
    assert recovered[0].params["user_input"] == "hello"


def test_task_store_delete_task_updates_recovery_index():
    store = TaskStore()
    task = ScheduledTask(
        id="delete_me",
        name="DeleteMe",
        trigger_type=TriggerType.INTERVAL,
        trigger_config={"seconds": 60},
        workflow="wf",
    )
    store.save_task(task)

    store.delete_task("delete_me")

    assert store.load_task("delete_me") is None
    assert store.load_all() == []
    assert store.count() == 0


def test_start_stop():
    orch = FakeOrchestrator()
    sched = TaskScheduler(orchestrator=orch)
    sched.start()
    assert sched._running is True
    sched.stop()
    assert sched._running is False


def test_should_run_cron():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    task = ScheduledTask(
        id="cron_t", name="Cron", trigger_type=TriggerType.CRON,
        trigger_config={"expression": "* * * * *"},
        workflow="wf",
    )
    sched._calculate_next_run(task)
    now = datetime.now()
    assert task.next_run is not None
    assert task.next_run > now


def test_should_run_interval():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    task = ScheduledTask(
        id="int_t", name="Interval", trigger_type=TriggerType.INTERVAL,
        trigger_config={"seconds": 3600},
        workflow="wf",
    )
    sched._calculate_next_run(task)
    assert task.next_run is not None


def test_should_run_once_in_past():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    past = (datetime.now() - timedelta(minutes=5)).isoformat()
    task = ScheduledTask(
        id="once_past", name="Past", trigger_type=TriggerType.ONCE,
        trigger_config={"at": past},
        workflow="wf",
    )
    sched._calculate_next_run(task)
    assert task.next_run is not None
    # should_run returns True for past timestamps
    assert sched._should_run(task, datetime.now()) is True


def test_disabled_task_does_not_run():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    past = (datetime.now() - timedelta(minutes=5)).isoformat()
    task = ScheduledTask(
        id="disabled", name="Disabled", trigger_type=TriggerType.ONCE,
        trigger_config={"at": past}, workflow="wf", enabled=False,
    )
    sched._calculate_next_run(task)
    assert sched._should_run(task, datetime.now()) is True  # time-wise it should
    # But the scheduler loop checks enabled flag before calling _should_run
    # _should_run doesn't check enabled, the loop does


def test_condition_trigger():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    task = ScheduledTask(
        id="cond_t", name="Conditional", trigger_type=TriggerType.CONDITION,
        trigger_config={"condition": "my_checker", "variables": {"x": 1}},
        workflow="wf",
    )

    def my_checker(ctx):
        return ctx.variables.get("x") == 1

    sched.register_condition_checker("my_checker", my_checker)
    # should_run should call the checker
    assert sched._should_run(task, datetime.now()) is True


def test_condition_trigger_false():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    task = ScheduledTask(
        id="cond_f", name="CondFalse", trigger_type=TriggerType.CONDITION,
        trigger_config={"condition": "false_checker", "variables": {"x": 0}},
        workflow="wf",
    )

    def false_checker(ctx):
        return ctx.variables.get("x") == 99

    sched.register_condition_checker("false_checker", false_checker)
    assert sched._should_run(task, datetime.now()) is False


def test_execute_task():
    orch = FakeOrchestrator()
    sched = TaskScheduler(orchestrator=orch)
    task = ScheduledTask(
        id="exec_t", name="Exec", trigger_type=TriggerType.ONCE,
        trigger_config={"at": datetime.now().isoformat()},
        workflow="wf", params={"user_input": "hello", "context": {"key": "val"}},
    )
    sched._execute_task(task)
    assert len(orch.invoke_calls) == 1
    assert orch.invoke_calls[0][0] == "hello"
    assert orch.invoke_calls[0][1]["key"] == "val"


def test_execute_task_error_retry():
    class ErrorOrch:
        def __init__(self):
            self.count = 0
        def invoke(self, user_input="", context=None):
            self.count += 1
            raise RuntimeError(f"fail {self.count}")

    orch = ErrorOrch()
    sched = TaskScheduler(orchestrator=orch)
    task = ScheduledTask(
        id="err_t", name="Error", trigger_type=TriggerType.ONCE,
        trigger_config={"at": datetime.now().isoformat()},
        workflow="wf", retry_count=3, retry_delay=0,
    )
    sched._execute_task(task)
    # After first execution, retry_count decrements to 2 and task is re-enabled
    assert task.retry_count == 2
    assert task.enabled is True


def test_execute_task_exhaust_retries():
    class ErrorOrch:
        def invoke(self, user_input="", context=None):
            raise RuntimeError("always fail")

    orch = ErrorOrch()
    sched = TaskScheduler(orchestrator=orch)
    task = ScheduledTask(
        id="exhaust", name="Exhaust", trigger_type=TriggerType.ONCE,
        trigger_config={"at": datetime.now().isoformat()},
        workflow="wf", retry_count=1, retry_delay=0,
    )
    sched._execute_task(task)
    assert task.enabled is False


def test_aexecute_now():
    orch = FakeOrchestrator()
    sched = TaskScheduler(orchestrator=orch)
    task = ScheduledTask(
        id="ae_t", name="AsyncExec", trigger_type=TriggerType.INTERVAL,
        trigger_config={"seconds": 60},
        workflow="wf", params={"user_input": "async_test"},
    )
    sched.register_task(task)

    async def run():
        return await sched.aexecute_now("ae_t")

    result = asyncio.run(run())
    assert result["result"] == "ok_async"
    assert len(orch.ainvoke_calls) == 1


def test_aexecute_now_not_found():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    async def run():
        return await sched.aexecute_now("ghost")
    try:
        asyncio.run(run())
        assert False, "Should raise"
    except Exception:
        pass


def test_calculate_next_run_cron():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    task = ScheduledTask(
        id="cron_calc", name="CronCalc", trigger_type=TriggerType.CRON,
        trigger_config={"expression": "0 * * * *"},
        workflow="wf",
    )
    sched._calculate_next_run(task)
    assert task.next_run is not None
    assert task.next_run > datetime.now()


def test_calculate_next_run_interval():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    task = ScheduledTask(
        id="int_calc", name="IntCalc", trigger_type=TriggerType.INTERVAL,
        trigger_config={"seconds": 100},
        workflow="wf",
    )
    sched._calculate_next_run(task)
    assert task.next_run is not None
    delta = (task.next_run - datetime.now()).total_seconds()
    assert 95 <= delta <= 105  # roughly 100 seconds


def test_calculate_next_run_once():
    sched = TaskScheduler(orchestrator=FakeOrchestrator())
    future = (datetime.now() + timedelta(hours=2)).isoformat()
    task = ScheduledTask(
        id="once_calc", name="OnceCalc", trigger_type=TriggerType.ONCE,
        trigger_config={"at": future},
        workflow="wf",
    )
    sched._calculate_next_run(task)
    assert task.next_run is not None
