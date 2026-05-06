"""Unit tests for the process module: Process, ProcessManager, data models."""

from datetime import datetime

from langdeep.core.process import (
    Process,
    ProcessManager,
    ProcessState,
    ProcessSignal,
    SuspendSignal,
)
from langdeep.core.errors import LangDeepError


def setup_function():
    """Reset singleton registries before each test."""
    from tests.conftest import clean_registries
    clean_registries()


# ── Models ──────────────────────────────────────────────────────────────────


def test_process_state_values():
    """ProcessState enum has expected values."""
    assert ProcessState.ACTIVE.value == "active"
    assert ProcessState.SUSPENDED.value == "suspended"
    assert ProcessState.TERMINATED.value == "terminated"
    assert ProcessState.AWAITING_HUMAN.value == "awaiting_human"


def test_process_signal_values():
    """ProcessSignal enum has expected values."""
    assert ProcessSignal.TERMINATE.value == "terminate"
    assert ProcessSignal.PAUSE.value == "pause"
    assert ProcessSignal.RESUME.value == "resume"
    assert ProcessSignal.INJECT_CONTEXT.value == "inject_context"


def test_process_defaults():
    """Process dataclass has correct default values."""
    p = Process(pid="p1", workflow_name="test")
    assert p.state == ProcessState.ACTIVE
    assert p.snapshot == {}
    assert p.checkpoints == []
    assert p.last_resumed_at is None
    assert p.signals == []
    assert p.error_count == 0
    assert isinstance(p.created_at, datetime)


def test_suspend_signal_inheritance():
    """SuspendSignal is a LangDeepError with the correct code."""
    exc = SuspendSignal(detail="pausing")
    assert isinstance(exc, LangDeepError)
    assert exc.code == "SUSPEND_SIGNAL"


# ── ProcessManager: create ──────────────────────────────────────────────────


def test_manager_create():
    """Creating a process returns a process with ACTIVE state and a pid."""
    pm = ProcessManager()
    proc = pm.create("test_workflow")
    assert proc.pid is not None
    assert proc.workflow_name == "test_workflow"
    assert proc.state == ProcessState.ACTIVE


def test_manager_create_with_snapshot():
    """Creating a process with an initial snapshot stores the snapshot."""
    pm = ProcessManager()
    proc = pm.create("test", initial_snapshot={"step": 1, "data": "hello"})
    assert proc.snapshot == {"step": 1, "data": "hello"}


# ── ProcessManager: suspend / resume / terminate ───────────────────────────


def test_manager_suspend():
    """Suspending a process sets its state to SUSPENDED."""
    pm = ProcessManager()
    proc = pm.create("test")
    result = pm.suspend(proc.pid)
    assert result is not None
    assert result.state == ProcessState.SUSPENDED


def test_manager_suspend_with_snapshot():
    """Suspending with a snapshot updates the process snapshot."""
    pm = ProcessManager()
    proc = pm.create("test")
    pm.suspend(proc.pid, snapshot={"progress": "half"})
    assert pm.get_process(proc.pid).snapshot == {"progress": "half"}


def test_manager_resume():
    """Resuming a suspended process sets it back to ACTIVE."""
    pm = ProcessManager()
    proc = pm.create("test")
    pm.suspend(proc.pid)
    result = pm.resume(proc.pid)
    assert result is not None
    assert result.state == ProcessState.ACTIVE
    assert result.last_resumed_at is not None


def test_manager_resume_with_context():
    """Resuming with context merges the context into the snapshot."""
    pm = ProcessManager()
    proc = pm.create("test", initial_snapshot={"a": 1})
    pm.suspend(proc.pid)
    pm.resume(proc.pid, context={"b": 2})
    assert pm.get_process(proc.pid).snapshot == {"a": 1, "b": 2}


def test_manager_terminate():
    """Terminating a process sets its state to TERMINATED."""
    pm = ProcessManager()
    proc = pm.create("test")
    result = pm.terminate(proc.pid)
    assert result is not None
    assert result.state == ProcessState.TERMINATED


def test_manager_suspend_already_terminated():
    """Suspending a terminated process returns None (idempotent)."""
    pm = ProcessManager()
    proc = pm.create("test")
    pm.terminate(proc.pid)
    assert pm.suspend(proc.pid) is None


def test_manager_resume_active_is_idempotent():
    """Resuming an already-active process returns it unchanged."""
    pm = ProcessManager()
    proc = pm.create("test")
    result = pm.resume(proc.pid)
    assert result is proc


# ── ProcessManager: signals ─────────────────────────────────────────────────


def test_manager_send_signal():
    """Sending a signal appends it to the process signal queue."""
    pm = ProcessManager()
    proc = pm.create("test")
    pm.send_signal(proc.pid, ProcessSignal.PAUSE)
    assert ProcessSignal.PAUSE in pm.get_process(proc.pid).signals


# ── ProcessManager: query ───────────────────────────────────────────────────


def test_manager_get_nonexistent():
    """Getting a non-existent process returns None."""
    pm = ProcessManager()
    assert pm.get_process("nonexistent") is None


def test_manager_list_all():
    """list_processes returns all created processes."""
    pm = ProcessManager()
    p1 = pm.create("w1")
    p2 = pm.create("w2")
    all_procs = pm.list_processes()
    assert len(all_procs) == 2
    assert p1.pid in (x.pid for x in all_procs)
    assert p2.pid in (x.pid for x in all_procs)


def test_manager_list_filtered():
    """list_processes with state_filter returns only matching processes."""
    pm = ProcessManager()
    p1 = pm.create("w1")
    pm.create("w2")
    pm.terminate(p1.pid)
    active = pm.list_processes(state_filter=ProcessState.ACTIVE)
    terminated = pm.list_processes(state_filter=ProcessState.TERMINATED)
    assert len(active) == 1
    assert len(terminated) == 1


# ── ProcessManager: error handling ─────────────────────────────────────────


def test_manager_suspend_nonexistent():
    """Suspending a non-existent pid returns None."""
    pm = ProcessManager()
    assert pm.suspend("ghost") is None


def test_manager_terminate_nonexistent():
    """Terminating a non-existent pid returns None."""
    pm = ProcessManager()
    assert pm.terminate("ghost") is None


def test_manager_send_signal_nonexistent():
    """Sending a signal to a non-existent pid returns None."""
    pm = ProcessManager()
    assert pm.send_signal("ghost", ProcessSignal.TERMINATE) is None
