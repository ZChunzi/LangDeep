"""ProcessManager — create, suspend, resume, and terminate long-running processes."""
import json
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..cache import BaseCacheBackend, MemoryCache
from ..logging import get_logger
from .models import Process, ProcessSignal, ProcessState

logger = get_logger(__name__)


class ProcessManager:
    """Manages the lifecycle of long-running workflow processes.

    Processes can be suspended (with snapshot), resumed, terminated,
    and listed.  State is persisted via a pluggable ``BaseCacheBackend``
    (defaults to ``MemoryCache`` — swap for Redis in production).

    Thread-safe: all mutations are guarded by a ``threading.Lock``.
    """

    def __init__(self, backend: Optional[BaseCacheBackend] = None):
        self._backend: BaseCacheBackend = backend or MemoryCache()
        self._processes: Dict[str, Process] = {}
        self._lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def create(self, workflow_name: str, initial_snapshot: Optional[Dict[str, Any]] = None) -> Process:
        """Create a new process in ACTIVE state.

        Args:
            workflow_name: Name of the workflow template or plan.
            initial_snapshot: Optional initial state dict.

        Returns:
            The newly created Process.
        """
        pid = uuid.uuid4().hex
        process = Process(
            pid=pid,
            workflow_name=workflow_name,
            snapshot=initial_snapshot or {},
        )
        with self._lock:
            self._processes[pid] = process
            self._persist(process)
        logger.info("Process created", extra={"pid": pid, "workflow": workflow_name})
        return process

    def suspend(self, pid: str, snapshot: Optional[Dict[str, Any]] = None) -> Optional[Process]:
        """Suspend a process, persisting its state snapshot.

        Returns None if the pid does not exist or the process is already
        TERMINATED (idempotent for already-suspended processes).
        """
        proc = self._get(pid)
        if proc is None or proc.state == ProcessState.TERMINATED:
            return None
        proc.state = ProcessState.SUSPENDED
        if snapshot is not None:
            proc.snapshot = snapshot
        with self._lock:
            self._persist(proc)
        logger.info("Process suspended", extra={"pid": pid})
        return proc

    def resume(self, pid: str, context: Optional[Dict[str, Any]] = None) -> Optional[Process]:
        """Resume a suspended process.

        If ``context`` is provided it is merged into the process snapshot.
        Returns None if the pid does not exist.
        """
        proc = self._get(pid)
        if proc is None:
            return None
        if proc.state in (ProcessState.ACTIVE, ProcessState.TERMINATED):
            return proc  # idempotent
        proc.state = ProcessState.ACTIVE
        proc.last_resumed_at = datetime.now()
        if context:
            proc.snapshot.update(context)
        with self._lock:
            self._persist(proc)
        logger.info("Process resumed", extra={"pid": pid})
        return proc

    def terminate(self, pid: str) -> Optional[Process]:
        """Terminate a process (idempotent).

        Returns None if the pid does not exist.
        """
        proc = self._get(pid)
        if proc is None:
            return None
        proc.state = ProcessState.TERMINATED
        with self._lock:
            self._persist(proc)
        logger.info("Process terminated", extra={"pid": pid})
        return proc

    def await_human(self, pid: str, snapshot: Optional[Dict[str, Any]] = None) -> Optional[Process]:
        """Mark a process as awaiting human input.

        This is used by workflow runtimes when task execution reaches a
        confirmation gate. Returns None if the process does not exist or is
        already terminated.
        """
        proc = self._get(pid)
        if proc is None or proc.state == ProcessState.TERMINATED:
            return None
        proc.state = ProcessState.AWAITING_HUMAN
        if snapshot is not None:
            proc.snapshot = snapshot
        with self._lock:
            self._persist(proc)
        logger.info("Process awaiting human input", extra={"pid": pid})
        return proc

    def update_snapshot(self, pid: str, snapshot: Dict[str, Any]) -> Optional[Process]:
        """Update a process snapshot without changing its lifecycle state.

        Returns None if the process does not exist. Terminated processes are
        returned unchanged.
        """
        proc = self._get(pid)
        if proc is None:
            return None
        if proc.state == ProcessState.TERMINATED:
            return proc
        proc.snapshot = snapshot
        with self._lock:
            self._persist(proc)
        logger.debug("Process snapshot updated", extra={"pid": pid})
        return proc

    # ── Signals ────────────────────────────────────────────────────────

    def send_signal(self, pid: str, signal: ProcessSignal) -> Optional[Process]:
        """Send a signal to a process.

        The signal is appended to the process's signal queue for later
        consumption by the workflow runtime.
        """
        proc = self._get(pid)
        if proc is None:
            return None
        proc.signals.append(signal)
        with self._lock:
            self._persist(proc)
        logger.info("Signal sent", extra={"pid": pid, "signal": signal.value})
        return proc

    # ── Query ──────────────────────────────────────────────────────────

    def get_process(self, pid: str) -> Optional[Process]:
        """Retrieve a process by PID (reads from cache, falls back to backend)."""
        return self._get(pid)

    def list_processes(self, state_filter: Optional[ProcessState] = None) -> List[Process]:
        """List all processes, optionally filtered by state."""
        with self._lock:
            processes = list(self._processes.values())
        if state_filter is not None:
            return [p for p in processes if p.state == state_filter]
        return processes

    # ── Internal helpers ───────────────────────────────────────────────

    def _get(self, pid: str) -> Optional[Process]:
        with self._lock:
            proc = self._processes.get(pid)
            if proc is not None:
                return proc
        # fallback to backend persistence
        return self._load(pid)

    def _persist(self, process: Process) -> None:
        try:
            data = {
                "pid": process.pid,
                "workflow_name": process.workflow_name,
                "state": process.state.value,
                "snapshot": process.snapshot,
                "checkpoints": process.checkpoints,
                "created_at": process.created_at.isoformat(),
                "last_resumed_at": process.last_resumed_at.isoformat() if process.last_resumed_at else None,
                "signals": [s.value for s in process.signals],
                "error_count": process.error_count,
            }
            self._backend.set(f"process:{process.pid}", json.dumps(data, default=str))
        except Exception:
            logger.warning("Failed to persist process", extra={"pid": process.pid})

    def _load(self, pid: str) -> Optional[Process]:
        try:
            raw = self._backend.get(f"process:{pid}")
            if raw is None:
                return None
            data = json.loads(raw) if isinstance(raw, str) else raw
            proc = Process(
                pid=data["pid"],
                workflow_name=data["workflow_name"],
                state=ProcessState(data["state"]),
                snapshot=data.get("snapshot", {}),
                checkpoints=data.get("checkpoints", []),
                created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
                last_resumed_at=datetime.fromisoformat(data["last_resumed_at"]) if data.get("last_resumed_at") else None,
                signals=[ProcessSignal(s) for s in data.get("signals", [])],
                error_count=data.get("error_count", 0),
            )
            with self._lock:
                self._processes[pid] = proc
            return proc
        except Exception:
            logger.warning("Failed to load persisted process", extra={"pid": pid})
            return None
