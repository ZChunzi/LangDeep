"""Data models and enums for the Process Manager."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..errors import LangDeepError


class ProcessState(Enum):
    """Lifecycle state of a managed process."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    AWAITING_HUMAN = "awaiting_human"
    TERMINATED = "terminated"


class ProcessSignal(Enum):
    """External signals that can be sent to a process."""
    TERMINATE = "terminate"
    PAUSE = "pause"
    RESUME = "resume"
    INJECT_CONTEXT = "inject_context"


@dataclass
class Process:
    """A long-running workflow process with suspend/resume support.

    The ``snapshot`` dict holds arbitrary workflow state that can be
    serialised and restored across process boundaries.
    """
    pid: str
    workflow_name: str
    state: ProcessState = ProcessState.ACTIVE
    snapshot: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_resumed_at: Optional[datetime] = None
    signals: List[ProcessSignal] = field(default_factory=list)
    error_count: int = 0


class SuspendSignal(LangDeepError):
    """Raised within a workflow node to signal voluntary suspension.

    The Process Manager catches this signal, persists the current state,
    and marks the process as SUSPENDED so it can be resumed later.
    """
    code = "SUSPEND_SIGNAL"
