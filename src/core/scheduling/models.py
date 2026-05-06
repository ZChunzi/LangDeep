"""Data models for the scheduling module."""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TriggerType(Enum):
    CRON = "cron"
    INTERVAL = "interval"
    CONDITION = "condition"
    EVENT = "event"
    ONCE = "once"


@dataclass
class ScheduledTask:
    """A scheduled workflow task."""
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
    """Context passed to condition checker functions."""
    variables: Dict[str, Any]
    last_result: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
