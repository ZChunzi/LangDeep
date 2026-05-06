"""Data models for observability."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class HealthStatus:
    """Aggregated health-check result.

    Attributes:
        status: Overall status — ``"healthy"``, ``"degraded"``, or ``"unhealthy"``.
        checks: Per-component check results.
        version: Application version string.
        timestamp: When the check was performed.
    """
    status: str = "healthy"
    checks: Dict[str, Any] = field(default_factory=dict)
    version: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
