"""Execution policy — concurrency and execution strategy control."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging import get_logger
from ..errors import InvalidPolicyError, ConfigurationError

logger = get_logger(__name__)


@dataclass
class ExecutionPolicy:
    """Controls how tasks are executed within a workflow.

    Attributes:
        max_concurrency: Maximum parallel tasks (default 5).
        strategy: One of "gather", "sequential", "priority_queue".
        retry_on: Optional list of error class names to trigger retry.
        max_retries: Maximum attempts for retry-capable task runners.
        retry_backoff: "exponential" or "fixed".
        timeout_seconds: Per-task timeout for async execution.
        fail_fast: Stop scheduling new batches after a task fails.
    """

    max_concurrency: int = 5
    strategy: str = "gather"
    retry_on: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_backoff: str = "exponential"
    timeout_seconds: float = 30.0
    fail_fast: bool = False

    VALID_STRATEGIES = {"gather", "sequential", "priority_queue"}
    VALID_BACKOFFS = {"exponential", "fixed"}

    def __post_init__(self):
        if self.strategy not in self.VALID_STRATEGIES:
            raise InvalidPolicyError(
                f"Invalid strategy '{self.strategy}'",
                context={"valid": sorted(self.VALID_STRATEGIES)},
            )
        if self.max_concurrency < 1:
            raise InvalidPolicyError(
                f"max_concurrency must be >= 1, got {self.max_concurrency}",
                context={"max_concurrency": self.max_concurrency},
            )
        if self.max_retries < 1:
            raise InvalidPolicyError(
                f"max_retries must be >= 1, got {self.max_retries}",
                context={"max_retries": self.max_retries},
            )
        if self.retry_backoff not in self.VALID_BACKOFFS:
            raise InvalidPolicyError(
                f"Invalid retry_backoff '{self.retry_backoff}'",
                context={"valid": sorted(self.VALID_BACKOFFS)},
            )
        if self.timeout_seconds <= 0:
            raise InvalidPolicyError(
                f"timeout_seconds must be > 0, got {self.timeout_seconds}",
                context={"timeout_seconds": self.timeout_seconds},
            )
        logger.debug(
            "Execution policy created",
            extra={"strategy": self.strategy, "max_concurrency": self.max_concurrency},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPolicy":
        allowed = {
            "max_concurrency",
            "strategy",
            "retry_on",
            "max_retries",
            "retry_backoff",
            "timeout_seconds",
            "fail_fast",
        }
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def from_file(cls, path: str) -> "ExecutionPolicy":
        import json
        import os
        if not os.path.isfile(path):
            raise ConfigurationError(
                f"Execution policy file not found: {path}",
                context={"path": path},
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        policy = cls.from_dict(data)
        logger.info("Execution policy loaded from file", extra={"path": path, "strategy": policy.strategy})
        return policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_concurrency": self.max_concurrency,
            "strategy": self.strategy,
            "retry_on": list(self.retry_on),
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff,
            "timeout_seconds": self.timeout_seconds,
            "fail_fast": self.fail_fast,
        }
