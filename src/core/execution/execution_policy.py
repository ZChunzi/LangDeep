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
    """

    max_concurrency: int = 5
    strategy: str = "gather"
    retry_on: List[str] = field(default_factory=list)

    VALID_STRATEGIES = {"gather", "sequential", "priority_queue"}

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
        logger.debug(
            "Execution policy created",
            extra={"strategy": self.strategy, "max_concurrency": self.max_concurrency},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPolicy":
        return cls(**{k: v for k, v in data.items() if k in ("max_concurrency", "strategy", "retry_on")})

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
        }
