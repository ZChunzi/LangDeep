"""Abstract base class for cache backends."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCacheBackend(ABC):
    """Abstract interface for cache backends.

    Supports TTL-based expiration and key-value storage.
    Implementations: in-memory, Redis, Memcached, etc.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key. Returns None if missing or expired."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        ...

    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""
        ...
