"""In-memory cache implementation with LRU eviction and TTL support."""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple

from .base import BaseCacheBackend


class MemoryCache(BaseCacheBackend):
    """Thread-safe in-memory cache with LRU eviction and optional TTL.

    Parameters:
        max_size: Maximum number of entries before LRU eviction (default 1024).
        default_ttl: Default TTL in seconds (None = no TTL).
    """

    def __init__(self, max_size: int = 1024, default_ttl: Optional[int] = None):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._data: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        # (expiry_timestamp_or_inf, value)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            expiry, value = self._data[key]
            if expiry < time.time():
                del self._data[key]
                return None
            # Mark as recently used
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = (time.time() + effective_ttl) if effective_ttl is not None else float("inf")
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (expiry, value)
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries when over max_size."""
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
