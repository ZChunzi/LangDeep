"""Built-in cache implementations with TTL support."""

import hashlib
import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


class FileCacheBackend(BaseCacheBackend):
    """Local file-backed cache using pickle serialization.

    This backend is intended for trusted local response caches. It should not be
    pointed at shared or untrusted directories because cache values are
    serialized with ``pickle``.
    """

    _INDEX_FILE = "index.json"

    def __init__(
        self,
        path: str,
        max_entries: int = 1024,
        default_ttl: Optional[int] = None,
    ):
        self._path = Path(path).expanduser()
        if self._path.exists() and not self._path.is_dir():
            raise ValueError(f"File cache path must be a directory: {self._path}")
        self._path.mkdir(parents=True, exist_ok=True)
        self._index_path = self._path / self._INDEX_FILE
        self._max_entries = max(0, int(max_entries))
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._index: Dict[str, Dict[str, Any]] = self._load_index()
        with self._lock:
            self._cleanup_expired_locked()
            self._evict_if_needed_locked()
            self._save_index_locked()

    @property
    def path(self) -> str:
        """Return the cache directory path."""
        return str(self._path)

    def get(self, key: str) -> Optional[Any]:
        digest = self._digest(key)
        with self._lock:
            entry = self._index.get(digest)
            if entry is None:
                return None
            if self._is_expired(entry):
                self._delete_digest_locked(digest)
                self._save_index_locked()
                return None
            value_path = self._value_path(digest)
            if not value_path.exists():
                self._delete_digest_locked(digest)
                self._save_index_locked()
                return None
            try:
                with value_path.open("rb") as f:
                    payload = pickle.load(f)  # nosec B301 - trusted local cache only.
            except Exception:
                self._delete_digest_locked(digest)
                self._save_index_locked()
                return None
            if not isinstance(payload, dict) or payload.get("key") != key:
                self._delete_digest_locked(digest)
                self._save_index_locked()
                return None
            entry["accessed_at"] = time.time()
            self._save_index_locked()
            return payload.get("value")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        digest = self._digest(key)
        now = time.time()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None

        with self._lock:
            tmp_path = self._path / f".{digest}.tmp"
            with tmp_path.open("wb") as f:
                pickle.dump({"key": key, "value": value}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self._value_path(digest))
            self._index[digest] = {
                "key": key,
                "created_at": self._index.get(digest, {}).get("created_at", now),
                "accessed_at": now,
                "expires_at": expires_at,
            }
            self._cleanup_expired_locked()
            self._evict_if_needed_locked()
            self._save_index_locked()

    def delete(self, key: str) -> bool:
        digest = self._digest(key)
        with self._lock:
            existed = digest in self._index or self._value_path(digest).exists()
            self._delete_digest_locked(digest)
            self._save_index_locked()
            return existed

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            for path in self._path.glob("*.pickle"):
                try:
                    path.unlink()
                except OSError:
                    pass
            for path in self._path.glob(".*.tmp"):
                try:
                    path.unlink()
                except OSError:
                    pass
            self._index.clear()
            self._save_index_locked()

    def __len__(self) -> int:
        with self._lock:
            self._cleanup_expired_locked()
            self._save_index_locked()
            return len(self._index)

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if not self._index_path.exists():
            return {}
        try:
            with self._index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for digest, entry in data.items():
            if isinstance(digest, str) and isinstance(entry, dict):
                result[digest] = entry
        return result

    def _save_index_locked(self) -> None:
        tmp_path = self._path / f".{self._INDEX_FILE}.tmp"
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._index, f, sort_keys=True, separators=(",", ":"))
        os.replace(tmp_path, self._index_path)

    def _cleanup_expired_locked(self) -> None:
        for digest, entry in list(self._index.items()):
            if self._is_expired(entry):
                self._delete_digest_locked(digest)

    def _evict_if_needed_locked(self) -> None:
        while len(self._index) > self._max_entries:
            digest = min(
                self._index,
                key=lambda item: float(self._index[item].get("accessed_at") or 0),
            )
            self._delete_digest_locked(digest)

    def _delete_digest_locked(self, digest: str) -> None:
        self._index.pop(digest, None)
        try:
            self._value_path(digest).unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _digest(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _value_path(self, digest: str) -> Path:
        return self._path / f"{digest}.pickle"

    @staticmethod
    def _is_expired(entry: Dict[str, Any]) -> bool:
        expires_at = entry.get("expires_at")
        return expires_at is not None and float(expires_at) < time.time()
