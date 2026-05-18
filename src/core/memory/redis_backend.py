"""Redis-backed implementation of the LangDeep memory backend contract."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage

from .base import BaseMemoryBackend, MemoryEntry, deserialize_message, serialize_message


class RedisMemoryBackend(BaseMemoryBackend):
    """Persist conversation memory entries in Redis.

    The ``redis`` package is optional. Pass a compatible client in tests or
    install the ``redis`` extra before constructing this backend without one.
    """

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        client: Optional[Any] = None,
        key_prefix: str = "langdeep:memory",
        ttl_seconds: Optional[int] = None,
    ):
        self._client = client or self._connect(redis_url)
        self._key_prefix = key_prefix.rstrip(":")
        self._ttl_seconds = ttl_seconds

    def store_entry(self, session_id: str, entry: MemoryEntry) -> None:
        key = self._session_key(session_id)
        entries = self._load_entries(session_id)
        for idx, existing in enumerate(entries):
            if existing.message_idx == entry.message_idx:
                self._client.lset(key, idx, self._encode_entry(entry))
                self._remember_session(session_id)
                self._refresh_ttl(key)
                return
        self._client.rpush(key, self._encode_entry(entry))
        self._remember_session(session_id)
        self._refresh_ttl(key)

    def store_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> int:
        key = self._session_key(session_id)
        start_idx = int(self._client.llen(key))
        payloads = [
            self._encode_entry(serialize_message(message, session_id, start_idx + idx))
            for idx, message in enumerate(messages)
        ]
        if payloads:
            self._client.rpush(key, *payloads)
            self._remember_session(session_id)
            self._refresh_ttl(key)
        return len(payloads)

    def load_messages(self, session_id: str) -> List[BaseMessage]:
        entries = self._load_entries(session_id)
        entries.sort(key=lambda entry: entry.message_idx)
        return [deserialize_message(entry) for entry in entries]

    def list_sessions(self) -> List[str]:
        values = self._client.smembers(self._sessions_key())
        return sorted(self._decode_value(value) for value in values)

    def delete_session(self, session_id: str) -> bool:
        deleted = bool(self._client.delete(self._session_key(session_id)))
        self._client.srem(self._sessions_key(), session_id)
        return deleted

    def clear(self) -> None:
        sessions = self.list_sessions()
        keys = [self._session_key(session_id) for session_id in sessions]
        if keys:
            self._client.delete(*keys)
        self._client.delete(self._sessions_key())

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def get_entry_count(self, session_id: Optional[str] = None) -> int:
        """Count entries, optionally filtered by session."""
        if session_id is not None:
            return int(self._client.llen(self._session_key(session_id)))
        return sum(int(self._client.llen(self._session_key(name))) for name in self.list_sessions())

    @staticmethod
    def _connect(redis_url: str) -> Any:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "RedisMemoryBackend requires the optional 'redis' dependency. "
                "Install langdeep[redis] or pass a compatible Redis client."
            ) from exc
        return redis.Redis.from_url(redis_url)

    def _remember_session(self, session_id: str) -> None:
        self._client.sadd(self._sessions_key(), session_id)
        self._refresh_ttl(self._sessions_key())

    def _refresh_ttl(self, key: str) -> None:
        if self._ttl_seconds is not None:
            self._client.expire(key, self._ttl_seconds)

    def _load_entries(self, session_id: str) -> List[MemoryEntry]:
        values = self._client.lrange(self._session_key(session_id), 0, -1)
        return [self._decode_entry(value) for value in values]

    def _session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}:session:{session_id}"

    def _sessions_key(self) -> str:
        return f"{self._key_prefix}:sessions"

    @staticmethod
    def _encode_entry(entry: MemoryEntry) -> str:
        payload = {
            "session_id": entry.session_id,
            "message_idx": entry.message_idx,
            "role": entry.role,
            "content": entry.content,
            "timestamp": entry.timestamp.isoformat(),
            "tool_calls": entry.tool_calls,
            "additional_kwargs": entry.additional_kwargs,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _decode_entry(cls, value: Any) -> MemoryEntry:
        payload: Dict[str, Any] = json.loads(cls._decode_value(value))
        return MemoryEntry(
            session_id=payload["session_id"],
            message_idx=int(payload["message_idx"]),
            role=payload["role"],
            content=payload["content"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            tool_calls=payload.get("tool_calls"),
            additional_kwargs=payload.get("additional_kwargs"),
        )

    @staticmethod
    def _decode_value(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
