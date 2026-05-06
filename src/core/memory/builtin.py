"""In-memory implementation of BaseMemoryBackend.

Suitable for development and testing. Data is lost on process exit.
"""

import threading
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.messages import BaseMessage

from .base import BaseMemoryBackend, MemoryEntry, deserialize_message


class InMemoryBackend(BaseMemoryBackend):
    """Thread-safe in-memory memory backend.

    Stores entries in a Dict[session_id, List[MemoryEntry]].
    """

    def __init__(self):
        self._data: Dict[str, List[MemoryEntry]] = {}
        self._lock = threading.Lock()

    def store_entry(self, session_id: str, entry: MemoryEntry) -> None:
        with self._lock:
            if session_id not in self._data:
                self._data[session_id] = []
            # Replace or append by message_idx
            for i, existing in enumerate(self._data[session_id]):
                if existing.message_idx == entry.message_idx:
                    self._data[session_id][i] = entry
                    return
            self._data[session_id].append(entry)

    def store_messages(
        self, session_id: str, messages: "Sequence[BaseMessage]"
    ) -> int:
        # Import here for type hint
        from typing import Sequence
        from .base import serialize_message

        entries = [
            serialize_message(msg, session_id, idx)
            for idx, msg in enumerate(messages)
        ]
        with self._lock:
            if session_id not in self._data:
                self._data[session_id] = []
            existing_count = len(self._data[session_id])
            for entry in entries:
                entry.message_idx = existing_count + entry.message_idx
                self._data[session_id].append(entry)
        return len(entries)

    def load_messages(self, session_id: str) -> List[BaseMessage]:
        with self._lock:
            entries = list(self._data.get(session_id, []))
        entries.sort(key=lambda e: e.message_idx)
        return [deserialize_message(e) for e in entries]

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._data:
                del self._data[session_id]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def close(self) -> None:
        with self._lock:
            self._data.clear()

    def get_entry_count(self, session_id: Optional[str] = None) -> int:
        """Count entries, optionally filtered by session."""
        with self._lock:
            if session_id:
                return len(self._data.get(session_id, []))
            return sum(len(v) for v in self._data.values())
