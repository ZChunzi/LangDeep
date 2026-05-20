"""SQLite-backed implementation of the LangDeep memory backend contract."""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union

from langchain_core.messages import BaseMessage

from .base import BaseMemoryBackend, MemoryEntry, deserialize_message, serialize_message


class SQLiteMemoryBackend(BaseMemoryBackend):
    """Persist conversation memory entries in a local SQLite database."""

    def __init__(
        self,
        database_path: Union[str, Path] = "langdeep_memory.sqlite3",
        *,
        connection: Optional[sqlite3.Connection] = None,
    ):
        self._database_path = str(database_path)
        self._owns_connection = connection is None
        self._connection = connection or sqlite3.connect(self._database_path, check_same_thread=False)
        self._lock = threading.RLock()
        self._initialize()

    def store_entry(self, session_id: str, entry: MemoryEntry) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO memory_entries (
                    session_id, message_idx, role, content, timestamp,
                    tool_calls, additional_kwargs
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                self._entry_values(session_id, entry),
            )
            self._connection.commit()

    def store_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> int:
        with self._lock:
            start_idx = self._next_message_idx(session_id)
            entries = [
                serialize_message(message, session_id, start_idx + idx)
                for idx, message in enumerate(messages)
            ]
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO memory_entries (
                    session_id, message_idx, role, content, timestamp,
                    tool_calls, additional_kwargs
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [self._entry_values(session_id, entry) for entry in entries],
            )
            self._connection.commit()
        return len(entries)

    def load_messages(self, session_id: str) -> List[BaseMessage]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT session_id, message_idx, role, content, timestamp,
                       tool_calls, additional_kwargs
                FROM memory_entries
                WHERE session_id = ?
                ORDER BY message_idx ASC
                """,
                (session_id,),
            ).fetchall()
        return [deserialize_message(self._entry_from_row(row)) for row in rows]

    def list_sessions(self) -> List[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT DISTINCT session_id FROM memory_entries ORDER BY session_id ASC"
            ).fetchall()
        return [row[0] for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM memory_entries WHERE session_id = ?",
                (session_id,),
            )
            self._connection.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM memory_entries")
            self._connection.commit()

    def close(self) -> None:
        if self._owns_connection:
            with self._lock:
                self._connection.close()

    def get_entry_count(self, session_id: Optional[str] = None) -> int:
        """Count entries, optionally filtered by session."""
        with self._lock:
            if session_id is None:
                row = self._connection.execute("SELECT COUNT(*) FROM memory_entries").fetchone()
            else:
                row = self._connection.execute(
                    "SELECT COUNT(*) FROM memory_entries WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        return int(row[0])

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    session_id TEXT NOT NULL,
                    message_idx INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tool_calls TEXT,
                    additional_kwargs TEXT,
                    PRIMARY KEY (session_id, message_idx)
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_session
                ON memory_entries (session_id, message_idx)
                """
            )
            self._connection.commit()

    def _next_message_idx(self, session_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(message_idx), -1) + 1 FROM memory_entries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0])

    @staticmethod
    def _entry_values(session_id: str, entry: MemoryEntry):
        return (
            session_id,
            entry.message_idx,
            entry.role,
            entry.content,
            entry.timestamp.isoformat(),
            entry.tool_calls,
            entry.additional_kwargs,
        )

    @staticmethod
    def _entry_from_row(row) -> MemoryEntry:
        return MemoryEntry(
            session_id=row[0],
            message_idx=int(row[1]),
            role=row[2],
            content=row[3],
            timestamp=datetime.fromisoformat(row[4]),
            tool_calls=row[5],
            additional_kwargs=row[6],
        )
