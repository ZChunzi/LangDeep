"""Abstract base class and data models for memory backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)


@dataclass
class MemoryEntry:
    """A single message entry stored in a memory backend.

    Stores serialized LangChain message data in a storage-agnostic format.
    """
    session_id: str
    message_idx: int
    role: str  # human, ai, system, tool
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: Optional[str] = None       # JSON-encoded list
    additional_kwargs: Optional[str] = None  # JSON-encoded dict


def serialize_message(msg: BaseMessage, session_id: str, idx: int) -> MemoryEntry:
    """Convert a LangChain BaseMessage to a MemoryEntry."""
    import json

    tool_calls = None
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = json.dumps(msg.tool_calls, default=str)
    extra = None
    if msg.additional_kwargs:
        extra = json.dumps(msg.additional_kwargs, default=str)

    return MemoryEntry(
        session_id=session_id,
        message_idx=idx,
        role=msg.type,
        content=msg.content if isinstance(msg.content, str) else str(msg.content),
        timestamp=datetime.now(),
        tool_calls=tool_calls,
        additional_kwargs=extra,
    )


def deserialize_message(entry: MemoryEntry) -> BaseMessage:
    """Reconstruct a LangChain BaseMessage from a MemoryEntry."""
    import json

    kwargs: Dict[str, Any] = {"content": entry.content}

    if entry.tool_calls:
        try:
            kwargs["tool_calls"] = json.loads(entry.tool_calls)
        except (json.JSONDecodeError, TypeError):
            pass
    if entry.additional_kwargs:
        try:
            kwargs["additional_kwargs"] = json.loads(entry.additional_kwargs)
        except (json.JSONDecodeError, TypeError):
            pass

    role_map = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
        "tool": ToolMessage,
    }
    cls = role_map.get(entry.role, HumanMessage)
    # ToolMessage requires tool_call_id
    if cls is ToolMessage and "tool_call_id" not in kwargs:
        kwargs["tool_call_id"] = "unknown"
    return cls(**kwargs)


class BaseMemoryBackend(ABC):
    """Abstract interface for memory/persistence backends.

    All memory backends (in-memory, Redis, SQLite, PostgreSQL, etc.)
    must implement these methods.
    """

    @abstractmethod
    def store_entry(self, session_id: str, entry: MemoryEntry) -> None:
        """Persist a single MemoryEntry."""
        ...

    def store_messages(
        self, session_id: str, messages: Sequence[BaseMessage]
    ) -> int:
        """Convert and persist a sequence of BaseMessages.

        Returns the number of entries stored.
        """
        entries = [
            serialize_message(msg, session_id, idx)
            for idx, msg in enumerate(messages)
        ]
        for entry in entries:
            self.store_entry(session_id, entry)
        return len(entries)

    @abstractmethod
    def load_messages(self, session_id: str) -> List[BaseMessage]:
        """Load all messages for a session, ordered by message_idx.

        Returns reconstructed BaseMessage instances.
        """
        ...

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its entries. Returns True if deleted."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Wipe all data from this backend."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources (connections, file handles)."""
        ...
