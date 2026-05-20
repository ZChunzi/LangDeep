"""Memory/persistence module with abstract backend interface and decorator support.

Provides:
- BaseMemoryBackend — abstract interface for all storage backends
- InMemoryBackend — built-in in-memory implementation for dev/testing
- SQLiteMemoryBackend — built-in SQLite implementation for local persistence
- MemoryRegistry — singleton registry for backend factories
- @memory — decorator for registering backends
"""

from .base import BaseMemoryBackend, MemoryEntry, serialize_message, deserialize_message
from .builtin import InMemoryBackend
from .sqlite_backend import SQLiteMemoryBackend
from .registry import memory_registry, MemoryRegistry
from .decorators import memory

__all__ = [
    "BaseMemoryBackend",
    "MemoryEntry",
    "serialize_message",
    "deserialize_message",
    "InMemoryBackend",
    "SQLiteMemoryBackend",
    "MemoryRegistry",
    "memory_registry",
    "memory",
]
