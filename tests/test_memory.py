"""Unit tests for the memory module: BaseMemoryBackend, InMemoryBackend, MemoryRegistry, @memory."""

from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from langdeep.core.memory import (
    InMemoryBackend,
    RedisMemoryBackend,
    SQLiteMemoryBackend,
    MemoryEntry,
    BaseMemoryBackend,
    MemoryRegistry,
    memory_registry,
    serialize_message,
    deserialize_message,
    memory,
)


def setup_function():
    """Reset singleton registries before each test."""
    from tests.conftest import clean_registries
    clean_registries()


# ── InMemoryBackend ──────────────────────────────────────────────────────────


def test_in_memory_store_entry():
    """Store a single MemoryEntry and retrieve it."""
    backend = InMemoryBackend()
    entry = MemoryEntry("s1", 0, "human", "Hello!", datetime.now())
    backend.store_entry("s1", entry)
    msgs = backend.load_messages("s1")
    assert len(msgs) == 1
    assert msgs[0].content == "Hello!"
    assert msgs[0].type == "human"


def test_in_memory_store_messages():
    """Store multiple BaseMessages and retrieve them in order."""
    backend = InMemoryBackend()
    messages = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello!"),
        HumanMessage(content="How are you?"),
    ]
    count = backend.store_messages("s2", messages)
    assert count == 3
    loaded = backend.load_messages("s2")
    assert len(loaded) == 3
    assert loaded[0].content == "Hi"
    assert loaded[1].content == "Hello!"
    assert loaded[2].content == "How are you?"


def test_in_memory_append_messages():
    """Calling store_messages again appends, does not overwrite."""
    backend = InMemoryBackend()
    backend.store_messages("s3", [HumanMessage(content="first")])
    backend.store_messages("s3", [HumanMessage(content="second")])
    loaded = backend.load_messages("s3")
    assert len(loaded) == 2


def test_in_memory_delete_session():
    """Delete a session and verify it is gone."""
    backend = InMemoryBackend()
    backend.store_messages("s4", [HumanMessage(content="data")])
    assert backend.delete_session("s4") is True
    assert backend.load_messages("s4") == []
    assert "s4" not in backend.list_sessions()


def test_in_memory_delete_session_not_found():
    """Delete a non-existent session returns False."""
    backend = InMemoryBackend()
    assert backend.delete_session("ghost") is False


def test_in_memory_clear():
    """Clear wipes all sessions."""
    backend = InMemoryBackend()
    backend.store_messages("a", [HumanMessage(content="x")])
    backend.store_messages("b", [HumanMessage(content="y")])
    backend.clear()
    assert backend.list_sessions() == []
    assert backend.load_messages("a") == []


def test_in_memory_list_sessions():
    """List sessions returns all stored session IDs."""
    backend = InMemoryBackend()
    assert backend.list_sessions() == []
    backend.store_messages("x", [HumanMessage(content="1")])
    backend.store_messages("y", [HumanMessage(content="2")])
    sessions = backend.list_sessions()
    assert "x" in sessions
    assert "y" in sessions
    assert len(sessions) == 2


def test_in_memory_entry_count():
    """get_entry_count reports correct counts."""
    backend = InMemoryBackend()
    assert backend.get_entry_count() == 0
    backend.store_messages("s", [HumanMessage(content="a"), AIMessage(content="b")])
    assert backend.get_entry_count("s") == 2
    assert backend.get_entry_count() == 2


def test_in_memory_close_is_idempotent():
    """close() clears data and can be called multiple times."""
    backend = InMemoryBackend()
    backend.store_messages("s", [HumanMessage(content="x")])
    backend.close()
    assert backend.list_sessions() == []
    backend.close()  # second call should not raise


def test_in_memory_concurrent_store():
    """Store messages to different sessions concurrently (thread safety)."""
    import threading

    backend = InMemoryBackend()
    results = []

    def writer(session_id, count):
        for i in range(count):
            backend.store_messages(session_id, [HumanMessage(content=f"msg_{i}")])
        results.append(True)

    threads = [
        threading.Thread(target=writer, args=("t1", 50)),
        threading.Thread(target=writer, args=("t2", 50)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(backend.load_messages("t1")) == 50
    assert len(backend.load_messages("t2")) == 50


# ── RedisMemoryBackend ───────────────────────────────────────────────────────


class FakeRedis:
    def __init__(self):
        self.lists = {}
        self.sets = {}
        self.expirations = {}
        self.closed = False

    def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(values)
        return len(self.lists[key])

    def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        stop = None if end == -1 else end + 1
        return values[start:stop]

    def lset(self, key, index, value):
        self.lists[key][index] = value

    def llen(self, key):
        return len(self.lists.get(key, []))

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)
        return 1

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def srem(self, key, value):
        values = self.sets.get(key, set())
        existed = value in values
        values.discard(value)
        return 1 if existed else 0

    def delete(self, *keys):
        deleted = 0
        for key in keys:
            if key in self.lists:
                deleted += 1
                del self.lists[key]
            if key in self.sets:
                deleted += 1
                del self.sets[key]
        return deleted

    def expire(self, key, ttl):
        self.expirations[key] = ttl

    def close(self):
        self.closed = True


def test_redis_memory_store_messages_appends_and_loads_in_order():
    backend = RedisMemoryBackend(client=FakeRedis())

    assert backend.store_messages("s", [HumanMessage(content="first")]) == 1
    assert backend.store_messages("s", [AIMessage(content="second")]) == 1

    loaded = backend.load_messages("s")
    assert [message.content for message in loaded] == ["first", "second"]
    assert [message.type for message in loaded] == ["human", "ai"]
    assert backend.get_entry_count("s") == 2


def test_redis_memory_store_entry_replaces_existing_index():
    backend = RedisMemoryBackend(client=FakeRedis())
    backend.store_messages("s", [HumanMessage(content="original")])

    backend.store_entry("s", MemoryEntry("s", 0, "human", "updated", datetime.now()))

    loaded = backend.load_messages("s")
    assert len(loaded) == 1
    assert loaded[0].content == "updated"


def test_redis_memory_sessions_delete_clear_and_close():
    client = FakeRedis()
    backend = RedisMemoryBackend(client=client, ttl_seconds=60)
    backend.store_messages("b", [HumanMessage(content="b")])
    backend.store_messages("a", [HumanMessage(content="a")])

    assert backend.list_sessions() == ["a", "b"]
    assert backend.get_entry_count() == 2
    assert backend.delete_session("a") is True
    assert backend.delete_session("missing") is False
    assert backend.list_sessions() == ["b"]
    assert client.expirations["langdeep:memory:session:b"] == 60

    backend.clear()
    assert backend.list_sessions() == []
    assert backend.load_messages("b") == []

    backend.close()
    assert client.closed is True


def test_redis_memory_roundtrips_tool_calls():
    backend = RedisMemoryBackend(client=FakeRedis())
    message = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {"query": "refund"}, "id": "call_1"}],
    )

    backend.store_messages("s", [message])

    loaded = backend.load_messages("s")
    assert loaded[0].tool_calls[0]["name"] == "lookup"


def test_redis_memory_requires_optional_dependency_without_client(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("missing redis")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        RedisMemoryBackend()
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "langdeep[redis]" in str(exc)


# ── SQLiteMemoryBackend ──────────────────────────────────────────────────────


def test_sqlite_memory_store_messages_appends_and_persists(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    backend = SQLiteMemoryBackend(db_path)
    backend.store_messages("s", [HumanMessage(content="first")])
    backend.store_messages("s", [AIMessage(content="second")])
    backend.close()

    reopened = SQLiteMemoryBackend(db_path)
    try:
        loaded = reopened.load_messages("s")
        assert [message.content for message in loaded] == ["first", "second"]
        assert [message.type for message in loaded] == ["human", "ai"]
        assert reopened.get_entry_count("s") == 2
    finally:
        reopened.close()


def test_sqlite_memory_store_entry_replaces_existing_index(tmp_path):
    backend = SQLiteMemoryBackend(tmp_path / "memory.sqlite3")
    try:
        backend.store_messages("s", [HumanMessage(content="original")])
        backend.store_entry("s", MemoryEntry("s", 0, "human", "updated", datetime.now()))

        loaded = backend.load_messages("s")
        assert len(loaded) == 1
        assert loaded[0].content == "updated"
    finally:
        backend.close()


def test_sqlite_memory_sessions_delete_and_clear(tmp_path):
    backend = SQLiteMemoryBackend(tmp_path / "memory.sqlite3")
    try:
        backend.store_messages("b", [HumanMessage(content="b")])
        backend.store_messages("a", [HumanMessage(content="a")])

        assert backend.list_sessions() == ["a", "b"]
        assert backend.get_entry_count() == 2
        assert backend.delete_session("a") is True
        assert backend.delete_session("missing") is False
        assert backend.list_sessions() == ["b"]

        backend.clear()
        assert backend.list_sessions() == []
        assert backend.load_messages("b") == []
    finally:
        backend.close()


def test_sqlite_memory_roundtrips_tool_calls_and_additional_kwargs(tmp_path):
    backend = SQLiteMemoryBackend(tmp_path / "memory.sqlite3")
    message = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {"query": "refund"}, "id": "call_1"}],
        additional_kwargs={"reasoning_content": "checked policy"},
    )
    try:
        backend.store_messages("s", [message])

        loaded = backend.load_messages("s")
        assert loaded[0].tool_calls[0]["name"] == "lookup"
        assert loaded[0].additional_kwargs["reasoning_content"] == "checked policy"
    finally:
        backend.close()


def test_sqlite_memory_accepts_in_memory_connection():
    import sqlite3

    connection = sqlite3.connect(":memory:")
    backend = SQLiteMemoryBackend(connection=connection)
    backend.store_messages("s", [HumanMessage(content="hi")])

    assert backend.load_messages("s")[0].content == "hi"
    backend.close()
    connection.execute("SELECT COUNT(*) FROM memory_entries")
    connection.close()


# ── Serialization ────────────────────────────────────────────────────────────


def test_serialize_human_message():
    """Serialize a HumanMessage and verify fields."""
    msg = HumanMessage(content="test content")
    entry = serialize_message(msg, "s", 0)
    assert entry.role == "human"
    assert entry.content == "test content"
    assert entry.message_idx == 0


def test_serialize_ai_with_tool_calls():
    """Serialize an AIMessage with tool_calls preserves tool_calls JSON."""
    import json
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1"}],
    )
    entry = serialize_message(msg, "s", 0)
    assert entry.role == "ai"
    assert entry.tool_calls is not None
    tool_calls = json.loads(entry.tool_calls)
    assert tool_calls[0]["name"] == "get_weather"


def test_deserialize_all_roles():
    """Deserialize entries for all four roles returns correct message types."""
    entries = [
        MemoryEntry("s", 0, "human", "human text", datetime.now()),
        MemoryEntry("s", 1, "ai", "ai text", datetime.now()),
        MemoryEntry("s", 2, "system", "system text", datetime.now()),
        MemoryEntry("s", 3, "tool", "tool result", datetime.now()),
    ]
    msgs = [deserialize_message(e) for e in entries]
    assert type(msgs[0]).__name__ == "HumanMessage"
    assert type(msgs[1]).__name__ == "AIMessage"
    assert type(msgs[2]).__name__ == "SystemMessage"
    assert type(msgs[3]).__name__ == "ToolMessage"
    # ToolMessage requires tool_call_id, verify default was set
    assert msgs[3].tool_call_id == "unknown"


def test_serialize_roundtrip():
    """Serialize a message and deserialize it back, content is preserved."""
    original = HumanMessage(content="roundtrip test")
    entry = serialize_message(original, "s", 0)
    restored = deserialize_message(entry)
    assert restored.content == original.content
    assert restored.type == original.type


# ── MemoryRegistry ──────────────────────────────────────────────────────────


def test_memory_registry_register_and_get():
    """Register a backend factory and retrieve the instance."""
    reg = MemoryRegistry()
    reg.register("my_backend", lambda: InMemoryBackend(), {"description": "test"})
    backend = reg.get_backend("my_backend")
    assert isinstance(backend, InMemoryBackend)
    assert "my_backend" in reg.list_backends()


def test_memory_registry_get_unregistered():
    """Getting an unregistered backend raises ConfigurationError."""
    from langdeep.core.errors import ConfigurationError
    reg = MemoryRegistry()
    try:
        reg.get_backend("does_not_exist")
        assert False, "Expected ConfigurationError"
    except ConfigurationError:
        pass


def test_memory_registry_remove():
    """Remove a registered backend."""
    reg = MemoryRegistry()
    reg.register("tmp", lambda: InMemoryBackend(), {})
    reg.remove("tmp")
    assert "tmp" not in reg.list_backends()


def test_memory_registry_get_metadata():
    """get_metadata returns the metadata dict."""
    reg = MemoryRegistry()
    reg.register("m", lambda: InMemoryBackend(), {"description": "my backend"})
    meta = reg.get_metadata("m")
    assert meta["description"] == "my backend"


# ── @memory decorator ───────────────────────────────────────────────────────


def test_memory_decorator_builtin():
    """@memory with pass uses InMemoryBackend automatically."""

    @memory(name="builtin_mem", description="no factory")
    def builtin_mem():
        pass

    backend = memory_registry.get_backend("builtin_mem")
    assert isinstance(backend, InMemoryBackend)
    backend.store_messages("s", [HumanMessage(content="works")])
    assert len(backend.load_messages("s")) == 1


def test_memory_decorator_custom():
    """@memory returning a custom backend uses the returned instance."""

    class CustomBackend(InMemoryBackend):
        pass

    @memory(name="custom_mem", description="custom backend")
    def custom_mem():
        return CustomBackend()

    backend = memory_registry.get_backend("custom_mem")
    assert isinstance(backend, CustomBackend)


def test_memory_decorator_name_defaults_to_function_name():
    """@memory without name uses the function name."""

    @memory()
    def my_default_name():
        pass

    assert "my_default_name" in memory_registry.list_backends()
    backend = memory_registry.get_backend("my_default_name")
    assert isinstance(backend, InMemoryBackend)
