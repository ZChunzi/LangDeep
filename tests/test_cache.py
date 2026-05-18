"""Unit tests for the cache module: MemoryCache, CacheRegistry, @cache decorator."""

import time

from langdeep.core.cache import (
    MemoryCache,
    FileCacheBackend,
    BaseCacheBackend,
    CacheRegistry,
    cache_registry,
    cache,
)


def setup_function():
    """Reset singleton registries before each test."""
    from tests.conftest import clean_registries
    clean_registries()


# ── MemoryCache basic operations ────────────────────────────────────────────


def test_cache_get_set():
    """Set a value and get it back."""
    c = MemoryCache()
    c.set("key1", "value1")
    assert c.get("key1") == "value1"


def test_cache_get_missing():
    """Getting a missing key returns None."""
    c = MemoryCache()
    assert c.get("nothing") is None


def test_cache_get_after_delete():
    """After delete, the key is gone."""
    c = MemoryCache()
    c.set("k", "v")
    c.delete("k")
    assert c.get("k") is None


def test_cache_delete_existing():
    """Delete returns True for existing keys."""
    c = MemoryCache()
    c.set("k", "v")
    assert c.delete("k") is True


def test_cache_delete_not_found():
    """Delete returns False for non-existent keys."""
    c = MemoryCache()
    assert c.delete("missing") is False


def test_cache_has():
    """has() reflects key existence."""
    c = MemoryCache()
    c.set("k", "v")
    assert c.has("k") is True
    c.delete("k")
    assert c.has("k") is False


def test_cache_clear():
    """Clear removes all entries."""
    c = MemoryCache()
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert c.get("a") is None
    assert c.get("b") is None
    assert len(c) == 0


def test_cache_len():
    """__len__ returns correct count."""
    c = MemoryCache()
    assert len(c) == 0
    c.set("a", 1)
    assert len(c) == 1
    c.set("b", 2)
    assert len(c) == 2
    c.delete("a")
    assert len(c) == 1


# ── LRU eviction ────────────────────────────────────────────────────────────


def test_cache_lru_eviction():
    """When over max_size, the oldest entry is evicted."""
    c = MemoryCache(max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # 'a' should be evicted
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_cache_lru_access_updates_order():
    """Accessing an entry moves it to recently-used position."""
    c = MemoryCache(max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    # Access 'a', making 'b' the oldest
    c.get("a")
    c.set("c", 3)  # 'b' should be evicted
    assert c.get("a") == 1  # still there (recently used)
    assert c.get("b") is None  # evicted
    assert c.get("c") == 3


def test_cache_lru_set_updates_order():
    """Re-setting an existing key makes it recently used."""
    c = MemoryCache(max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("a", 100)  # update 'a', making 'b' the oldest
    c.set("c", 3)    # 'b' should be evicted
    assert c.get("a") == 100
    assert c.get("b") is None
    assert c.get("c") == 3


def test_cache_no_eviction_under_limit():
    """Entries within max_size are all kept."""
    c = MemoryCache(max_size=10)
    for i in range(10):
        c.set(f"k{i}", i)
    for i in range(10):
        assert c.get(f"k{i}") == i


# ── TTL ─────────────────────────────────────────────────────────────────────


def test_cache_ttl_expiry():
    """Entries expire after their TTL."""
    c = MemoryCache(default_ttl=None)
    c.set("k", "v", ttl=0.01)  # 10ms TTL
    assert c.get("k") == "v"  # still there immediately
    time.sleep(0.02)
    assert c.get("k") is None  # expired


def test_cache_default_ttl():
    """default_ttl is used when no per-key TTL is given."""
    c = MemoryCache(default_ttl=0.01)
    c.set("k", "v")
    assert c.get("k") == "v"
    time.sleep(0.02)
    assert c.get("k") is None


def test_cache_no_ttl():
    """Without TTL, entries persist indefinitely."""
    c = MemoryCache(default_ttl=None)
    c.set("k", "v")
    assert c.get("k") == "v"
    # no expiry


# ── Thread safety ───────────────────────────────────────────────────────────


def test_cache_concurrent_access():
    """Concurrent get/set does not corrupt data."""
    import threading

    c = MemoryCache(max_size=100)
    errors = []

    def worker(n):
        for i in range(100):
            try:
                c.set(f"w{n}_k{i}", n * 1000 + i)
                val = c.get(f"w{n}_k{i}")
                assert val == n * 1000 + i
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"Concurrent access failed: {errors}"


# ── FileCacheBackend ────────────────────────────────────────────────────────


def test_file_cache_get_set_and_miss(tmp_path):
    """FileCacheBackend stores and retrieves pickle-serializable values."""
    c = FileCacheBackend(str(tmp_path), max_entries=10)
    c.set("key1", {"value": 1})

    assert c.get("key1") == {"value": 1}
    assert c.get("missing") is None
    assert c.has("key1") is True
    assert len(c) == 1


def test_file_cache_persists_across_instances(tmp_path):
    """A new FileCacheBackend instance can read existing cache entries."""
    first = FileCacheBackend(str(tmp_path), max_entries=10)
    first.set("key1", "persisted")

    second = FileCacheBackend(str(tmp_path), max_entries=10)
    assert second.get("key1") == "persisted"


def test_file_cache_ttl_expiry(tmp_path):
    """Expired file cache entries are removed and not returned."""
    c = FileCacheBackend(str(tmp_path), max_entries=10, default_ttl=0.01)
    c.set("short", "lived")
    assert c.get("short") == "lived"

    time.sleep(0.02)
    assert c.get("short") is None
    assert len(c) == 0


def test_file_cache_max_entries_evicts_oldest(tmp_path):
    """FileCacheBackend evicts least recently accessed entries over max_entries."""
    c = FileCacheBackend(str(tmp_path), max_entries=2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1
    c.set("c", 3)

    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3
    assert len(c) == 2


def test_file_cache_delete_and_clear(tmp_path):
    """Delete and clear remove file-backed entries."""
    c = FileCacheBackend(str(tmp_path), max_entries=10)
    c.set("a", 1)
    c.set("b", 2)

    assert c.delete("a") is True
    assert c.delete("a") is False
    assert c.get("a") is None
    assert c.get("b") == 2

    c.clear()
    assert c.get("b") is None
    assert len(c) == 0


def test_file_cache_rejects_file_path(tmp_path):
    """FileCacheBackend requires a directory path."""
    file_path = tmp_path / "cache-file"
    file_path.write_text("not a directory")

    try:
        FileCacheBackend(str(file_path))
        assert False, "Should reject file path"
    except ValueError as exc:
        assert "directory" in str(exc)


# ── CacheRegistry ───────────────────────────────────────────────────────────


def test_cache_registry_register_and_get():
    """Register a cache backend factory and retrieve the instance."""
    reg = CacheRegistry()
    reg.register("my_cache", lambda: MemoryCache(max_size=10), {"ttl": 300})
    backend = reg.get_backend("my_cache")
    assert isinstance(backend, MemoryCache)
    assert "my_cache" in reg.list_backends()


def test_cache_registry_get_unregistered():
    """Getting an unregistered cache backend raises ConfigurationError."""
    from langdeep.core.errors import ConfigurationError
    reg = CacheRegistry()
    try:
        reg.get_backend("ghost")
        assert False, "Expected ConfigurationError"
    except ConfigurationError:
        pass


def test_cache_registry_metadata():
    """get_metadata returns the correct metadata."""
    reg = CacheRegistry()
    reg.register("m", lambda: MemoryCache(), {"ttl": 600})
    meta = reg.get_metadata("m")
    assert meta["ttl"] == 600


def test_cache_registry_remove():
    """Remove a registered cache backend."""
    reg = CacheRegistry()
    reg.register("tmp", lambda: MemoryCache(), {})
    reg.remove("tmp")
    assert "tmp" not in reg.list_backends()


# ── @cache decorator ────────────────────────────────────────────────────────


def test_cache_decorator_builtin():
    """@cache with pass uses built-in MemoryCache."""

    @cache(name="my_cache", ttl=300, max_entries=64)
    def my_cache():
        pass

    backend = cache_registry.get_backend("my_cache")
    assert isinstance(backend, MemoryCache)
    backend.set("test", "ok")
    assert backend.get("test") == "ok"


def test_cache_decorator_custom():
    """@cache returning a custom backend uses the returned instance."""

    class CustomCache(MemoryCache):
        pass

    @cache(name="custom_cache", ttl=600)
    def custom_cache():
        return CustomCache(max_size=32)

    backend = cache_registry.get_backend("custom_cache")
    assert isinstance(backend, CustomCache)


def test_cache_decorator_name_defaults_to_function():
    """@cache without name uses the function name."""

    @cache()
    def auto_named():
        pass

    assert "auto_named" in cache_registry.list_backends()


def test_cache_decorator_kwargs_passed():
    """The decorator passes max_entries correctly to MemoryCache."""

    @cache(name="small", max_entries=2)
    def small():
        pass

    backend = cache_registry.get_backend("small")
    backend.set("a", 1)
    backend.set("b", 2)
    backend.set("c", 3)  # evicts 'a'
    assert backend.get("a") is None
    assert backend.get("b") == 2
