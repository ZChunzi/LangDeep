"""Caching module with abstract backend interface and decorator support.

Provides:
- BaseCacheBackend — abstract interface for all cache backends
- MemoryCache — built-in LRU+TTL in-memory cache
- CacheRegistry — singleton registry for cache backend factories
- @cache — decorator for registering cache backends
"""

from .base import BaseCacheBackend
from .builtin import MemoryCache
from .registry import cache_registry, CacheRegistry
from .decorators import cache

__all__ = [
    "BaseCacheBackend",
    "MemoryCache",
    "CacheRegistry",
    "cache_registry",
    "cache",
]
