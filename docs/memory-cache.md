# Memory And Cache

LangDeep separates conversation memory from generic cache backends.

## Memory

Memory backends implement session message storage. The built-in
`InMemoryBackend` is useful for tests and examples.

```python
from langdeep import memory


@memory(name="session_memory")
def session_memory():
    pass
```

Use `FlowOrchestrator(..., memory="session_memory")` and pass `session_id` to
`chat()` or `chat_text()`.

## Cache

Cache backends implement `get`, `set`, `delete`, `has`, and `clear`.

LLM response caching is disabled by default because it changes model semantics.
