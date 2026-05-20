# Orchestrator

`FlowOrchestrator` is the main runtime entry point.

## Core Methods

- `invoke()`
- `ainvoke()`
- `astream()`
- `chat()`
- `chat_text()`
- `invoke_messages()`
- `invoke_state()`
- `health()`
- `get_metrics()`

## Inputs

`invoke()` accepts a string, one LangChain message, a sequence of messages, a
state dict with `messages`, or wrapper dicts with `input`, `user_input`, or
`content`.

## Multi-Turn Chat

Use `chat()` or `chat_text()` with a registered memory backend and stable
`session_id`.

```python
text = orchestrator.chat_text("Continue the case", session_id="case-123")
```
