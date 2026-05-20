# Decorators

Decorators are the main registration API.

## Models And Providers

```python
from langdeep import model


@model(name="mock_chat", provider="mock", model_name="mock-chat")
def mock_chat():
    pass
```

## Agents

```python
from langdeep import agent


@agent(name="support_agent", description="Handle support.", routing_keywords=["support"])
def support_agent():
    return MyRunnableAgent()
```

## Tools

Use `@register_tool` for LangChain-compatible tools. Tool functions must have a
docstring because LangChain validates tool descriptions.

## Storage, IM, And Sandbox

- `@memory`: register session memory backends.
- `@cache`: register cache backends.
- `@im_channel`: register messaging platform handlers.
- `@sandbox`: register code execution backends.

Run `validate_runtime(instantiate_agents=True)` after imports to catch broken
references.
