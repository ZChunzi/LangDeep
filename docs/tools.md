# Tools

Tools are LangChain tools registered through `@register_tool`.

## Registration

```python
from langdeep import register_tool


@register_tool(name="lookup_order", description="Look up an order.")
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return "delivered"
```

## Policy

LangDeep can wrap tools with `PolicyAwareTool` to enforce confirmation,
workspace roots, timeout behavior, audit logging, and metrics collection.

Mark mutating tools with `requires_confirmation=True`. For file tools, configure
workspace roots and avoid passing raw user-controlled paths directly.
