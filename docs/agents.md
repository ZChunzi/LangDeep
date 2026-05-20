# Agents

Agents are runnable objects registered with metadata.

## Contract

An agent should implement `invoke(self, state)`. For async compatibility, also
implement `ainvoke(self, state)`.

Return a graph-compatible state update, usually a `messages` list containing an
assistant message.

## Routing Metadata

Use clear descriptions and routing keywords. Keyword routing is checked before
LLM routing, so it is useful for common intents.

## Tool References

Agent metadata may list tools. `validate_runtime()` checks that referenced tools
exist and reports missing references before serving traffic.
