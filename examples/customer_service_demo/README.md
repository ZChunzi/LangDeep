# Customer Service Demo

This demo shows a small but realistic e-commerce support workflow using only
local mock components. It is intended as a learning entry point for LangDeep
runtime wiring.

It demonstrates:

- model registration with the built-in `mock` provider
- agent routing and multi-turn chat through `FlowOrchestrator.chat_text()`
- registered tools backed by a local mock knowledge base
- in-memory session memory
- tool execution policy with confirmation for case creation
- tool audit records and in-process metrics

Run from the repository root:

```bash
python examples/customer_service_demo/demo.py
```

The script runs a three-turn support conversation for order `LD-1001`, opens a
return case after the user confirms the action, and prints memory, audit, and
metrics summaries.

To adapt this demo for a real provider, replace `support_demo_mock_chat` with
your production model registration and keep the same agent, tool, memory, and
policy structure.
