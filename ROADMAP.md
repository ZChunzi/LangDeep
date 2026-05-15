# LangDeep Roadmap

This roadmap is a planning document, not a release promise. Priorities may
change based on user feedback, maintainer capacity, security needs, and upstream
LangChain/LangGraph changes.

## Current Focus

- Keep ordinary application development simple and low-boilerplate.
- Preserve advanced LangChain and LangGraph interoperability.
- Make multi-agent orchestration predictable, observable, and testable.
- Expand examples that can be run locally without paid services.

## Near-Term Priorities

### Usability

- Add more runnable examples under `examples/`.
- Improve public helper APIs around messages, chat, memory, and tool use.
- Keep README examples covered by tests.

### Core Runtime

- Tighten the agent `invoke` / `ainvoke` runtime contract.
- Improve structured workflow plan validation and status transitions.
- Reduce hidden fallback behavior in planner/router paths.
- Continue hardening orchestrator state handling.

### Backends and Integrations

- Add Redis and SQLite memory backends.
- Add Prometheus metrics export.
- Add OpenTelemetry tracing adapters.
- Add FastAPI and CLI examples for real deployment patterns.

### Security and Operations

- Document sandbox safety boundaries with concrete examples.
- Add stronger examples for tool confirmation and workspace policies.
- Improve security review checklists for provider, tool, and webhook changes.

## Later Work

- Docker-backed sandbox implementation.
- More provider compatibility adapters.
- More complete workflow template library.
- Real-world customer service demo.
- Better release automation and changelog generation.

## Not Planned Without Design Review

- Replacing LangChain/LangGraph core abstractions.
- Distributed registry storage as a built-in default.
- Running untrusted code without an external isolation layer.
- Network-dependent tests in the default CI path.
