# LangDeep Roadmap

This roadmap is a planning document for contributors and maintainers. It is not a strict release promise. Priorities may change based on user feedback, maintainer capacity, security needs, and upstream LangChain/LangGraph changes.

## Current focus

LangDeep is moving from an early framework prototype toward a contributor-friendly, production-aware multi-agent workflow framework.

Priority themes:

1. Runtime stability
2. Runnable examples
3. Documentation accuracy
4. Provider compatibility
5. Persistence backends
6. Tool governance and auditability
7. Observability, diagnostics, and auditability
8. Production service boundary guidance
9. Plugin and skill governance

## Phase 1 — Contributor readiness

- [x] Add CI workflow
- [x] Add issue templates
- [x] Add pull request template
- [x] Add contributing guide
- [x] Add security policy
- [x] Add Dependabot configuration
- [x] Add CodeQL workflow
- [x] Ensure CI is stable across supported Python versions
- [ ] Add more `good first issue` tasks
- [ ] Add contributor-friendly examples

## Phase 2 — Runtime hardening

- [ ] Normalize `invoke` / `ainvoke` agent runnable contract
- [ ] Improve structured task result propagation
- [ ] Avoid string-based success/failure detection in aggregation
- [ ] Strengthen workflow plan validation and error messages
- [ ] Improve async execution fallback behavior
- [ ] Add regression tests for direct route and planned route edge cases

## Phase 3 — Examples and docs

- [ ] Add `examples/basic_mock_agent.py`
- [x] Add enterprise customer service workflow example
- [ ] Add provider examples for OpenAI, DeepSeek, and Ollama
- [ ] Add memory/cache backend examples
- [x] Add FastAPI integration example
- [ ] Keep README, README.zh-CN, and developer guide synchronized

## Phase 4 — Persistence and operations

- [x] Redis memory backend
- [x] SQLite memory backend
- [x] File-backed cache backend
- [x] Prometheus metrics exporter
- [x] OpenTelemetry tracing adapter
- [x] Structured audit log sink
- [ ] External durable checkpointer adapter
- [ ] Durable process state backend
- [ ] Idempotency keys for workflow and task execution

## Phase 5 — Security and sandboxing

- [x] Document safe tool policy patterns
- [x] Add Docker/container sandbox backend prototype
- [ ] Add stricter workspace policy examples
- [ ] Add security-focused tests for tool policy and sandbox boundaries
- [ ] Improve secret provider documentation

## Phase 6 — Release maturity

- [x] Formal release checklist
- [x] CHANGELOG.md
- [ ] CODEOWNERS after maintainers are added
- [x] PyPI trusted publishing
- [ ] Tag protection for `v*` release tags
- [ ] Optional docs site

## Phase 7 — Enterprise substrate and plugin governance

- [x] Low-coupling skill manifest and registry module
- [x] In-memory and JSONL audit sink foundation
- [ ] Production FastAPI starter with auth, rate limits, audit middleware, and metrics
- [ ] Plugin lifecycle states: loaded, enabled, disabled, failed, unloaded
- [ ] Registry namespace unification across all registries
- [ ] Orchestrator reload/rebuild API for dynamic component changes
- [ ] Workflow plan versioning and execution snapshots

## Phase 8 — Protocol ecosystem

- [x] Low-coupling protocol registry and MCP/A2A endpoint contracts
- [ ] MCP client adapter behind policy and audit controls
- [ ] MCP server adapter for selected LangDeep tools, prompts, resources, and skills
- [ ] A2A Agent Card generation and synchronous task support
- [ ] A2A client wrapper for remote agents under policy control

## Not planned without design review

- Replacing LangChain/LangGraph core abstractions
- Distributed registry storage as a built-in default
- Running untrusted code without an external isolation layer
- Network-dependent tests in the default CI path

## How to help

Look for issues labeled:

- `good first issue`
- `help wanted`
- `documentation`
- `enhancement`
- `area: examples`

Large changes should start with a design issue before implementation.
