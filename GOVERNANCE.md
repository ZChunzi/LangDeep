# Governance

LangDeep uses lightweight maintainer-led governance. The goal is fast,
pragmatic progress while preserving reliability, compatibility, and a clear
project direction.

## Decision Making

Maintainers make final decisions on:

- Public API changes.
- Release timing and version numbers.
- Security fixes.
- Large refactors.
- New provider/backend integrations.
- Issue and pull request prioritization.

For routine fixes and documentation changes, one maintainer approval is
sufficient. For breaking changes, security-sensitive changes, or large design
changes, maintainers should request an issue or design discussion first.

## Project Scope

LangDeep provides:

- Registration-driven multi-agent workflow assembly.
- Orchestration on top of LangChain and LangGraph.
- Extension points for providers, tools, agents, memory, cache, sandbox,
  observability, and process lifecycle.
- Practical docs and runnable examples.

LangDeep does not aim to replace LangChain/LangGraph, provide a hosted agent
platform, or include production infrastructure such as distributed queues,
databases, or full telemetry stacks by default.

## Labels and Triage

Issues and PRs should use labels from these groups:

- `type: bug`, `type: docs`, `type: feature`, `type: refactor`, `type: test`
- `area: orchestrator`, `area: planner`, `area: executor`,
  `area: agent-builder`, `area: providers`, `area: memory`, `area: cache`,
  `area: sandbox`, `area: observability`, `area: docs`, `area: examples`
- `level: good-first-issue`, `level: intermediate`, `level: advanced`
- `priority: p0`, `priority: p1`, `priority: p2`

## Compatibility Policy

Prefer additive APIs and compatibility wrappers. If a breaking change is
unavoidable, document migration steps and use an appropriate version bump.

## Conflict Resolution

Technical disagreements should be resolved through code, tests, reproducible
examples, and documented tradeoffs. If consensus is not reached, maintainers
make the final decision based on project goals and maintenance cost.
