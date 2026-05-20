# Changelog

All notable changes to LangDeep are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning where practical.

## [Unreleased]

### Added

- Nothing yet.

### Changed

- Nothing yet.

## [2.0.17] - 2026-05-20

### Added

- `build_doctor_report(audit_sink=...)` audit diagnostics with schema version,
  sink type, durability classification, JSONL path reporting, and production
  warnings for missing or in-memory audit sinks.

## [2.0.16] - 2026-05-20

### Added

- Skill lifecycle states for plugin governance: `loaded`, `enabled`,
  `disabled`, `failed`, and `unloaded`.
- Skill registry lifecycle APIs: `enable()`, `disable()`, `unload()`,
  `get_lifecycle_state()`, `get_failure_reason()`, and `check_health()`.
- `SkillHealth`, `SkillHealthStatus`, and `SkillLifecycleState` exports for
  health hooks and lifecycle-aware service integrations.

## [2.0.15] - 2026-05-20

### Added

- Low-coupling `langdeep.core.protocols` module for MCP, A2A, and custom
  protocol endpoint declarations, callable/class adapters, namespace-aware
  registry routing, and decorator registration.
- Structured `ProtocolError` and `ProtocolAdapterNotFoundError` errors.

## [2.0.14] - 2026-05-20

### Added

- Low-coupling `langdeep.core.skills` module with manifest, runtime context,
  namespace-aware registry, decorator, and JSON/YAML manifest loaders.
- Enterprise audit foundation in `langdeep.core.audit`, including structured
  audit events, in-memory and JSONL sinks, and recursive sensitive-field
  redaction.
- Runtime `requirements.txt` generated from the core `pyproject.toml`
  dependencies.

### Changed

- CI now enables pip caching for Python setup steps.
- `.gitignore` now ignores only the repository-root `/workflows/` directory so
  `.github/workflows` remains trackable.

### Fixed

- Removed unresolved merge markers from `tests/test_observability.py`.

## [2.0.13] - 2026-05-18

### Added

- CLI entry point for LangDeep runtime inspection.
- Runtime diagnostics and health-check improvements.
- Public examples for mock-provider agent usage.

### Security

- Documented sandbox and subprocess boundary limitations.
