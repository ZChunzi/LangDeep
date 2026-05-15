# Maintainers

Maintainers are responsible for keeping LangDeep usable, secure, tested, and
aligned with its project scope.

## Current Maintainer

- `ZChunzi`

## Maintainer Responsibilities

- Triage issues and label them by type, area, level, and priority.
- Review pull requests for correctness, compatibility, tests, and docs.
- Keep public examples runnable.
- Protect release quality and versioning discipline.
- Coordinate security reports privately.
- Keep governance, roadmap, and contributor docs current.

## Review Standards

Maintainers should check:

- Does the change solve the stated issue?
- Does it preserve backward compatibility?
- Are public APIs documented?
- Are tests focused and deterministic?
- Does the change introduce security or operational risk?
- Is the implementation consistent with existing registry/decorator/runtime
  patterns?

## Release Responsibilities

Before tagging a release, maintainers should verify:

```bash
python -m ruff check src tests
python -m pytest --cov=src --cov-report=term-missing
python -m compileall -q src tests
```

Use the smallest reasonable version bump for each release. Patch releases are
preferred for compatibility fixes, documentation updates, and small usability
improvements.

## Adding Maintainers

New maintainers should have a sustained contribution history, good review
judgment, respect for project scope, and consistent handling of security and
compatibility concerns. Maintainer additions require explicit approval from the
current maintainer set.
