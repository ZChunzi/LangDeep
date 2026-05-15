# Maintainers

This document describes how LangDeep maintainership works.

Maintainers are responsible for keeping LangDeep usable, secure, tested, and aligned with its project scope.

## Current maintainer

- `@ZChunzi` — project owner and core maintainer

## Maintainer ladder

LangDeep uses a gradual maintainer model.

### Contributor

A contributor submits issues, discussions, documentation changes, examples, tests, bug fixes, or feature PRs.

### Regular contributor

A regular contributor has multiple accepted contributions and understands the project's review and testing expectations.

### Module maintainer

A module maintainer is trusted to review and help maintain a specific area, such as:

- orchestrator/runtime
- providers
- tools and policy
- memory/cache
- sandbox/security
- observability
- documentation/examples
- CI/release engineering

### Core maintainer

A core maintainer can approve larger architectural changes, help manage releases, and make cross-module decisions.

## Becoming a maintainer

A contributor may be considered for module maintainership after:

- making several high-quality merged PRs
- demonstrating good issue and review communication
- understanding LangDeep's runtime contracts and security boundaries
- helping other contributors successfully land changes

## Maintainer responsibilities

- Triage issues and label them by type, area, level, and priority.
- Review pull requests for correctness, compatibility, tests, and docs.
- Keep public examples runnable.
- Protect release quality and versioning discipline.
- Coordinate security reports privately.
- Keep governance, roadmap, and contributor docs current.

## Review standards

Maintainers should check:

- Does the change solve the stated issue?
- Does it preserve backward compatibility?
- Are public APIs documented?
- Are tests focused and deterministic?
- Does the change introduce security or operational risk?
- Is the implementation consistent with existing registry/decorator/runtime patterns?
- Does the change fit the roadmap or linked issue?

## Merge policy

For early-stage development:

- small documentation and example PRs may be squash-merged after review
- behavior changes should include tests
- public API changes should update documentation
- security-sensitive changes require extra review
- large architecture changes should start with an issue or design discussion

## Release responsibilities

Before tagging a release, maintainers should verify:

```bash
python -m ruff check src tests
python -m pytest --ignore=tests/test_sandbox.py
python -m compileall -q src tests
python -m build --no-isolation
```

Use the smallest reasonable version bump for each release. Patch releases are preferred for compatibility fixes, documentation updates, and small usability improvements.

## Adding maintainers

New maintainers should have a sustained contribution history, good review judgment, respect for project scope, and consistent handling of security and compatibility concerns. Maintainer additions require explicit approval from the current maintainer set.
