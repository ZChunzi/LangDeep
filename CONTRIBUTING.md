# Contributing to LangDeep

Thank you for your interest in improving LangDeep. This guide explains how to report issues, propose changes, prepare a development environment, run quality gates, and submit pull requests that are easy to review.

LangDeep is an annotation-driven multi-agent workflow framework built on LangChain and LangGraph. Contributions should preserve the framework's current architecture, keep public APIs stable where possible, and improve reliability for real application and enterprise usage.

## Code of Conduct

Be respectful, specific, and constructive. Focus discussions on technical facts, reproducible behavior, and maintainable solutions. Harassment, personal attacks, or intentionally disruptive behavior are not acceptable in project spaces.

## Ways to Contribute

Useful contributions include:

- Bug reports with a minimal reproduction.
- Fixes for runtime, orchestration, registry, provider, cache, memory, sandbox, observability, or documentation issues.
- Tests that cover missing behavior or prevent regressions.
- Provider integrations that follow the existing `ModelConfig` and provider factory contracts.
- Documentation improvements that match the current implementation.
- Performance, reliability, or diagnostics improvements with clear tradeoffs.

Before starting a large feature or behavior-changing refactor, open an issue or discussion first. This helps align the design before implementation work begins.

## Reporting Issues

When opening a bug report, include:

- LangDeep version or commit SHA.
- Python version and operating system.
- Installed dependency set, especially optional provider packages.
- A minimal code sample or test case that reproduces the issue.
- Expected behavior and actual behavior.
- Full traceback or relevant logs, with secrets removed.

For feature requests, describe the use case, the proposed API or behavior, and why the existing extension points are insufficient.

## Security Reports

Do not report security vulnerabilities through public GitHub issues.

If you discover a vulnerability, contact the maintainer privately through the email listed in `pyproject.toml`. Include enough detail to reproduce and assess the issue. Avoid sharing exploit details publicly until a fix or mitigation is available.

Security-sensitive areas include:

- Sandbox execution.
- Secrets handling.
- IM/webhook integrations.
- External workflow plan ingestion.
- Tool execution boundaries.
- Provider authentication and request routing.

## Development Setup

Fork the repository, then clone your fork:

```bash
git clone https://github.com/<your-user>/LangDeep.git
cd LangDeep
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install development dependencies:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install optional provider and persistence dependencies only when your change needs them:

```bash
python -m pip install -e ".[all]"
python -m pip install -e ".[persist]"
```

## Repository Layout

The most important paths are:

- `src/`: LangDeep package source code.
- `tests/`: unit, integration, edge-case, and diagnostics tests.
- `tests/run_all.py`: repository test runner with ordered module execution.
- `scripts/run_tests.py`: convenience wrapper around `tests/run_all.py`.
- `docs/`: developer and architecture documentation.
- `pyproject.toml`: package metadata, dependencies, pytest, ruff, and coverage configuration.
- `README.md` and `README.zh-CN.md`: public project documentation.

## Development Workflow

Use a focused branch name:

```bash
git checkout -b fix/runtime-diagnostics
```

Keep changes scoped. Avoid unrelated formatting churn, generated artifacts, dependency lock changes, or broad refactors unless they are required for the issue being solved.

Recommended local loop:

```bash
python -m ruff check src tests
python -m pytest --cov --cov-report=term-missing --cov-report=xml
python tests/run_all.py
python -m compileall -q src tests
python -m build --no-isolation
```

For faster iteration, run a targeted subset first:

```bash
python -m pytest tests/test_diagnostics.py
python tests/run_all.py --filter diagnostics
```

## Quality Gates

Pull requests should pass the same checks used by maintainers:

```bash
# Static correctness checks
python -m ruff check src tests

# Standard pytest suite with coverage
python -m pytest --cov --cov-report=term-missing --cov-report=xml

# LangDeep custom ordered runner
python tests/run_all.py

# Syntax and import compilation check
python -m compileall -q src tests

# Package build check
python -m build --no-isolation
```

The current coverage threshold is configured in `pyproject.toml` under `[tool.coverage.report]`. Do not lower the threshold to make a pull request pass.

## Testing Guidelines

Prefer tests that exercise public behavior instead of private implementation details.

Use the existing test patterns:

- Reset singleton registries between tests when registering models, tools, agents, memory, cache, or providers.
- Use local mock models and test doubles instead of real LLM API calls.
- Use `tmp_path` for filesystem work.
- Cover both success and failure paths for registries, orchestrator flows, diagnostics, provider setup, and execution policies.
- Add regression tests for every bug fix.
- Keep tests deterministic; avoid sleeps, network calls, wall-clock assumptions, and externally mutable state.

When adding a new test module for the custom runner, register it in `TEST_MODULES` inside `tests/run_all.py` in dependency order.

## Code Style

LangDeep targets Python 3.9 and newer.

Follow these expectations:

- Keep public APIs explicit and documented.
- Preserve backward compatibility unless the change is intentionally versioned and documented.
- Prefer existing registries, decorators, schemas, and extension points over new parallel abstractions.
- Validate inputs at framework boundaries and return structured diagnostics where possible.
- Keep provider-specific behavior inside provider factories or provider modules.
- Avoid hard-coded secrets, real API calls in tests, and environment-specific paths.
- Use concise comments only when the code is not self-explanatory.

## Documentation

Documentation changes should be accurate for the current codebase. If you change public behavior, update the relevant docs in the same pull request:

- `README.md` for user-facing English documentation.
- `README.zh-CN.md` for Simplified Chinese documentation when the same public information changes.
- `docs/developer-guide.md` for architecture, extension, testing, and operation details.
- Examples or docstrings when APIs change.

Do not document APIs that are not implemented. If a feature is experimental, state its current limitations clearly.

## Commit Messages

Use Conventional Commits:

- `feat:` user-visible feature.
- `fix:` bug fix.
- `docs:` documentation-only change.
- `test:` test-only change.
- `refactor:` behavior-preserving code restructuring.
- `perf:` performance improvement.
- `chore:` tooling, packaging, or maintenance.

Examples:

```text
fix: validate agent tool references during startup
docs: add multilingual readme
test: cover runtime diagnostics warnings
```

## Pull Request Checklist

Before opening a pull request, confirm that:

- The PR has a clear title and describes the reason for the change.
- The implementation is scoped to the issue or feature.
- Public API changes are documented.
- New behavior has tests.
- The quality gate commands pass locally, or any skipped command is explained.
- No secrets, credentials, generated caches, coverage files, or local build artifacts are included.
- Backward compatibility risks are called out in the PR description.

## Review Process

Maintainers review for correctness, API consistency, test coverage, security implications, and documentation accuracy. Reviews may request changes before merge. Keep follow-up commits focused and avoid force-pushing unrelated rewrites during review unless needed to resolve conflicts or clean up history.

## Release Notes

Changes that affect users should include a short release-note style summary in the PR description. Mention migrations, behavior changes, new dependencies, security implications, or operational impact.

## License

By contributing to LangDeep, you agree that your contributions are licensed under the [MIT License](./LICENSE).
