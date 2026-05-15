# Contributing to LangDeep

LangDeep is an annotation-driven multi-agent workflow framework built on
LangChain and LangGraph. Contributions should make the framework easier to use
without weakening its extension points, runtime safety, or compatibility with
the LangChain ecosystem.

## Good First Contribution Areas

Start with scoped tasks that have clear acceptance criteria:

- Documentation corrections in `README.md`, `README.zh-CN.md`, or
  `docs/developer-guide.md`.
- Examples under `examples/` that run without paid external services.
- Tests that cover public APIs such as `FlowOrchestrator`, decorators,
  registries, memory, sandbox, tools, and provider adapters.
- Small usability helpers that reduce boilerplate while preserving advanced
  LangChain/LangGraph interoperability.

Large API changes, provider additions, new storage backends, or sandbox changes
should start with an issue before implementation.

## Development Setup

```bash
git clone https://github.com/<your-user>/LangDeep.git
cd LangDeep/LangDeep
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Provider-specific extras are optional:

```bash
python -m pip install -e ".[deepseek]"
python -m pip install -e ".[all]"
```

## Quality Gates

Run these before opening a pull request:

```bash
python -m ruff check src tests
python -m pytest --cov=src --cov-report=term-missing
python -m compileall -q src tests
```

Use targeted tests while iterating:

```bash
python -m pytest tests/test_orchestrator.py
python -m pytest tests/test_readme_examples.py
```

Do not lower the configured coverage threshold to make a change pass.

## Coding Guidelines

- Keep public APIs explicit, typed where useful, and documented.
- Prefer existing registries, decorators, schemas, and extension points over
  parallel abstractions.
- Preserve backward compatibility unless an issue explicitly approves a
  breaking change.
- Validate inputs at framework boundaries and raise LangDeep structured errors.
- Keep provider-specific behavior inside provider adapters or provider
  factories.
- Do not add real network calls to tests.
- Do not commit secrets, generated coverage files, build artifacts, or local
  virtual environments.

## Documentation Guidelines

Documentation must match implemented behavior. When public behavior changes,
update the relevant docs in the same pull request:

- `README.md`
- `README.zh-CN.md`
- `docs/developer-guide.md`
- examples under `examples/`

README examples must be runnable. If an example requires external services,
say so explicitly and provide a no-network alternative.

## Pull Request Process

1. Pick or open an issue with clear acceptance criteria.
2. Keep the change scoped to that issue.
3. Add or update tests for behavior changes.
4. Update docs and examples for public API changes.
5. Fill in the PR template completely.
6. Wait for maintainer review before expanding scope.

## Commit Messages

Use concise Conventional Commit-style messages:

- `fix:` bug fixes
- `feat:` user-visible features
- `docs:` documentation-only changes
- `test:` tests
- `refactor:` behavior-preserving restructuring
- `chore:` maintenance

Examples:

```text
fix: normalize orchestrator invoke state input
docs: add runnable quick start example
test: cover public message helpers
```

## Review Expectations

Maintainers review for correctness, API consistency, test coverage, security
impact, documentation accuracy, and long-term maintainability. Reviews may ask
for smaller scope, additional tests, or clearer docs before merge.
