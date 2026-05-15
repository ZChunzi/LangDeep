# Contributing to LangDeep

Thank you for contributing to LangDeep.

LangDeep is an annotation-driven multi-agent workflow framework built on LangChain and LangGraph. The project is in an early but active stage, so small, focused contributions are especially valuable.

## Good first contribution areas

Start with scoped tasks that have clear acceptance criteria:

- Documentation corrections in `README.md`, `README.zh-CN.md`, or `docs/developer-guide.md`.
- Runnable examples under `examples/` that run without paid external services.
- Tests for public APIs such as `FlowOrchestrator`, decorators, registries, memory, sandbox, tools, and provider adapters.
- Small usability helpers that reduce boilerplate while preserving advanced LangChain/LangGraph interoperability.

Large API changes, provider additions, new storage backends, sandbox changes, or orchestration behavior changes should start with an issue before implementation.

## Development setup

```bash
git clone https://github.com/ZChunzi/LangDeep.git
cd LangDeep
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel build
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
git clone https://github.com/ZChunzi/LangDeep.git
cd LangDeep
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel build
pip install -e ".[dev]"
```

Provider-specific extras are optional:

```bash
python -m pip install -e ".[deepseek]"
python -m pip install -e ".[all]"
```

## Local checks

Run the relevant checks before opening a pull request:

```bash
python -m ruff check src tests
python -m pytest --ignore=tests/test_sandbox.py
python -m compileall -q src tests
python -m build --no-isolation
```

If you change examples, also run:

```bash
python -m ruff check src tests examples
python -m compileall -q src tests examples
```

The subprocess sandbox tests may depend on local OS behavior and are excluded from the default CI smoke suite.

## Coding guidelines

- Keep public APIs explicit, typed where useful, and documented.
- Prefer existing registries, decorators, schemas, and extension points over parallel abstractions.
- Preserve backward compatibility unless an issue explicitly approves a breaking change.
- Validate inputs at framework boundaries and raise LangDeep structured errors.
- Keep provider-specific behavior inside provider adapters or provider factories.
- Do not add real network calls to unit tests.
- Do not commit secrets, generated coverage files, build artifacts, or local virtual environments.

## Documentation guidelines

Documentation must match implemented behavior. When public behavior changes, update the relevant docs in the same pull request:

- `README.md`
- `README.zh-CN.md`
- `docs/developer-guide.md`
- examples under `examples/`

README examples must be runnable. If an example requires external services, say so explicitly and provide a no-network alternative.

## Pull request process

1. Pick or open an issue with clear acceptance criteria.
2. Keep the change scoped to that issue.
3. Add or update tests for behavior changes.
4. Update docs and examples for public API changes.
5. Fill in the PR template completely.
6. Wait for maintainer review before expanding scope.

## Commit messages

Use concise Conventional Commit-style messages:

- `fix:` bug fixes
- `feat:` user-visible features
- `docs:` documentation-only changes
- `test:` tests
- `refactor:` behavior-preserving restructuring
- `ci:` CI and workflow changes
- `chore:` maintenance

Examples:

```text
fix: normalize orchestrator invoke state input
docs: add runnable quick start example
test: cover public message helpers
```

## Security and privacy

- Do not include API keys, private logs, customer data, access tokens, or credentials in issues, PRs, tests, or examples.
- Do not run untrusted code through the built-in subprocess sandbox in production.
- Report security vulnerabilities privately. See `SECURITY.md`.

## Review expectations

Maintainers review for correctness, API consistency, test coverage, security impact, documentation accuracy, and long-term maintainability. Reviews may ask for smaller scope, additional tests, or clearer docs before merge.
