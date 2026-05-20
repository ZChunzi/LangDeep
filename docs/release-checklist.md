# Release Checklist

Use this checklist for every LangDeep release. The goal is a repeatable process
that can be audited after the fact.

## 1. Prepare

- Confirm the release owner and reviewer.
- Confirm all release-blocking issues and pull requests are closed or deferred.
- Confirm the target version follows the project versioning policy.
- Confirm PyPI and TestPyPI trusted publishing environments are configured in
  repository settings.
- Confirm tag protection for `v*` is configured in repository settings.

## 2. Update Versioned Files

- Update `pyproject.toml` `project.version`.
- Update `src/__init__.py` `__version__`.
- Update README and documentation version references when they mention the
  release version.
- Update `CHANGELOG.md` by moving relevant entries from `Unreleased` into a
  dated release section.

## 3. Validate Locally

Run from a clean checkout:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests docs examples README.md README.zh-CN.md SECURITY.md pyproject.toml
python -m pytest --cov=src --cov-report=term-missing --cov-report=xml
python tests/run_all.py
python -m compileall -q src tests examples
python -m build --no-isolation
python -m twine check dist/*
```

Inspect `dist/` and confirm both wheel and sdist were created.

## 4. Publish to TestPyPI

- Open the `Publish Python distribution` GitHub Actions workflow.
- Run it manually with repository `testpypi`.
- Confirm the workflow uses GitHub OIDC trusted publishing rather than a stored
  PyPI token.
- Install from TestPyPI in a temporary environment:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ langdeep
python -c "import langdeep; print(langdeep.__version__)"
```

## 5. Tag and Publish to PyPI

- Create an annotated tag:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

- Confirm the publish workflow completes for the `pypi` environment.
- Confirm the release is visible on PyPI.
- Create a GitHub release using the changelog section as release notes.

## 6. Post-Release

- Verify installation from PyPI in a fresh virtual environment.
- Run a no-network smoke test:

```bash
python - <<'PY'
import langdeep
from langdeep import validate_runtime
print(langdeep.__version__)
print(validate_runtime().to_dict())
PY
```

- Close or update the release tracking issue.
- Announce the release with links to PyPI, GitHub release notes, and changelog.

## 7. Rollback

PyPI files cannot be replaced safely. If a release is bad:

- Yank the affected version on PyPI instead of deleting files.
- Open a follow-up issue describing impact and mitigation.
- Publish a patch release with a new version.
- Document the yanked release and fix in `CHANGELOG.md`.
