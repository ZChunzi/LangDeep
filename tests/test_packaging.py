"""Tests for packaging metadata and optional dependency groups."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility
    import tomli as tomllib


def test_provider_extras_cover_builtin_provider_groups():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    expected = {
        "openai",
        "azure-openai",
        "deepseek",
        "anthropic",
        "google-genai",
        "vertexai",
        "ollama",
        "all",
        "dev",
    }
    assert expected.issubset(extras)
    assert "langchain-google-vertexai" in extras["all"]
    assert "persist" not in extras
