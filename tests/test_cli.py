"""Tests for the LangDeep command line interface."""

import json

import pytest

from langdeep import __version__
from langdeep.cli import main
from langdeep.core.registry.agent_registry import AgentMetadata, agent_registry
from conftest import clean_registries


def setup_function():
    clean_registries()


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"langdeep {__version__}"


def test_cli_health_outputs_json(capsys):
    assert main(["health"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "healthy"
    assert payload["version"] == __version__
    assert "checks" in payload
    assert "timestamp" in payload


def test_cli_diagnostics_outputs_json(capsys):
    assert main(["diagnostics"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["error_count"] == 0
    assert payload["issues"] == []


def test_cli_diagnostics_returns_failure_for_errors(capsys):
    agent_registry.register(
        "broken",
        lambda: object(),
        AgentMetadata(name="broken", description="", model_name="missing_model"),
    )

    assert main(["diagnostics"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_count"] >= 1


def test_cli_list_registry_outputs_registered_names(capsys):
    agent_registry.register(
        "assistant",
        lambda: object(),
        AgentMetadata(name="assistant", description="Assistant"),
    )

    assert main(["list", "agents"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"agents": ["assistant"]}
