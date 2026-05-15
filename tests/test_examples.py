"""Smoke tests for runnable examples."""

import importlib.util
from pathlib import Path

from conftest import clean_registries


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_basic_mock_agent_example_runs(capsys):
    clean_registries()
    example_path = Path(__file__).resolve().parents[1] / "examples" / "basic_mock_agent.py"
    module = load_module(example_path)

    try:
        module.main()
        output = capsys.readouterr().out
    finally:
        clean_registries()

    assert "Question: How is the weather in Beijing?" in output
    assert "Beijing: sunny, 25C" in output
