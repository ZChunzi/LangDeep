"""Manifest loading helpers for LangDeep skills."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from ..errors import ConfigurationError
from .models import SkillManifest


def load_skill_manifest(data: Dict[str, Any]) -> SkillManifest:
    """Validate a manifest dictionary."""
    if not isinstance(data, dict):
        raise ConfigurationError(
            "Skill manifest must be a mapping",
            context={"actual": type(data).__name__},
        )
    return SkillManifest.from_dict(data)


def load_skill_manifest_file(path: Union[str, Path]) -> SkillManifest:
    """Load a skill manifest from JSON or YAML."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ConfigurationError(
            "Skill manifest file does not exist",
            context={"path": str(manifest_path)},
        )
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    return load_skill_manifest(data)
