"""Low-coupling skill extension points."""

from .decorators import skill
from .loader import load_skill_manifest, load_skill_manifest_file
from .models import (
    SKILL_SCHEMA_VERSION,
    Skill,
    SkillAdapters,
    SkillCapability,
    SkillContext,
    SkillHealth,
    SkillHealthStatus,
    SkillLifecycleState,
    SkillManifest,
    capabilities_from_names,
)
from .registry import SkillRegistry, skill_registry

__all__ = [
    "SKILL_SCHEMA_VERSION",
    "Skill",
    "SkillAdapters",
    "SkillCapability",
    "SkillContext",
    "SkillHealth",
    "SkillHealthStatus",
    "SkillLifecycleState",
    "SkillManifest",
    "SkillRegistry",
    "capabilities_from_names",
    "load_skill_manifest",
    "load_skill_manifest_file",
    "skill",
    "skill_registry",
]
