"""Skill registration decorators."""

from functools import wraps
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from .models import Skill, SkillCapability, SkillManifest
from .registry import skill_registry


def skill(
    name: str,
    version: str,
    description: str = "",
    owner: str = "",
    tags: Optional[Sequence[str]] = None,
    required_permissions: Optional[Sequence[str]] = None,
    capabilities: Optional[Iterable[Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    compatibility: Optional[Dict[str, Any]] = None,
    extensions: Optional[Dict[str, Any]] = None,
    replace: bool = True,
):
    """Register a zero-argument skill factory.

    The decorated factory is registered only as a factory. The skill is not
    instantiated or activated until ``skill_registry.get_skill()`` or
    ``skill_registry.activate()`` is called.
    """

    manifest = SkillManifest(
        name=name,
        version=version,
        description=description,
        owner=owner,
        tags=list(tags or []),
        required_permissions=list(required_permissions or []),
        capabilities=[
            cap if isinstance(cap, SkillCapability) else SkillCapability.from_dict(cap)
            for cap in list(capabilities or [])
        ],
        dependencies=dict(dependencies or {}),
        compatibility=dict(compatibility or {}),
        extensions=dict(extensions or {}),
    )

    def decorator(factory: Callable[[], Skill]):
        skill_registry.register(manifest.name, factory, manifest=manifest, replace=replace)

        @wraps(factory)
        def wrapper(*args, **kwargs):
            return factory(*args, **kwargs)

        return wrapper

    return decorator
