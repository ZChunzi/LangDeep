"""Namespace-aware registry for LangDeep skills."""

import copy
import threading
from typing import Any, Dict, List, Optional

from ..errors import ConfigurationError, SkillNotFoundError
from ..logging import get_logger
from .models import (
    Skill,
    SkillAdapters,
    SkillContext,
    SkillFactory,
    SkillHealth,
    SkillLifecycleState,
    SkillManifest,
)
from .models import ensure_skill, normalize_adapters, normalize_health

logger = get_logger(__name__)


class SkillRegistry:
    """Registry for low-coupling skill factories and lifecycle state."""

    _registries: Dict[str, "SkillRegistry"] = {}
    _class_lock = threading.Lock()

    def __new__(cls, namespace: str = "default"):
        namespace = namespace or "default"
        with cls._class_lock:
            if namespace not in cls._registries:
                instance = super().__new__(cls)
                instance._namespace = namespace
                instance._factories: Dict[str, SkillFactory] = {}
                instance._manifests: Dict[str, SkillManifest] = {}
                instance._instances: Dict[str, Skill] = {}
                instance._active_adapters: Dict[str, SkillAdapters] = {}
                instance._lifecycle_states: Dict[str, SkillLifecycleState] = {}
                instance._failure_reasons: Dict[str, str] = {}
                instance._registry_lock = threading.RLock()
                cls._registries[namespace] = instance
            return cls._registries[namespace]

    @classmethod
    def for_namespace(cls, namespace: str) -> "SkillRegistry":
        """Return an isolated registry for a namespace."""
        return cls(namespace=namespace)

    @property
    def namespace(self) -> str:
        return self._namespace

    def register(
        self,
        name: str,
        factory: SkillFactory,
        manifest: Optional[SkillManifest] = None,
        *,
        replace: bool = True,
    ) -> None:
        """Register a skill factory without activating the skill."""
        name = _require_name(name)
        if manifest is not None and manifest.name != name:
            raise ConfigurationError(
                "Skill manifest name does not match registry key",
                context={"registry_name": name, "manifest_name": manifest.name},
            )
        with self._registry_lock:
            if name in self._factories and not replace:
                raise ConfigurationError(
                    f"Skill '{name}' is already registered",
                    context={"skill": name, "namespace": self._namespace},
                )
            self._factories[name] = factory
            if manifest is not None:
                self._manifests[name] = manifest
            else:
                self._manifests.pop(name, None)
            self._instances.pop(name, None)
            self._active_adapters.pop(name, None)
            self._lifecycle_states[name] = SkillLifecycleState.LOADED
            self._failure_reasons.pop(name, None)
        logger.info("Skill registered", extra={"skill": name, "namespace": self._namespace})

    def register_skill(self, skill: Skill, *, replace: bool = True) -> None:
        """Register an existing skill instance."""
        skill = ensure_skill(skill)
        name = skill.manifest.name
        with self._registry_lock:
            if name in self._factories and not replace:
                raise ConfigurationError(
                    f"Skill '{name}' is already registered",
                    context={"skill": name, "namespace": self._namespace},
                )
            self._factories[name] = lambda skill=skill: skill
            self._manifests[name] = skill.manifest
            self._instances[name] = skill
            self._active_adapters.pop(name, None)
            self._lifecycle_states[name] = SkillLifecycleState.LOADED
            self._failure_reasons.pop(name, None)
        logger.info("Skill instance registered", extra={"skill": name, "namespace": self._namespace})

    def get_skill(self, name: str) -> Skill:
        """Return a skill instance by name, creating it lazily."""
        name = _require_name(name)
        with self._registry_lock:
            if name not in self._factories:
                raise SkillNotFoundError(
                    f"Skill '{name}' is not registered",
                    context={"available": list(self._factories.keys()), "namespace": self._namespace},
                )
            if name not in self._instances:
                skill = ensure_skill(self._factories[name](), name=name)
                self._instances[name] = skill
                self._manifests.setdefault(name, skill.manifest)
            return self._instances[name]

    def get_manifest(self, name: str) -> SkillManifest:
        """Return a skill manifest without activating the skill."""
        name = _require_name(name)
        with self._registry_lock:
            manifest = self._manifests.get(name)
        if manifest is not None:
            return copy.deepcopy(manifest)
        return copy.deepcopy(self.get_skill(name).manifest)

    def list_skills(self) -> List[str]:
        """List registered skill names."""
        with self._registry_lock:
            return list(self._factories.keys())

    def activate(self, name: str, context: Optional[SkillContext] = None) -> SkillAdapters:
        """Validate and activate a skill."""
        context = context or SkillContext(namespace=self._namespace)
        skill = self.get_skill(name)
        try:
            warnings = skill.validate(context)
            context.audit(
                "skill.validate",
                skill=name,
                warnings=list(warnings or []),
                version=skill.manifest.version,
            )
            adapters = normalize_adapters(skill.activate(context), skill)
            with self._registry_lock:
                self._active_adapters[name] = adapters
                self._lifecycle_states[name] = SkillLifecycleState.ENABLED
                self._failure_reasons.pop(name, None)
            context.audit(
                "skill.activate",
                skill=name,
                version=skill.manifest.version,
                adapters=list(_non_empty_adapter_names(adapters)),
            )
            return adapters
        except Exception as exc:
            with self._registry_lock:
                self._lifecycle_states[name] = SkillLifecycleState.FAILED
                self._failure_reasons[name] = str(exc)
            context.audit(
                "skill.failed",
                skill=name,
                version=skill.manifest.version,
                reason=str(exc),
                phase="activate",
            )
            raise

    def enable(self, name: str, context: Optional[SkillContext] = None) -> SkillAdapters:
        """Enable a skill by validating and activating it."""
        return self.activate(name, context=context)

    def deactivate(self, name: str, context: Optional[SkillContext] = None) -> None:
        """Deactivate a skill if it is registered."""
        context = context or SkillContext(namespace=self._namespace)
        skill = self.get_skill(name)
        skill.deactivate(context)
        with self._registry_lock:
            self._active_adapters.pop(name, None)
            self._lifecycle_states[name] = SkillLifecycleState.DISABLED
            self._failure_reasons.pop(name, None)
        context.audit("skill.deactivate", skill=name, version=skill.manifest.version)

    def disable(self, name: str, context: Optional[SkillContext] = None) -> None:
        """Disable a skill and release active adapters."""
        self.deactivate(name, context=context)

    def unload(self, name: str, context: Optional[SkillContext] = None) -> None:
        """Unload a skill factory, instance, manifest, and active adapters."""
        name = _require_name(name)
        context = context or SkillContext(namespace=self._namespace)
        with self._registry_lock:
            if name not in self._factories and self._lifecycle_states.get(name) != SkillLifecycleState.UNLOADED:
                raise SkillNotFoundError(
                    f"Skill '{name}' is not registered",
                    context={"available": list(self._factories.keys()), "namespace": self._namespace},
                )
            skill = self._instances.get(name)
        if skill is not None and skill.active:
            skill.deactivate(context)
        with self._registry_lock:
            self._factories.pop(name, None)
            self._manifests.pop(name, None)
            self._instances.pop(name, None)
            self._active_adapters.pop(name, None)
            self._failure_reasons.pop(name, None)
            self._lifecycle_states[name] = SkillLifecycleState.UNLOADED
        context.audit("skill.unload", skill=name)

    def get_active_adapters(self, name: str) -> Optional[SkillAdapters]:
        """Return adapters from the last activation, if any."""
        with self._registry_lock:
            return self._active_adapters.get(name)

    def get_lifecycle_state(self, name: str) -> SkillLifecycleState:
        """Return the lifecycle state for a skill."""
        name = _require_name(name)
        with self._registry_lock:
            state = self._lifecycle_states.get(name)
            if state is None:
                raise SkillNotFoundError(
                    f"Skill '{name}' is not registered",
                    context={"available": list(self._factories.keys()), "namespace": self._namespace},
                )
            return state

    def get_failure_reason(self, name: str) -> Optional[str]:
        """Return the most recent lifecycle failure reason, if any."""
        name = _require_name(name)
        with self._registry_lock:
            return self._failure_reasons.get(name)

    def check_health(self, name: str, context: Optional[SkillContext] = None) -> SkillHealth:
        """Run a skill health hook and normalize its result."""
        context = context or SkillContext(namespace=self._namespace)
        skill = self.get_skill(name)
        try:
            health = normalize_health(skill.health(context))
            context.audit(
                "skill.health",
                skill=name,
                version=skill.manifest.version,
                status=health.status.value,
            )
            return health
        except Exception as exc:
            with self._registry_lock:
                self._lifecycle_states[name] = SkillLifecycleState.FAILED
                self._failure_reasons[name] = str(exc)
            context.audit(
                "skill.failed",
                skill=name,
                version=skill.manifest.version,
                reason=str(exc),
                phase="health",
            )
            raise

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable registry snapshot."""
        with self._registry_lock:
            return {
                "namespace": self._namespace,
                "skills": list(self._factories.keys()),
                "manifests": {
                    name: manifest.as_dict()
                    for name, manifest in self._manifests.items()
                },
                "active": sorted(self._active_adapters.keys()),
                "lifecycle": {
                    name: state.value
                    for name, state in self._lifecycle_states.items()
                },
                "failures": dict(self._failure_reasons),
            }

    def reset(self) -> None:
        """Clear all registered skills and lifecycle state."""
        with self._registry_lock:
            self._factories.clear()
            self._manifests.clear()
            self._instances.clear()
            self._active_adapters.clear()
            self._lifecycle_states.clear()
            self._failure_reasons.clear()


def _require_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("Skill name must be a non-empty string")
    return name.strip()


def _non_empty_adapter_names(adapters: SkillAdapters) -> List[str]:
    return [
        name
        for name, value in adapters.as_dict().items()
        if name != "metadata" and value
    ]


skill_registry = SkillRegistry()
