"""Core skill contracts for low-coupling capability packages."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..errors import ConfigurationError, SkillError

SKILL_SCHEMA_VERSION = "langdeep.skill.v1"
VALID_CAPABILITY_KINDS = {
    "tool",
    "agent",
    "prompt",
    "workflow",
    "resource",
    "mcp",
    "a2a",
}


class SkillLifecycleState(str, Enum):
    """Lifecycle state for a registered skill."""

    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    FAILED = "failed"
    UNLOADED = "unloaded"


class SkillHealthStatus(str, Enum):
    """Health status returned by a skill health hook."""

    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SkillCapability:
    """A single capability exposed by a skill."""

    kind: str
    name: str
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = _require_text(self.kind, "capability.kind")
        self.name = _require_text(self.name, "capability.name")
        if self.kind not in VALID_CAPABILITY_KINDS:
            raise ConfigurationError(
                f"Unsupported skill capability kind '{self.kind}'",
                context={"kind": self.kind, "supported": sorted(VALID_CAPABILITY_KINDS)},
            )
        self.description = self.description or ""
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCapability":
        """Build a capability from a manifest dictionary."""
        return cls(
            kind=data.get("kind", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            metadata=dict(data.get("metadata") or {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable capability representation."""
        return {
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": dict(self.metadata),
        }


@dataclass
class SkillManifest:
    """Schema-versioned metadata for a LangDeep skill."""

    name: str
    version: str
    description: str = ""
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    capabilities: List[SkillCapability] = field(default_factory=list)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    compatibility: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SKILL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _require_text(self.schema_version, "schema_version")
        if self.schema_version != SKILL_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported skill manifest schema '{self.schema_version}'",
                context={
                    "schema_version": self.schema_version,
                    "supported": SKILL_SCHEMA_VERSION,
                },
            )
        self.name = _require_text(self.name, "name")
        self.version = _require_text(self.version, "version")
        self.description = self.description or ""
        self.owner = self.owner or ""
        self.tags = _normalize_text_list(self.tags, "tags")
        self.required_permissions = _normalize_text_list(
            self.required_permissions,
            "required_permissions",
        )
        self.capabilities = [
            cap if isinstance(cap, SkillCapability) else SkillCapability.from_dict(cap)
            for cap in self.capabilities
        ]
        self.dependencies = dict(self.dependencies or {})
        self.compatibility = dict(self.compatibility or {})
        self.extensions = dict(self.extensions or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillManifest":
        """Build a manifest from a dictionary loaded from JSON or YAML."""
        return cls(
            schema_version=data.get("schema_version", SKILL_SCHEMA_VERSION),
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            tags=list(data.get("tags") or []),
            required_permissions=list(data.get("required_permissions") or []),
            capabilities=list(data.get("capabilities") or []),
            dependencies=dict(data.get("dependencies") or {}),
            compatibility=dict(data.get("compatibility") or {}),
            extensions=dict(data.get("extensions") or {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable manifest representation."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "owner": self.owner,
            "tags": list(self.tags),
            "required_permissions": list(self.required_permissions),
            "capabilities": [cap.as_dict() for cap in self.capabilities],
            "dependencies": dict(self.dependencies),
            "compatibility": dict(self.compatibility),
            "extensions": dict(self.extensions),
        }


@dataclass
class SkillAdapters:
    """Protocol/runtime adapters exposed by an activated skill.

    Values are intentionally plain dictionaries so the skills module does not
    depend on LangChain, MCP, A2A, or server frameworks.
    """

    tools: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, Any] = field(default_factory=dict)
    prompts: Dict[str, Any] = field(default_factory=dict)
    workflows: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    mcp: Dict[str, Any] = field(default_factory=dict)
    a2a: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return adapters grouped by integration surface."""
        return {
            "tools": dict(self.tools),
            "agents": dict(self.agents),
            "prompts": dict(self.prompts),
            "workflows": dict(self.workflows),
            "resources": dict(self.resources),
            "mcp": dict(self.mcp),
            "a2a": dict(self.a2a),
            "metadata": dict(self.metadata),
        }


@dataclass
class SkillContext:
    """Runtime context passed to skill lifecycle hooks."""

    tenant_id: str = "default"
    namespace: str = "default"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    process_id: Optional[str] = None
    policy: Optional[Any] = None
    audit_sink: Optional[Any] = None
    secrets_provider: Optional[Any] = None
    memory_backend: Optional[Any] = None
    cache_backend: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    tracing_adapter: Optional[Any] = None
    runtime_limits: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def audit(self, event_type: str, **payload: Any) -> None:
        """Emit a lifecycle audit event when an audit sink is configured."""
        event = {
            "event_type": event_type,
            "tenant_id": self.tenant_id,
            "namespace": self.namespace,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "process_id": self.process_id,
            **payload,
        }
        sink = self.audit_sink
        if sink is None:
            return
        if hasattr(sink, "record"):
            sink.record(event)
        elif callable(sink):
            sink(event)
        elif hasattr(sink, "append"):
            sink.append(event)
        else:
            raise SkillError(
                "Skill audit sink must expose record(), be callable, or support append()",
                context={"sink_type": type(sink).__name__},
            )


@dataclass
class SkillHealth:
    """Serializable health result for a skill."""

    status: SkillHealthStatus = SkillHealthStatus.UNKNOWN
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = normalize_health_status(self.status)
        self.message = self.message or ""
        self.details = dict(self.details or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillHealth":
        """Build a health result from a dictionary."""
        return cls(
            status=data.get("status", SkillHealthStatus.UNKNOWN),
            message=data.get("message", ""),
            details=dict(data.get("details") or {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable health result."""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": dict(self.details),
        }


class Skill:
    """Base class for reusable LangDeep capabilities."""

    def __init__(self, manifest: SkillManifest):
        self._manifest = manifest
        self._active = False
        self._adapters = SkillAdapters()

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    @property
    def active(self) -> bool:
        return self._active

    def validate(self, context: Optional[SkillContext] = None) -> List[str]:
        """Validate runtime prerequisites before activation.

        Return a list of warnings. Raise ``SkillError`` or
        ``ConfigurationError`` for hard failures.
        """
        return []

    def activate(self, context: Optional[SkillContext] = None) -> SkillAdapters:
        """Activate the skill and return exposed adapters."""
        self._active = True
        return self._adapters

    def deactivate(self, context: Optional[SkillContext] = None) -> None:
        """Deactivate the skill and release runtime state."""
        self._active = False

    def health(self, context: Optional[SkillContext] = None) -> SkillHealth:
        """Return skill health. Override for dependency checks."""
        status = SkillHealthStatus.HEALTHY if self._active else SkillHealthStatus.UNKNOWN
        return SkillHealth(status=status)

    def get_adapters(self) -> SkillAdapters:
        """Return the adapters exposed by the last activation."""
        return self._adapters


SkillFactory = Callable[[], Skill]


def ensure_skill(value: Any, *, name: Optional[str] = None) -> Skill:
    """Validate and return a ``Skill`` instance."""
    if not isinstance(value, Skill):
        raise SkillError(
            "Skill factory returned an invalid object",
            context={"expected": "Skill", "actual": type(value).__name__, "name": name},
        )
    if name and value.manifest.name != name:
        raise ConfigurationError(
            "Skill manifest name does not match registry key",
            context={"registry_name": name, "manifest_name": value.manifest.name},
        )
    return value


def normalize_adapters(value: Optional[Any], skill: Skill) -> SkillAdapters:
    """Normalize lifecycle return values into ``SkillAdapters``."""
    if value is None:
        return skill.get_adapters()
    if isinstance(value, SkillAdapters):
        return value
    if isinstance(value, dict):
        return SkillAdapters(**{key: dict(value.get(key) or {}) for key in SkillAdapters().__dict__})
    raise SkillError(
        "Skill activation must return SkillAdapters, a mapping, or None",
        context={"actual": type(value).__name__, "skill": skill.manifest.name},
    )


def normalize_health(value: Optional[Any]) -> SkillHealth:
    """Normalize health hook return values into ``SkillHealth``."""
    if value is None:
        return SkillHealth()
    if isinstance(value, SkillHealth):
        return value
    if isinstance(value, dict):
        return SkillHealth.from_dict(value)
    if isinstance(value, str):
        return SkillHealth(status=value)
    raise SkillError(
        "Skill health hook must return SkillHealth, a mapping, a status string, or None",
        context={"actual": type(value).__name__},
    )


def normalize_lifecycle_state(value: Any) -> SkillLifecycleState:
    """Normalize lifecycle state aliases and enum values."""
    if isinstance(value, SkillLifecycleState):
        return value
    try:
        return SkillLifecycleState(str(value).strip().lower())
    except ValueError as exc:
        raise ConfigurationError(
            f"Unsupported skill lifecycle state '{value}'",
            context={"supported": [item.value for item in SkillLifecycleState]},
            cause=exc,
        )


def normalize_health_status(value: Any) -> SkillHealthStatus:
    """Normalize health status aliases and enum values."""
    if isinstance(value, SkillHealthStatus):
        return value
    try:
        return SkillHealthStatus(str(value).strip().lower())
    except ValueError as exc:
        raise ConfigurationError(
            f"Unsupported skill health status '{value}'",
            context={"supported": [item.value for item in SkillHealthStatus]},
            cause=exc,
        )


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Skill manifest field '{field_name}' must be a non-empty string")
    return value.strip()


def _normalize_text_list(values: Sequence[Any], field_name: str) -> List[str]:
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                f"Skill manifest field '{field_name}' must contain only non-empty strings",
                context={"value": value},
            )
        result.append(value.strip())
    return result


def capabilities_from_names(kind: str, names: Iterable[str]) -> List[SkillCapability]:
    """Build simple capability descriptors from a kind and names."""
    return [SkillCapability(kind=kind, name=name) for name in names]
