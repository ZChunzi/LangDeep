"""Audit payload redaction utilities."""

from typing import Any, Dict, Iterable

DEFAULT_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}
REDACTED = "[REDACTED]"


def redact_audit_payload(
    payload: Any,
    sensitive_keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS,
) -> Any:
    """Return a copy of *payload* with sensitive fields redacted."""
    keys = {key.lower() for key in sensitive_keys}
    return _redact(payload, keys)


def _redact(value: Any, sensitive_keys: set) -> Any:
    if isinstance(value, dict):
        result: Dict[Any, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key), sensitive_keys):
                result[key] = REDACTED
            else:
                result[key] = _redact(item, sensitive_keys)
        return result
    if isinstance(value, list):
        return [_redact(item, sensitive_keys) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, sensitive_keys) for item in value)
    return value


def _is_sensitive_key(key: str, sensitive_keys: set) -> bool:
    normalized = key.lower().replace("-", "_")
    parts = set(normalized.split("_"))
    return normalized in sensitive_keys or bool(parts & sensitive_keys)
