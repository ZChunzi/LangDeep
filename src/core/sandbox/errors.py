"""Sandbox-specific error types."""
from ..errors import SandboxError

__all__ = ["SandboxError", "SandboxTimeoutError", "SandboxImportError"]


class SandboxTimeoutError(SandboxError):
    """Code execution exceeded the allowed time limit."""
    code = "SANDBOX_TIMEOUT"


class SandboxImportError(SandboxError):
    """Code attempted to import a restricted module."""
    code = "SANDBOX_IMPORT_ERROR"
