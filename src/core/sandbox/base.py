"""Abstract base class and result model for sandbox backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SandboxResult:
    """Result from executing code in a sandbox.

    Attributes:
        stdout: Standard output text.
        stderr: Standard error text.
        exit_code: Process exit code (0 for success).
        duration_ms: Execution time in milliseconds.
        artifacts: Files created during execution, keyed by filename.
    """
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    artifacts: Dict[str, bytes] = field(default_factory=dict)


class BaseSandbox(ABC):
    """Abstract sandbox backend for isolated code execution.

    Subclasses must implement :meth:`run` and provide the execution
    isolation mechanism (subprocess, container, etc.).
    """

    SUPPORTED_LANGUAGES = {"python", "shell", "python3"}

    @abstractmethod
    def run(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        environment: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, bytes]] = None,
        workspace_dir: Optional[str] = None,
        network_access: bool = False,
        **kwargs: Any,
    ) -> SandboxResult:
        """Execute *code* in an isolated environment.

        Args:
            code: Source code to execute.
            language: Language identifier (``"python"``, ``"shell"``).
            timeout: Maximum execution time in seconds.
            environment: Environment variables for the execution.
            files: Input files to make available, keyed by filename.
            workspace_dir: Shared directory for cross-sandbox data exchange.
            network_access: Whether network access is allowed. Backends must either
                enforce this setting or reject unsupported values.
            **kwargs: Subclass-specific extensions.

        Returns:
            A :class:`SandboxResult` with captured output and artifacts.

        Raises:
            SandboxTimeoutError: Execution exceeded *timeout*.
            SandboxImportError: Code attempted a disallowed import.
            SandboxError: Other execution failures.
        """
        ...
