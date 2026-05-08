"""Built-in subprocess-based sandbox implementation."""
import ast
import os
import shutil
import subprocess  # nosec
import tempfile
import time
from typing import Any, Dict, List, Optional, Set

from ..logging import get_logger
from .base import BaseSandbox, SandboxResult
from .errors import SandboxImportError, SandboxTimeoutError, SandboxError

logger = get_logger(__name__)

# Default set of allowed imports for Python sandbox execution.
# Imports not in this set are rejected by the static AST checker.
_DEFAULT_ALLOWED_IMPORTS: Set[str] = {
    "json", "math", "random", "statistics", "datetime",
    "re", "collections", "itertools", "functools", "typing",
    "uuid", "copy", "enum", "decimal", "fractions",
    "hashlib", "base64", "textwrap", "string",
    "pathlib", "os", "sys", "time",
}

_MAX_ARTIFACT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class SubprocessSandbox(BaseSandbox):
    """Sandbox that executes code in a subprocess with resource limits.

    This backend is a local execution helper, not a hardened security boundary
    for untrusted code. Use a container, VM, or dedicated remote sandbox backend
    when executing untrusted user input.

    Features:
        - Subprocess isolation with configurable timeout
        - AST-based import whitelist checking
        - Memory limit via ``resource.setrlimit`` (Linux)
        - Workspace directory for cross-sandbox data exchange
        - Automatic artifact collection and cleanup
    """

    def __init__(
        self,
        tmp_dir_base: Optional[str] = None,
        max_artifact_size_mb: int = 10,
        max_memory_mb: int = 256,
    ):
        self._tmp_dir_base = tmp_dir_base
        self._max_artifact_size = max_artifact_size_mb * 1024 * 1024
        self._max_memory_mb = max_memory_mb

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
        # ── Input validation ──────────────────────────────────────────
        if not code or not code.strip():
            raise ValueError("code must be a non-empty string")
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported: {sorted(self.SUPPORTED_LANGUAGES)}"
            )
        if network_access:
            raise SandboxError(
                detail=(
                    "SubprocessSandbox does not support network isolation or "
                    "network access control. Use a container or VM sandbox backend."
                ),
                context={"network_access": network_access},
            )

        # ── Static import check (Python only) ─────────────────────────
        if language in ("python", "python3"):
            allowed = kwargs.get("allowed_imports", _DEFAULT_ALLOWED_IMPORTS)
            self._check_imports(code, allowed)

        # ── Prepare working directory ─────────────────────────────────
        own_temp = workspace_dir is None
        work_dir = workspace_dir or tempfile.mkdtemp(dir=self._tmp_dir_base)
        before_files: Set[str] = set()

        try:
            if own_temp:
                os.makedirs(work_dir, exist_ok=True)

            # Snapshot existing files for artifact detection
            before_files = self._list_files(work_dir)

            # Write input files
            if files:
                for name, content in files.items():
                    path = os.path.join(work_dir, name)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(content)

            # Write code file
            ext = ".py" if language in ("python", "python3") else ".sh"
            code_path = os.path.join(work_dir, f"run{ext}")
            with open(code_path, "w") as f:
                f.write(code)

            # ── Build subprocess command ──────────────────────────────
            if language in ("python", "python3"):
                cmd = ["python3", code_path]
            else:
                cmd = ["bash", code_path]

            env = os.environ.copy()
            if environment:
                env.update(environment)
            if workspace_dir:
                env["WORKSPACE_DIR"] = workspace_dir

            memory_limit_bytes = self._max_memory_mb * 1024 * 1024

            # ── Execute ───────────────────────────────────────────────
            start = time.monotonic()
            try:
                completed = subprocess.run(  # nosec
                    cmd,
                    cwd=work_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    preexec_fn=_build_preexec(memory_limit_bytes) if _PREEVEC_AVAILABLE else None,
                )
            except subprocess.TimeoutExpired:
                raise SandboxTimeoutError(
                    detail=f"Execution timed out after {timeout}s",
                    context={"timeout": timeout, "language": language},
                ) from None

            duration_ms = int((time.monotonic() - start) * 1000)

            # ── Collect artifacts ─────────────────────────────────────
            after_files = self._list_files(work_dir)
            new_files = after_files - before_files - {f"run{ext}", "run.py", "run.sh"}
            artifacts = self._collect_artifacts(work_dir, new_files)

            return SandboxResult(
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                exit_code=completed.returncode,
                duration_ms=duration_ms,
                artifacts=artifacts,
            )

        except (SandboxTimeoutError, SandboxImportError):
            raise
        except Exception as exc:
            raise SandboxError(
                detail=f"Sandbox execution failed: {exc}",
                context={"language": language, "timeout": timeout},
                cause=exc,
            ) from exc
        finally:
            if own_temp:
                self._cleanup(work_dir)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _check_imports(code: str, allowed: Set[str]) -> None:
        """Inspect AST for imports; reject any not in the *allowed* set."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Let the subprocess surface syntax errors naturally
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in allowed:
                        raise SandboxImportError(
                            detail=f"Import of '{alias.name}' is not allowed",
                            context={"module": alias.name},
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in allowed:
                    raise SandboxImportError(
                        detail=f"Import from '{node.module}' is not allowed",
                        context={"module": node.module},
                    )

    @staticmethod
    def _list_files(directory: str) -> Set[str]:
        """Recursively list relative paths under *directory*."""
        result: Set[str] = set()
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                full = os.path.join(root, fname)
                result.add(os.path.relpath(full, directory))
        return result

    def _collect_artifacts(
        self, work_dir: str, filenames: Set[str]
    ) -> Dict[str, bytes]:
        """Read new files created during execution, respecting size limits."""
        artifacts: Dict[str, bytes] = {}
        total = 0
        for name in sorted(filenames):
            path = os.path.join(work_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                content = open(path, "rb").read()
            except OSError:
                continue
            total += len(content)
            if total > self._max_artifact_size:
                logger.warning(
                    "Artifact collection exceeds size limit, truncating",
                    extra={"limit_mb": self._max_artifact_size / (1024 * 1024)},
                )
                break
            artifacts[name] = content
        return artifacts

    @staticmethod
    def _cleanup(work_dir: str) -> None:
        """Remove the temporary working directory."""
        try:
            shutil.rmtree(work_dir)
        except Exception:
            logger.warning("Failed to clean up sandbox directory", extra={"dir": work_dir})


def _build_preexec(memory_limit_bytes: int):
    """Return a preexec function that sets RLIMIT_AS (Linux only)."""
    def _preexec() -> None:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        except (ImportError, ValueError, resource.error):
            pass
    return _preexec


#: Whether ``resource.setrlimit`` is available (Linux).
_PREEVEC_AVAILABLE: bool = False
try:
    import resource  # noqa: F401
    _PREEVEC_AVAILABLE = True
except ImportError:
    pass
