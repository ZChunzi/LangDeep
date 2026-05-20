"""Docker-backed sandbox implementation."""
import os
import shutil
import subprocess  # nosec
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from ..logging import get_logger
from .base import BaseSandbox, SandboxResult
from .errors import SandboxError, SandboxTimeoutError

logger = get_logger(__name__)


class DockerSandbox(BaseSandbox):
    """Sandbox that executes code inside a Docker container.

    This backend is opt-in and does not replace ``SubprocessSandbox``. It uses
    the local Docker CLI and daemon to provide a stronger process and filesystem
    boundary than direct subprocess execution, while leaving image hardening,
    daemon policy, resource limits, and host security configuration to the
    deployment environment.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        tmp_dir_base: Optional[str] = None,
        max_artifact_size_mb: int = 10,
        docker_executable: str = "docker",
        network_mode: str = "none",
        remove_container: bool = True,
    ):
        self.image = image
        self._tmp_dir_base = tmp_dir_base
        self._max_artifact_size = max_artifact_size_mb * 1024 * 1024
        self._docker_executable = docker_executable
        self._network_mode = network_mode
        self._remove_container = remove_container

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
        if not code or not code.strip():
            raise ValueError("code must be a non-empty string")
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported: {sorted(self.SUPPORTED_LANGUAGES)}"
            )
        self._ensure_docker_available()

        own_temp = workspace_dir is None
        work_dir = workspace_dir or tempfile.mkdtemp(dir=self._tmp_dir_base)

        try:
            os.makedirs(work_dir, exist_ok=True)
            before_files = self._list_files(work_dir)
            self._write_input_files(work_dir, files or {})

            ext = ".py" if language in ("python", "python3") else ".sh"
            run_file = f"run{ext}"
            code_path = os.path.join(work_dir, run_file)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            cmd = self._build_command(
                work_dir=work_dir,
                run_file=run_file,
                language=language,
                environment=environment or {},
                network_access=network_access,
                extra_args=kwargs.get("docker_args", ()),
            )

            start = time.monotonic()
            try:
                completed = subprocess.run(  # nosec
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                raise SandboxTimeoutError(
                    detail=f"Execution timed out after {timeout}s",
                    context={"timeout": timeout, "language": language, "image": self.image},
                ) from None

            duration_ms = int((time.monotonic() - start) * 1000)
            after_files = self._list_files(work_dir)
            new_files = after_files - before_files - {"run.py", "run.sh", run_file}
            artifacts = self._collect_artifacts(work_dir, new_files)

            return SandboxResult(
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                exit_code=completed.returncode,
                duration_ms=duration_ms,
                artifacts=artifacts,
            )
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            if isinstance(exc, SandboxError):
                raise
            raise SandboxError(
                detail=f"Docker sandbox execution failed: {exc}",
                context={"language": language, "timeout": timeout, "image": self.image},
                cause=exc,
            ) from exc
        finally:
            if own_temp:
                self._cleanup(work_dir)

    def _ensure_docker_available(self) -> None:
        if shutil.which(self._docker_executable) is None:
            raise SandboxError(
                detail=f"Docker executable '{self._docker_executable}' is not available",
                context={"docker_executable": self._docker_executable},
            )

    def _build_command(
        self,
        *,
        work_dir: str,
        run_file: str,
        language: str,
        environment: Dict[str, str],
        network_access: bool,
        extra_args: Iterable[str],
    ) -> list:
        cmd = [self._docker_executable, "run"]
        if self._remove_container:
            cmd.append("--rm")
        if not network_access:
            cmd.extend(["--network", self._network_mode])
        cmd.extend(["-v", f"{os.path.abspath(work_dir)}:/workspace:rw", "-w", "/workspace"])

        for key, value in sorted(environment.items()):
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend(list(extra_args))
        cmd.append(self.image)

        if language in ("python", "python3"):
            cmd.extend(["python", f"/workspace/{run_file}"])
        else:
            cmd.extend(["bash", f"/workspace/{run_file}"])
        return cmd

    @staticmethod
    def _write_input_files(work_dir: str, files: Dict[str, bytes]) -> None:
        base = Path(work_dir).resolve()
        for name, content in files.items():
            target = (base / name).resolve()
            if base not in target.parents and target != base:
                raise ValueError(f"Input file path escapes workspace: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    @staticmethod
    def _list_files(directory: str) -> Set[str]:
        result: Set[str] = set()
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                full = os.path.join(root, fname)
                result.add(os.path.relpath(full, directory))
        return result

    def _collect_artifacts(self, work_dir: str, filenames: Set[str]) -> Dict[str, bytes]:
        artifacts: Dict[str, bytes] = {}
        total = 0
        for name in sorted(filenames):
            path = os.path.join(work_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as f:
                    content = f.read()
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
        try:
            shutil.rmtree(work_dir)
        except Exception:
            logger.warning("Failed to clean up docker sandbox directory", extra={"dir": work_dir})
