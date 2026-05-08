"""Unit tests for the sandbox module: SubprocessSandbox, SandboxRegistry, @sandbox."""

import os
import tempfile

from langdeep.core.sandbox import (
    BaseSandbox,
    SandboxResult,
    SubprocessSandbox,
    SandboxRegistry,
    sandbox_registry,
    sandbox,
    SandboxError,
    SandboxTimeoutError,
    SandboxImportError,
)
from langdeep.core.errors import ConfigurationError


def setup_function():
    """Reset singleton registries before each test."""
    from tests.conftest import clean_registries
    clean_registries()


# ── BaseSandbox ─────────────────────────────────────────────────────────────


def test_base_sandbox_is_abstract():
    """BaseSandbox cannot be instantiated directly."""
    try:
        BaseSandbox()  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# ── SubprocessSandbox: basic execution ──────────────────────────────────────


def test_subprocess_print():
    """Basic Python print statement."""
    sb = SubprocessSandbox()
    result = sb.run("print(42)")
    assert result.stdout.strip() == "42"
    assert result.exit_code == 0


def test_subprocess_shell_echo():
    """Shell echo command."""
    sb = SubprocessSandbox()
    result = sb.run("echo hello", language="shell")
    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0


def test_subprocess_stderr():
    """Stderr is captured separately."""
    sb = SubprocessSandbox()
    result = sb.run("import sys; print('err', file=sys.stderr)", language="python")
    assert result.stderr.strip() == "err"


def test_subprocess_shell_exit():
    """Shell exit code is reported."""
    sb = SubprocessSandbox()
    result = sb.run("exit 42", language="shell")
    assert result.exit_code == 42


# ── SubprocessSandbox: timeout ──────────────────────────────────────────────


def test_subprocess_timeout():
    """Execution that exceeds timeout raises SandboxTimeoutError."""
    sb = SubprocessSandbox()
    try:
        sb.run("import time; time.sleep(10)", timeout=1)
        assert False, "Should have raised SandboxTimeoutError"
    except SandboxTimeoutError:
        pass


# ── SubprocessSandbox: import restrictions ─────────────────────────────────


def test_subprocess_restricted_import():
    """Importing a disallowed module raises SandboxImportError."""
    sb = SubprocessSandbox()
    try:
        sb.run("import flask")
        assert False, "Should have raised SandboxImportError"
    except SandboxImportError as exc:
        assert "flask" in exc.detail


def test_subprocess_allowed_import():
    """Importing an allowed module succeeds."""
    sb = SubprocessSandbox()
    result = sb.run("import json; print(json.dumps({'a': 1}))")
    assert result.stdout.strip() == '{"a": 1}'


def test_subprocess_from_import_restricted():
    """from-import of a disallowed module raises SandboxImportError."""
    sb = SubprocessSandbox()
    try:
        sb.run("from django.http import JsonResponse")
        assert False, "Should have raised SandboxImportError"
    except SandboxImportError:
        pass


# ── SubprocessSandbox: environment variables ────────────────────────────────


def test_subprocess_environment():
    """Environment variables are passed to the subprocess."""
    sb = SubprocessSandbox()
    result = sb.run(
        "import os; print(os.environ.get('MY_VAR'))",
        environment={"MY_VAR": "hello"},
    )
    assert result.stdout.strip() == "hello"


# ── SubprocessSandbox: workspace / data exchange ────────────────────────────


def test_subprocess_artifacts():
    """Files created during execution are collected as artifacts."""
    sb = SubprocessSandbox()
    result = sb.run("open('output.txt', 'w').write('data')")
    assert "output.txt" in result.artifacts
    assert result.artifacts["output.txt"] == b"data"


def test_subprocess_input_files():
    """Input files are available to the sandbox."""
    sb = SubprocessSandbox()
    result = sb.run(
        "print(open('input.txt').read())",
        files={"input.txt": b"input_data"},
    )
    assert result.stdout.strip() == "input_data"


def test_subprocess_workspace_data_exchange():
    """Two sandbox calls sharing a workspace_dir can exchange data."""
    sb = SubprocessSandbox()
    with tempfile.TemporaryDirectory() as d:
        r1 = sb.run('open("shared.txt", "w").write("hello")', workspace_dir=d)
        assert "shared.txt" in r1.artifacts

        r2 = sb.run("print(open('shared.txt').read())", workspace_dir=d)
        assert r2.stdout.strip() == "hello"


# ── SubprocessSandbox: empty code validation ────────────────────────────────


def test_subprocess_empty_code_raises():
    """Empty code string raises ValueError."""
    sb = SubprocessSandbox()
    try:
        sb.run("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_subprocess_unsupported_language_raises():
    """Unsupported language raises ValueError."""
    sb = SubprocessSandbox()
    try:
        sb.run("print(1)", language="ruby")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_subprocess_network_access_rejected():
    """SubprocessSandbox rejects network_access instead of pretending to enforce it."""
    sb = SubprocessSandbox()
    try:
        sb.run("print(1)", network_access=True)
        assert False, "Should have raised SandboxError"
    except SandboxError as exc:
        assert "network" in exc.detail.lower()


# ── SandboxRegistry ─────────────────────────────────────────────────────────


def test_registry_has_builtin():
    """Registry comes pre-loaded with 'subprocess' backend."""
    backends = sandbox_registry.list_backends()
    assert "subprocess" in backends


def test_registry_get_backend():
    """get_backend returns a BaseSandbox instance."""
    sb = sandbox_registry.get_backend("subprocess")
    assert isinstance(sb, BaseSandbox)


def test_registry_get_nonexistent_raises():
    """get_backend for an unregistered name raises ConfigurationError."""
    try:
        sandbox_registry.get_backend("nonexistent_sandbox")
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError:
        pass


def test_registry_register_and_get():
    """Register a custom sandbox and retrieve it."""

    class SimpleSandbox(BaseSandbox):
        def run(self, code, **kwargs):
            return SandboxResult(stdout="ok")

    sandbox_registry.register("simple", lambda: SimpleSandbox(), description="Simple")
    backend = sandbox_registry.get_backend("simple")
    result = backend.run("anything")
    assert result.stdout == "ok"


def test_registry_remove():
    """Remove a registered backend."""
    class TempSandbox(BaseSandbox):
        def run(self, code, **kwargs):
            return SandboxResult()

    sandbox_registry.register("temp", lambda: TempSandbox())
    sandbox_registry.remove("temp")
    assert "temp" not in sandbox_registry.list_backends()


def test_registry_clear():
    """Clear resets to only the built-in backend."""
    class TempSandbox(BaseSandbox):
        def run(self, code, **kwargs):
            return SandboxResult()

    sandbox_registry.register("temp", lambda: TempSandbox())
    sandbox_registry.clear()
    assert sandbox_registry.list_backends() == ["subprocess"]


def test_registry_duplicate_overwrites():
    """Registering the same name twice overwrites the previous entry."""
    class First(BaseSandbox):
        def run(self, code, **kwargs):
            return SandboxResult(stdout="first")

    class Second(BaseSandbox):
        def run(self, code, **kwargs):
            return SandboxResult(stdout="second")

    sandbox_registry.register("dup", lambda: First())
    sandbox_registry.register("dup", lambda: Second())
    result = sandbox_registry.get_backend("dup").run("")
    assert result.stdout == "second"


# ── @sandbox decorator ──────────────────────────────────────────────────────


def test_sandbox_decorator_builtin_factory():
    """@sandbox with no factory value (None) creates SubprocessSandbox."""

    @sandbox(name="auto_sub", description="Auto subprocess")
    def factory():
        return None

    sb = sandbox_registry.get_backend("auto_sub")
    assert isinstance(sb, BaseSandbox)
    result = sb.run("print(99)")
    assert result.stdout.strip() == "99"


def test_sandbox_decorator_custom_backend():
    """@sandbox with a custom factory registers the custom backend."""

    class Custom(BaseSandbox):
        def run(self, code, **kwargs):
            return SandboxResult(stdout="custom")

    @sandbox(name="custom_backend", description="Custom")
    def custom_factory():
        return Custom()

    sb = sandbox_registry.get_backend("custom_backend")
    result = sb.run("")
    assert result.stdout == "custom"


def test_sandbox_decorator_name_defaults_to_function_name():
    """@sandbox without name uses the factory function name."""

    class AutoName(BaseSandbox):
        def run(self, code, **kwargs):
            return SandboxResult(stdout="auto")

    @sandbox(description="Auto name")
    def my_factory():
        return AutoName()

    assert "my_factory" in sandbox_registry.list_backends()


def test_sandbox_decorator_validates_return_type():
    """@sandbox raises TypeError if factory returns non-BaseSandbox."""

    try:
        @sandbox(name="bad_return")
        def bad_factory():
            return "not_a_sandbox"  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
