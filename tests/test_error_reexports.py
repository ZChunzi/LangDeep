"""Tests for compatibility error re-export modules."""

from langdeep.core.errors import ProcessError, SecretsError
from langdeep.core.process.errors import ProcessError as ReexportedProcessError
from langdeep.core.secrets.errors import SecretsError as ReexportedSecretsError


def test_process_error_reexport():
    assert ReexportedProcessError is ProcessError


def test_secrets_error_reexport():
    assert ReexportedSecretsError is SecretsError
