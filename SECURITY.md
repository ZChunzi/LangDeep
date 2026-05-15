# Security Policy

## Supported Versions

Security fixes are handled on the latest released version. Older versions may
receive fixes only when maintainers have capacity and the change is low risk.

## Reporting a Vulnerability

Do not report security vulnerabilities through public GitHub issues.

Report privately using the maintainer contact in `pyproject.toml`. Include:

- Affected version or commit SHA.
- Minimal reproduction steps.
- Impact assessment.
- Whether the issue is already public.
- Any suggested mitigation.

Maintainers will acknowledge valid reports as soon as practical and coordinate
fixes before public disclosure.

## Security-Sensitive Areas

Extra care is required for changes involving:

- Sandbox execution.
- Tool execution policies and workspace boundaries.
- Secrets handling.
- Webhook and IM integrations.
- Provider authentication and request routing.
- Workflow plans accepted from external users.
- File-system and subprocess access.

## Sandbox Boundary

`SubprocessSandbox` is a convenience isolation layer for trusted or
semi-trusted local tasks. It is not a complete boundary for hostile code. Use
container, VM, or platform sandboxing for untrusted workloads.

## Dependency Security

Keep provider SDKs and LangChain/LangGraph dependencies current in application
deployments. Avoid adding new runtime dependencies unless they are necessary and
actively maintained.
