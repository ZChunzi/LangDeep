# Security

Security-sensitive LangDeep areas include tools, sandbox execution, secrets,
provider credentials, webhooks, workflow plans, and persistent memory/cache
backends.

## Tool Policy

- Require confirmation for mutating tools.
- Configure workspace roots for file tools.
- Record audit logs for user-triggered tool calls.

## Sandbox

`SubprocessSandbox` is not a full security boundary. Use container or VM
isolation for untrusted code.

## Secrets

Do not hard-code provider API keys in model decorators or examples. Prefer
environment variables or `SecretsManager`.

See the repository-level [Security Policy](../SECURITY.md).
