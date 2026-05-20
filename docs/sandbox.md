# Sandbox

Sandbox backends execute code behind the `BaseSandbox` interface.

## Built-In Backend

`SubprocessSandbox` runs code in a local subprocess. It is suitable for trusted
or semi-trusted local tasks, but it is not a complete security boundary for
hostile code.

## Usage

```python
from langdeep.core.sandbox import SubprocessSandbox


result = SubprocessSandbox().run("print(42)")
print(result.stdout)
```

For untrusted code, use container, VM, network, filesystem, and resource
isolation outside the subprocess backend.
