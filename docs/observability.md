# Observability

LangDeep provides lightweight runtime inspection tools.

## Health

```python
from langdeep import HealthChecker


status = HealthChecker(version="2.0.14").check_all()
print(status.status)
```

## Diagnostics

```python
from langdeep import validate_runtime


diagnostics = validate_runtime(instantiate_agents=True)
diagnostics.raise_for_errors()
```

## CLI

```bash
langdeep health
langdeep diagnostics --instantiate-agents
langdeep list
```
