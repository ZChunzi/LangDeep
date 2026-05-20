# Deployment

This guide summarizes production integration checks.

## Startup

1. Import all modules that register models, providers, tools, agents, memory,
   cache, IM channels, and sandbox backends.
2. Run `validate_runtime(instantiate_agents=True)`.
3. Run `langdeep health` or `HealthChecker().check_all()`.
4. Fail startup on diagnostics errors.

## HTTP Integration

Use an application framework such as FastAPI around `FlowOrchestrator.chat()` or
`chat_text()`. Keep request validation, authentication, rate limiting, and audit
logging at the service boundary.

## CI Gates

```bash
python -m ruff check src tests
python -m pytest --cov=src --cov-report=term-missing
python -m compileall -q src tests
python -m build --no-isolation
```
