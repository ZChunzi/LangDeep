# LangDeep Developer Guide

Version: `2.0.13`

This guide documents the current LangDeep architecture and APIs as implemented in the repository. It is written for framework users, application engineers, and maintainers who need to build, extend, test, or operate LangDeep-based systems.

## 1. Project Scope

LangDeep is a Python framework for building multi-agent workflows on top of LangChain and LangGraph. It favors registration-driven application assembly:

- Decorators register models, providers, tools, agents, memory backends, cache backends, IM channels, and sandbox backends.
- Global registries hold metadata and lazily instantiate runtime objects.
- `FlowOrchestrator` builds a LangGraph workflow from the current registry state.
- Runtime diagnostics and health checks help catch configuration drift before or during service operation.

LangDeep is not a hosted agent platform. It does not provide a production database, a secure remote sandbox, a distributed queue, or a complete observability stack by itself. Those concerns are intentionally exposed through extension points.

## 2. Installation

Install the published package:

```bash
pip install langdeep
```

Install from this repository:

```bash
git clone https://github.com/ZChunzi/LangDeep.git
cd LangDeep/LangDeep
pip install -e .
```

Optional dependency groups:

```bash
pip install -e ".[all]"
pip install -e ".[dev]"
```

The `all` extra installs optional model provider dependencies. Individual provider
groups are available as `openai`, `azure-openai`, `deepseek`, `anthropic`,
`google-genai`, `vertexai`, and `ollama`.
The `dev` extra includes test, coverage, lint, and build tooling used by the repository.

## 3. Runtime Model

The standard lifecycle is:

1. Import application modules.
2. Decorators register components into singleton registries.
3. Optionally run `validate_runtime()`.
4. Create `FlowOrchestrator`.
5. Invoke workflows with `invoke()`, `ainvoke()`, or `astream()`.

The orchestrator compiles a LangGraph graph when it is constructed. Agents registered after `FlowOrchestrator(...)` is created will not become graph nodes in that already-created orchestrator. Register components before constructing the orchestrator.

## 4. Public Package API

The top-level `langdeep` package exports:

- `FlowOrchestrator`
- `ExecutionPolicy`
- `RoutingStrategy`, `KeywordRoutingStrategy`
- `PlanGenerator`, `LLMPlanGenerator`, `FallbackPlanGenerator`
- `TaskRunner`, `RetryTaskRunner`
- `ResultMerger`, `LLMMerger`, `ConcatMerger`
- `model`, `provider`, `register_provider`
- `register_tool`, `regist_tool`, `agent`
- `memory`, `cache`, `im_channel`
- `ModelConfig`
- `Message`, `UserMessage`, `AssistantMessage`, `user_message`,
  `assistant_message`, `message_text`, `last_assistant_text`
- `WorkflowPlan`, `WorkflowTask`, `validate_workflow_plan`
- `HealthChecker`, `MetricsCollector`
- `DiagnosticIssue`, `RuntimeDiagnostics`, `RuntimeValidator`, `validate_runtime`
- `DeepSeekChatModel`, `configure_deepseek_v4`,
  `build_deepseek_payload_messages`
- sandbox, process, secrets, logging, and structured error helpers

Current package version is exposed as:

```python
import langdeep

assert langdeep.__version__ == "2.0.13"
```

## 5. Registries

LangDeep uses singleton registries. They are simple in-process registries, not distributed configuration stores.

| Registry | Module | Responsibility |
|---|---|---|
| `model_registry` | `langdeep.core.registry.model_registry` | Model configs, lazy model instances, optional LLM response cache |
| `provider_registry` | `langdeep.core.registry.model_registry` | Provider factories |
| `agent_registry` | `langdeep.core.registry.agent_registry` | Agent factories, metadata, lazy agent instances |
| `tool_registry` | `langdeep.core.registry.tool_registry` | LangChain tools and tool metadata |
| `memory_registry` | `langdeep.core.memory.registry` | Memory backend factories |
| `cache_registry` | `langdeep.core.cache.registry` | Cache backend factories |
| `im_channel_registry` | `langdeep.core.im.registry` | IM channel handlers and adapters |
| `sandbox_registry` | `langdeep.core.sandbox.registry` | Sandbox backends |

Tests reset registries directly for isolation. Application code should prefer public APIs.

## 6. Models

Register a model with `@model`:

```python
import os
from langdeep import model


@model(
    name="gpt4o",
    provider="openai",
    model_name="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
    max_tokens=2048,
)
def gpt4o():
    pass
```

`ModelConfig` fields:

| Field | Meaning |
|---|---|
| `provider` | Provider registry name |
| `model_name` | Provider-specific model name |
| `base_url` | Optional provider endpoint |
| `api_key` | Optional API key |
| `temperature` | Model temperature |
| `max_tokens` | Optional response token cap |
| `extra_params` | Provider-specific keyword arguments |

Built-in provider names:

- `openai`
- `anthropic`
- `azure_openai`
- `ollama`
- `vertexai`
- `google_genai`
- `deepseek`
- `mock`

Model instances are created lazily by `model_registry.get_model(name)` and cached. Re-registering the same model name clears the cached instance.

Public config snapshot APIs:

```python
from langdeep.core.registry.model_registry import model_registry

config = model_registry.get_config("gpt4o")
configs = model_registry.list_model_configs()
```

Both methods return copies, so callers cannot mutate registry state accidentally.

### DeepSeek v4 Compatibility

The built-in `deepseek` provider returns `DeepSeekChatModel`. It wraps
LangChain's `ChatOpenAI` request and streaming conversion paths so DeepSeek
thinking-mode metadata survives framework boundaries.

DeepSeek v4 models such as `deepseek-v4-pro` and `deepseek-v4-flash` return
`reasoning_content` in thinking mode. For tool-call turns, that field must be
sent back with the assistant message in later requests. `deepseek-reasoner`
uses a different rule and must not receive prior `reasoning_content` in request
history. LangDeep's default `reasoning_content_policy="auto"` selects these
behaviors by model name.

Recommended v4 registration:

```python
import os
from langdeep import configure_deepseek_v4, model


@model(
    name="deepseek_v4",
    provider="deepseek",
    model_name="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    extra_params=configure_deepseek_v4(
        thinking="enabled",
        reasoning_effort="high",
    ),
)
def deepseek_v4():
    pass
```

`@model` accepts provider-specific settings in two equivalent forms:

```python
@model(..., extra_params={"timeout": 30})
@model(..., timeout=30)
```

If both forms are used, direct keyword arguments override values from
`extra_params`. For DeepSeek compatibility, top-level `thinking={...}` is also
accepted and normalized into `extra_body={"thinking": ...}` before constructing
`DeepSeekChatModel`.

For custom providers, reuse `build_deepseek_payload_messages(messages, profile)`
to convert LangChain `BaseMessage` objects into DeepSeek-compatible request
dictionaries. Override `reasoning_content_policy` only when a gateway or model
alias has provider-specific behavior:

- `auto`: choose based on model name and thinking toggle.
- `tool_calls`: replay reasoning only for assistant messages with tool calls.
- `preserve`: always replay reasoning when present.
- `drop`: remove reasoning from request history.

## 7. Providers

Use a provider when a model backend is not covered by the built-ins or when you need to override default provider behavior.

Provider factories receive `ModelConfig` and return a LangChain `BaseChatModel`.

```python
from langchain_core.language_models import BaseChatModel
from langdeep import ModelConfig, provider, register_provider


@provider(name="custom_provider")
def create_custom_provider(config: ModelConfig) -> BaseChatModel:
    ...


def create_other_provider(config: ModelConfig) -> BaseChatModel:
    ...


register_provider("other_provider", create_other_provider)
```

The top-level `register_provider(name, factory)` function requires both arguments. For decorator-style provider registration, use `@provider(name=...)`.

## 8. Tools

`@register_tool` wraps a Python function with LangChain's tool decorator and registers metadata. `@regist_tool` remains available as a backward-compatible alias. The function must have a docstring because LangChain validates tool descriptions before LangDeep stores metadata:

```python
from langdeep import register_tool


@register_tool(
    name="lookup_policy",
    description="Look up an internal policy document.",
    category="knowledge",
    tags=["internal", "policy"],
    requires_confirmation=False,
    timeout=10,
)
def lookup_policy(query: str) -> str:
    """Look up an internal policy document."""
    return f"policy results for {query}"
```

Tool metadata fields:

- `name`
- `description`
- `category`
- `tags`
- `requires_confirmation`
- `timeout`

The executor can place tasks into a `waiting_confirmation` status when a task references tools marked as requiring confirmation.

## 9. Agents

Register agents with `@agent`:

```python
from langdeep import AssistantMessage, UserMessage, agent


@agent(
    name="support_agent",
    description="Handles support requests.",
    capabilities=["support"],
    routing_keywords=["support", "help"],
    model="gpt4o",
    tools=["lookup_policy"],
)
def support_agent():
    class SupportAgent:
        def invoke(self, state):
            question = ""
            for message in reversed(state.get("messages", [])):
                if isinstance(message, UserMessage):
                    question = str(message.content)
                    break
            return {"messages": [AssistantMessage(content=f"Support answer for: {question}")]}

        async def ainvoke(self, state):
            return self.invoke(state)

    return SupportAgent()
```

Agent factories should return a runnable object. LangDeep validates that the created object can be invoked by the orchestration layer.

Agent metadata fields:

| Field | Meaning |
|---|---|
| `name` | Registry name |
| `description` | Human-readable description used by routing prompts |
| `capabilities` | Capability labels |
| `routing_keywords` | Keyword fast-path routing hints |
| `model_name` | Model registry name |
| `tools` | Tool registry names |
| `system_prompt` | Inline system prompt |
| `prompt_path` | Prompt file path |
| `priority` | Routing/execution priority metadata |
| `auto_build` | Use registered agent builder if factory returns `None` |
| `agent_type` | Agent builder registry key |

Auto-built ReAct agent:

```python
@agent(
    name="react_support",
    description="Auto-built support agent.",
    model="gpt4o",
    tools=["lookup_policy"],
    auto_build=True,
    agent_type="react",
)
def react_support():
    pass
```

## 10. FlowOrchestrator

`FlowOrchestrator` builds and executes the runtime graph.

```python
from langdeep import ExecutionPolicy, FlowOrchestrator


policy = ExecutionPolicy(
    strategy="gather",
    max_concurrency=5,
    max_retries=3,
    timeout_seconds=30,
)

orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    execution_policy=policy,
    enable_checkpoint=False,
)
```

Constructor arguments:

| Argument | Purpose |
|---|---|
| `supervisor_model` | Model used by supervisor, planner, and aggregator |
| `max_retries` | Retry count when no explicit `ExecutionPolicy` is supplied |
| `enable_checkpoint` | Enables LangGraph `MemorySaver` by default |
| `prompt_dir` | Optional custom prompt directory |
| `component_dirs` | Directories to auto-import Python components from |
| `llm_timeout` | LLM/task timeout fallback |
| `checkpointer` | Custom LangGraph checkpointer |
| `routing_strategy` | Custom `RoutingStrategy` |
| `workflow_templates_dir` | Directory containing workflow templates |
| `execution_policy` | `ExecutionPolicy` instance |
| `custom_nodes` | Extra graph nodes keyed by node name |
| `plan_generator` | Custom planner implementation |
| `task_runner` | Custom executor implementation |
| `result_merger` | Custom aggregator implementation |
| `memory` | Registered memory backend name |
| `process_manager` | Optional `ProcessManager` |
| `strict_component_import` | Raise on auto-import failure |

Public methods:

```python
from langdeep import user_message

result = orchestrator.invoke("Summarize the incident")
result = await orchestrator.ainvoke("Summarize the incident")
result = orchestrator.chat("Summarize the incident", session_id="cli")
text = orchestrator.chat_text("Summarize the incident", session_id="cli")
result = orchestrator.invoke_messages([user_message("Summarize the incident")])
result = orchestrator.invoke_state({"messages": [user_message("Summarize the incident")]})

async for chunk in orchestrator.astream("Summarize the incident"):
    print(chunk)

health = orchestrator.health()
metrics = orchestrator.get_metrics()
graph = orchestrator.graph
```

There is no public `run()` method in v2.0.0. Use `invoke()`.

`invoke()` is intentionally permissive for framework interoperability:

- `str`: normal request text.
- `BaseMessage`: one LangChain message.
- `Sequence[BaseMessage]`: explicit message history managed by the caller.
- `{"messages": [...]}`: LangGraph-style state input.
- `{"input": "..."}`, `{"user_input": "..."}`, or `{"content": "..."}`:
  dictionary wrappers from generated code or HTTP adapters.

For multi-turn applications, prefer `chat(user_input, session_id=...)` with a
registered memory backend. For advanced graph integration, prefer
`invoke_state(...)` so the call site states that it is passing graph state.

## 11. Graph Architecture

The graph is built around these nodes:

- `supervisor`
- `planner`
- `executor`
- `aggregator`
- one node per registered agent
- optional custom nodes

Execution paths:

1. Direct route: `START -> supervisor -> agent -> aggregator -> END`
2. Planned route: `START -> supervisor -> planner -> executor -> aggregator -> END`
3. Custom route: `START -> supervisor -> custom_node -> aggregator -> END`

The supervisor first tries `KeywordRoutingStrategy`. If no keyword route matches, it calls the supervisor model with a `route_to_node` tool and parses the selected target.

## 12. State Shape

The orchestrator graph state includes:

| Field | Meaning |
|---|---|
| `messages` | LangChain messages accumulated by graph nodes |
| `next` | Supervisor routing target |
| `current_task` | Current workflow task ID |
| `task_context` | Shared arbitrary context |
| `agent_results` | Task/agent results keyed by ID |
| `workflow_plan` | List of workflow task dictionaries |
| `error_count` | Error counter |
| `max_retries` | Retry limit |
| `aggregation_done` | Aggregation guard |

## 13. Routing

Default routing is two-tier:

1. `KeywordRoutingStrategy` checks registered agent `routing_keywords`.
2. `DefaultRouter` asks the supervisor model to call `route_to_node(next_node=...)`.

Custom strategy:

```python
from typing import Any, Dict, List, Optional
from langdeep import RoutingStrategy


class AlwaysPlanner(RoutingStrategy):
    def route(self, user_input: str, available_agents: List[Dict[str, Any]]) -> Optional[str]:
        return "planner"
```

Use it:

```python
orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    routing_strategy=AlwaysPlanner(),
)
```

A strategy may return:

- a registered agent name
- `"planner"`
- a registered custom node name
- `None` to fall through to LLM routing

## 14. Workflow Plans

Workflow tasks are dictionaries with at least:

- `id`
- `agent`
- `depends_on`
- `status`

Example:

```python
plan = [
    {"id": "collect", "agent": "research_agent", "depends_on": [], "status": "pending"},
    {"id": "write", "agent": "writer_agent", "depends_on": ["collect"], "status": "pending"},
]

result = orchestrator.invoke("Prepare a market brief", workflow_plan=plan)
```

Validate before execution:

```python
from langdeep import validate_workflow_plan


validate_workflow_plan(
    plan,
    available_agents=["research_agent", "writer_agent"],
    available_tools=[],
)
```

`validate_workflow_plan()` catches duplicate task IDs, missing dependencies, circular dependencies, unknown agents, and unknown tools when corresponding registries are supplied.

## 15. Execution Policy

`ExecutionPolicy` controls workflow task execution:

```python
from langdeep import ExecutionPolicy


policy = ExecutionPolicy(
    max_concurrency=3,
    strategy="priority_queue",
    retry_on=["TimeoutError"],
    max_retries=3,
    retry_backoff="exponential",
    timeout_seconds=30,
    fail_fast=False,
)
```

Supported strategies:

- `gather`: execute ready tasks concurrently up to `max_concurrency`
- `sequential`: execute one ready task at a time
- `priority_queue`: execute higher priority ready tasks first

Supported retry backoffs:

- `exponential`
- `fixed`

Policy objects can be serialized:

```python
data = policy.to_dict()
policy = ExecutionPolicy.from_dict(data)
```

## 16. Extension Points

LangDeep exposes abstract extension points:

- `RoutingStrategy`
- `PlanGenerator`
- `TaskRunner`
- `ResultMerger`

Custom result merger:

```python
import json
from typing import Dict

from langdeep import ResultMerger


class JsonMerger(ResultMerger):
    def merge(self, user_request: str, agent_results: Dict[str, str]) -> str:
        return json.dumps(
            {
                "request": user_request,
                "results": agent_results,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
```

Inject it:

```python
orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    result_merger=JsonMerger(),
)
```

Aggregator input and output contract:

- `agent_results` may contain plain final-answer strings, LangChain message
  shaped dictionaries, or executor-style dictionaries such as
  `{"success": True, "data": "...", "error": ""}`.
- Failed, skipped, or `waiting_confirmation` executor results are excluded from
  synthesis and recorded as failed aggregation inputs.
- `ResultMerger.merge(user_request, agent_results)` receives only successful
  final-answer text as `Dict[str, str]`.
- Aggregator returns `{"messages": [AssistantMessage(...)], "aggregation_done": True}`.

## 17. Prompts

Prompt templates are Markdown files loaded by `MarkdownPromptLoader`.

Built-in prompt package:

- `langdeep.resources.prompts.supervisor`
- `langdeep.resources.prompts.planner`
- `langdeep.resources.prompts.dynamic_planner`
- `langdeep.resources.prompts.aggregator`
- `langdeep.resources.prompts.customer_service`
- `langdeep.resources.prompts.web_research_system`

Use `prompt_dir` to override built-in prompts:

```python
orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    prompt_dir="./prompts",
)
```

Prompt files use front matter with variables:

```markdown
---
name: planner
version: 1.0
variables: [user_request, available_agents]
---

Plan this request:

{user_request}
```

## 18. Memory

Register memory backends with `@memory`:

```python
from langdeep import memory


@memory(name="session_memory", description="In-memory session storage")
def session_memory():
    pass
```

If the factory returns `None`, the decorator creates an `InMemoryBackend`.

Custom backends should implement `BaseMemoryBackend`:

- `store_entry(session_id, entry)`
- `store_messages(session_id, messages)`
- `load_messages(session_id)`
- `list_sessions()`
- `delete_session(session_id)`
- `clear()`
- `close()`

## 19. Cache

Register cache backends with `@cache`:

```python
from langdeep import cache


@cache(name="llm_cache", ttl=300, max_entries=1024)
def llm_cache():
    pass
```

If the factory returns `None`, the decorator creates a `MemoryCache`.

Custom backends should implement `BaseCacheBackend`:

- `get(key)`
- `set(key, value, ttl=None)`
- `delete(key)`
- `has(key)`
- `clear()`

LLM response caching is disabled by default. Enable it explicitly:

```python
from langdeep.core.registry.model_registry import model_registry


model_registry.enable_response_cache(ttl=300, max_entries=1024)
```

Then invoke through:

```python
response = model_registry.invoke_with_cache("gpt4o", messages)
```

## 20. IM Integration

The IM subsystem provides:

- `@im_channel`
- `IMChannelRegistry`
- `WebhookReceiver`
- `IMMessage`
- `PlatformType`

It is designed as an adapter layer. Production deployments should verify platform signature validation, retry behavior, and message permission policy for their own IM provider.

## 21. Sandbox

Sandbox support includes:

- `BaseSandbox`
- `SubprocessSandbox`
- `sandbox_registry`
- `@sandbox`

The built-in subprocess sandbox is suitable for trusted or semi-trusted local execution tasks. It is not a complete security boundary for hostile code. Enterprise deployments should use OS/container isolation, resource quotas, network policy, filesystem policy, and audit logging around sandbox usage.

## 22. Process Management

Process support includes:

- `ProcessManager`
- `ProcessState`
- `Process`
- `SuspendSignal`
- process signals

Use it when workflows need lifecycle tracking, pause/resume semantics, or long-running process state.

## 23. Secrets

Secrets support includes:

- `SecretsManager`
- `SecretsProvider`
- `EnvSecretsProvider`

Example:

```python
from langdeep import EnvSecretsProvider, secrets_manager


secrets_manager.add_provider(EnvSecretsProvider())
api_key = secrets_manager.get("OPENAI_API_KEY")
```

Do not hard-code API keys in decorators or documentation examples for production usage. Read them from a secrets provider or environment-specific configuration.

## 24. Observability

Health checks:

```python
from langdeep import HealthChecker


status = HealthChecker(version="2.0.13").check_all()
print(status.status)
print(status.checks)
```

`HealthChecker` checks:

- registered model backends
- registered memory backends
- registered cache backends
- agent registry consistency
- tool registry consistency

Metrics:

```python
from langdeep import MetricsCollector


metrics = MetricsCollector()
metrics.counter("requests")
metrics.gauge("workers", 4)
metrics.histogram("latency_ms", 120.0)
snapshot = metrics.get_metrics()
```

## 25. Runtime Diagnostics

Diagnostics are startup/preflight checks. They are stricter than health checks because they validate registry wiring before traffic starts.

```python
from langdeep import validate_runtime


diagnostics = validate_runtime(instantiate_agents=True)
diagnostics.raise_for_errors()
```

Checks include:

- model registry names are non-empty
- model provider names are registered
- model names are non-empty
- model `temperature` is numeric and usually within `[0, 2]`
- `max_tokens` is positive when supplied
- agent metadata names match registry keys
- agent model references exist
- agent tool references exist
- optional agent instantiation succeeds
- tool metadata names match registry keys
- tools have useful descriptions

Serialize diagnostics:

```python
payload = diagnostics.to_dict()
```

Issue shape:

```python
{
    "severity": "error",
    "component": "agent",
    "name": "support_agent",
    "message": "agent references tools that are not registered",
    "context": {"missing_tools": ["lookup_policy"]},
}
```

Recommended enterprise bootstrap:

```python
def boot():
    import my_app.models
    import my_app.tools
    import my_app.agents

    validate_runtime(instantiate_agents=True).raise_for_errors()
    return FlowOrchestrator(
        supervisor_model="gpt4o",
        enable_checkpoint=True,
        strict_component_import=True,
    )
```

## 26. Error Handling

All framework exceptions derive from `LangDeepError`.

Important error groups:

- `ConfigurationError`
- `ModelError`
- `ModelNotFoundError`
- `ProviderNotFoundError`
- `ProviderImportError`
- `AgentError`
- `AgentBuildError`
- `AgentNotFoundError`
- `ToolError`
- `ToolNotFoundError`
- `ExecutionError`
- `OrchestrationError`

Errors provide a structured `to_dict()` method:

```python
try:
    validate_runtime().raise_for_errors()
except Exception as exc:
    if hasattr(exc, "to_dict"):
        print(exc.to_dict())
    raise
```

## 27. Logging And Trace Context

Logging helpers:

- `get_logger(name)`
- `set_trace_context(...)`
- `get_trace_id()`

`FlowOrchestrator.invoke()` creates a trace context and clears it after completion. Logs use structured `extra` fields extensively.

## 28. Testing

Run full pytest suite with coverage:

```bash
python -m pytest --cov --cov-report=term-missing --cov-report=xml
```

Run the repository test runner:

```bash
python tests/run_all.py
```

Run lint:

```bash
python -m ruff check src tests
```

Run compile check:

```bash
python -m compileall -q src tests
```

Build package artifacts:

```bash
python -m build --no-isolation
```

Current coverage gate is configured in `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 90
```

## 29. Release Checklist

Before tagging a release:

1. Update `pyproject.toml` version.
2. Update `src/__init__.py` `__version__`.
3. Update README and docs version references.
4. Run `python -m ruff check src tests`.
5. Run `python -m pytest --cov --cov-report=term-missing --cov-report=xml`.
6. Run `python tests/run_all.py`.
7. Run `python -m compileall -q src tests`.
8. Run `python -m build --no-isolation`.
9. Commit the release changes.
10. Create and push the tag.

## 30. Enterprise Deployment Guidance

Recommended minimum production controls:

- Run `validate_runtime(instantiate_agents=True)` during service startup.
- Use `strict_component_import=True` in orchestrator construction.
- Load API keys from `SecretsManager` or a platform secret store.
- Use explicit `ExecutionPolicy` values for concurrency, retries, and timeout.
- Validate workflow plans before execution when accepting externally supplied plans.
- Avoid running untrusted code in `SubprocessSandbox` without additional isolation.
- Add request-level audit logging around user input, selected route, workflow plan, tool usage, and final status.
- Export `HealthChecker` results to your service health endpoint.
- Export `MetricsCollector` snapshots or wrap metrics with your standard telemetry system.
- Keep provider SDK versions pinned in application deployments.

## 31. Known Boundaries

- Registries are in-process singletons; they are not distributed registries.
- `ainvoke()` is async-compatible but currently delegates to synchronous `invoke()`.
- Built-in cache and memory backends are process-local unless replaced.
- Built-in subprocess sandbox is not sufficient for hostile code isolation.
- Built-in metrics are in-process snapshots, not a replacement for Prometheus/OpenTelemetry.
- Provider SDK imports happen when corresponding model instances are created.

## 32. Minimal Smoke Test

Use this as a no-network smoke test for a fresh checkout:

```python
from langdeep import AssistantMessage, FlowOrchestrator, agent, last_assistant_text, model, validate_runtime


@model(name="mock_chat", provider="mock", model_name="mock-chat")
def mock_chat():
    pass


@agent(name="echo", description="Echo agent", routing_keywords=["echo"], model="mock_chat")
def echo():
    class Echo:
        def invoke(self, state):
            return {"messages": [AssistantMessage(content="ok")]}

        async def ainvoke(self, state):
            return self.invoke(state)

    return Echo()


validate_runtime(instantiate_agents=True).raise_for_errors()
result = FlowOrchestrator(supervisor_model="mock_chat", enable_checkpoint=False).invoke("echo")
assert last_assistant_text(result) == "ok"
```
