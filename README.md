# LangDeep

LangDeep is a decorator-driven multi-agent workflow framework built on LangChain and LangGraph. It provides model, provider, tool, agent, memory, cache, IM, sandbox, process, scheduling, observability, and runtime diagnostic primitives for building controlled agent systems.

**Current version:** `2.0.0`

**Project status:** Beta. The core API is stable enough for controlled internal pilots. Production usage should still include organization-specific reviews for credentials, model providers, audit logging, sandbox policy, persistence, and operations.

## What v2.0.0 Provides

- Decorator-based registration for models, providers, tools, agents, memory, cache, IM channels, and sandboxes.
- `FlowOrchestrator` for supervisor routing, optional planning, task execution, and aggregation.
- Two-tier routing: keyword fast path first, LLM tool-call fallback second.
- Workflow execution policies for `gather`, `sequential`, and `priority_queue` strategies.
- Structured workflow schema validation through `WorkflowPlan`, `WorkflowTask`, and `validate_workflow_plan`.
- Runtime preflight diagnostics through `validate_runtime()` and `RuntimeValidator`.
- Health checks and in-process metrics through `HealthChecker` and `MetricsCollector`.
- Built-in mock model provider for local tests and examples.
- Optional provider integrations for OpenAI-compatible, Anthropic, Azure OpenAI, Ollama, Vertex AI, Google GenAI, and DeepSeek providers.

## Installation

Install the package:

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
pip install -e ".[persist]"
pip install -e ".[dev]"
```

## Quick Start

This example uses the built-in `mock` provider, so it does not require any external API key.

```python
from langchain_core.messages import AIMessage, HumanMessage

from langdeep import FlowOrchestrator, agent, model, regist_tool, validate_runtime


@model(name="mock_chat", provider="mock", model_name="mock-chat")
def mock_chat():
    pass


@regist_tool(name="get_weather", description="Return a mocked weather report.")
def get_weather(city: str) -> str:
    """Return a mocked weather report."""
    return f"{city}: sunny, 25C"


@agent(
    name="weather_agent",
    description="Answers simple weather questions.",
    routing_keywords=["weather", "天气"],
    model="mock_chat",
    tools=["get_weather"],
)
def weather_agent():
    class WeatherAgent:
        def invoke(self, state):
            question = ""
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    question = str(message.content)
                    break
            answer = f"Question: {question}\n{get_weather('Beijing')}"
            return {"messages": [AIMessage(content=answer)]}

        async def ainvoke(self, state):
            return self.invoke(state)

    return WeatherAgent()


validate_runtime(instantiate_agents=True).raise_for_errors()

orchestrator = FlowOrchestrator(
    supervisor_model="mock_chat",
    enable_checkpoint=False,
)

result = orchestrator.invoke("北京天气怎么样？")

for message in reversed(result["messages"]):
    if isinstance(message, AIMessage) and message.content:
        print(message.content)
        break
```

## Core API

### Models

Register a model with `@model`. The `name` is the LangDeep registry name. The `provider` must match a registered provider.

```python
import os
from langdeep import model


@model(
    name="gpt4o",
    provider="openai",
    model_name="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
)
def register_gpt4o():
    pass
```

Built-in provider names include:

- `openai`
- `anthropic`
- `azure_openai`
- `ollama`
- `vertexai`
- `google_genai`
- `deepseek`
- `mock`

### Providers

Use `@provider` or `register_provider(name, factory)` when you need to add or override a model provider. Provider factories receive a `ModelConfig` and must return a `BaseChatModel`.

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

### Tools

`@regist_tool` wraps a Python function as a LangChain tool and registers metadata for filtering and diagnostics. The wrapped function must have a docstring because LangChain validates tool descriptions before LangDeep stores metadata.

```python
from langdeep import regist_tool


@regist_tool(
    name="search_docs",
    description="Search internal documentation.",
    category="knowledge",
    tags=["internal", "docs"],
)
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"results for {query}"
```

### Agents

Agent factories registered by `@agent` should return an object with at least `invoke(state)`. For async execution compatibility, also provide `ainvoke(state)`.

```python
from langchain_core.messages import AIMessage
from langdeep import agent


@agent(
    name="support_agent",
    description="Handles support questions.",
    capabilities=["support"],
    routing_keywords=["support", "help"],
    model="gpt4o",
    tools=["search_docs"],
)
def support_agent():
    class SupportAgent:
        def invoke(self, state):
            return {"messages": [AIMessage(content="Support response")]}

        async def ainvoke(self, state):
            return self.invoke(state)

    return SupportAgent()
```

For automatic LangGraph ReAct agent construction, set `auto_build=True` and make sure the referenced model and tools are registered:

```python
@agent(
    name="react_support",
    description="Auto-built ReAct support agent.",
    model="gpt4o",
    tools=["search_docs"],
    auto_build=True,
)
def react_support():
    pass
```

## FlowOrchestrator

`FlowOrchestrator` is the main runtime entry point.

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

result = orchestrator.invoke("Analyze this request")
```

Public methods:

- `invoke(user_input, context=None, workflow_plan=None, template_name=None)`: synchronous workflow execution.
- `ainvoke(user_input, context=None, workflow_plan=None, template_name=None)`: async-compatible wrapper.
- `astream(user_input, context=None, **kwargs)`: async generator over graph stream chunks.
- `health()`: returns aggregated health information.
- `get_metrics()`: returns in-process metrics from `MetricsCollector`.
- `graph`: exposes the compiled LangGraph graph.

Constructor options include:

- `supervisor_model`: model registry name used by supervisor, planner, and aggregator.
- `max_retries`: retry count used when no explicit `ExecutionPolicy` is supplied.
- `enable_checkpoint`: enables LangGraph `MemorySaver` checkpointing by default.
- `prompt_dir`: optional custom prompt directory.
- `component_dirs`: Python module directories to auto-import.
- `routing_strategy`: custom `RoutingStrategy`.
- `workflow_templates_dir`: directory for YAML/JSON workflow templates.
- `execution_policy`: `ExecutionPolicy` instance.
- `custom_nodes`: custom LangGraph node callables.
- `plan_generator`, `task_runner`, `result_merger`: extension points.
- `memory`: registered memory backend name.
- `process_manager`: optional `ProcessManager`.
- `strict_component_import`: fail startup on component import errors.

## Workflow Plans

You can pass an explicit workflow plan to bypass LLM planning.

```python
plan = [
    {"id": "collect", "agent": "research_agent", "depends_on": [], "status": "pending"},
    {"id": "write", "agent": "writer_agent", "depends_on": ["collect"], "status": "pending"},
]

result = orchestrator.invoke("Prepare a market brief", workflow_plan=plan)
```

Validate workflow plans before execution:

```python
from langdeep import validate_workflow_plan


validate_workflow_plan(
    plan,
    available_agents=["research_agent", "writer_agent"],
    available_tools=[],
)
```

## Runtime Diagnostics

`validate_runtime()` is designed for enterprise startup checks. It validates static registry wiring before the service accepts traffic.

```python
from langdeep import validate_runtime


diagnostics = validate_runtime(instantiate_agents=True)
diagnostics.raise_for_errors()
```

It checks:

- Model registry names and model names are non-empty.
- Model providers are registered.
- Model `temperature` and `max_tokens` are sane.
- Agent metadata names match registry keys.
- Agent model references exist.
- Agent tool references exist.
- Optional agent instantiation succeeds.
- Tool metadata is internally consistent.

The result is serializable:

```python
{
    "ok": True,
    "error_count": 0,
    "warning_count": 0,
    "issues": [],
}
```

## Health And Metrics

```python
health = orchestrator.health()
metrics = orchestrator.get_metrics()
```

`HealthChecker` probes registered model, memory, and cache backends and reports agent/tool registry consistency. `MetricsCollector` provides counters, gauges, histograms, and snapshot retrieval.

## Memory And Cache

Register memory backends:

```python
from langdeep import memory


@memory(name="session_memory", description="In-process session memory")
def session_memory():
    pass
```

Register cache backends:

```python
from langdeep import cache


@cache(name="llm_cache", ttl=300, max_entries=1024)
def llm_cache():
    pass
```

If the decorated factory returns `None`, LangDeep uses built-in in-memory implementations.

LLM response caching is opt-in through the model registry:

```python
from langdeep.core.registry.model_registry import model_registry


model_registry.enable_response_cache(ttl=300, max_entries=1024)
```

## Sandbox, Process, And Secrets

LangDeep includes supporting infrastructure for production-oriented workflows:

- `SubprocessSandbox` and `@sandbox` for controlled local command execution. The built-in subprocess sandbox is not a security boundary for untrusted code.
- `ProcessManager` and `ProcessState` for long-running workflow lifecycle tracking.
- `SecretsManager` and `EnvSecretsProvider` for layered environment-variable secret resolution.

## Testing

Run the standard test suite:

```bash
python -m pytest --cov --cov-report=term-missing --cov-report=xml
```

Run the project test runner:

```bash
python tests/run_all.py
```

Run lint and build checks:

```bash
python -m ruff check src tests
python -m compileall -q src tests
python -m build --no-isolation
```

## Version Notes

`2.0.0` adds runtime diagnostics, stronger health checks, public model config snapshot APIs, updated documentation, and a Beta package classifier. It remains source-compatible with the v1.2 decorator and orchestrator patterns, but users should run `validate_runtime()` in CI or service bootstrap to catch configuration drift early.

## License

LangDeep is released under the MIT license.
