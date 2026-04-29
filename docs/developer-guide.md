# LangDeep Framework — Developer Guide

**Version 0.3.0** | **License MIT**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Getting Started](#2-getting-started)
3. [Architecture](#3-architecture)
4. [Models & Providers](#4-models--providers)
5. [Agents](#5-agents)
6. [Tools](#6-tools)
7. [Routing](#7-routing)
8. [Workflows & Planning](#8-workflows--planning)
9. [Execution Policies](#9-execution-policies)
10. [Prompts](#10-prompts)
11. [Checkpointing & State](#11-checkpointing--state)
12. [Task Scheduler](#12-task-scheduler)
13. [Error Handling](#13-error-handling)
14. [Logging & Tracing](#14-logging--tracing)
15. [API Reference](#15-api-reference)
16. [Extension Points](#16-extension-points)
17. [Best Practices](#17-best-practices)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Overview

LangDeep is a decorator-driven multi-agent orchestration framework built on LangGraph. It provides a supervisor-pattern workflow engine that decomposes user requests, routes them to specialized agents, executes tasks with dependency-aware parallelism, and aggregates results into coherent responses.

**Design philosophy:**

- **Registration, not configuration.** Use Python decorators to register models, agents, tools, and providers. No YAML config files required for basic usage.
- **Pluggable everything.** Inject custom routing strategies, plan generators, task runners, result mergers, and execution policies.
- **Convention over boilerplate.** Drop Python files into `models/`, `agents/`, `tools/` directories — they are auto-discovered and registered.

**Key capabilities:**

| Capability | Mechanism |
|---|---|
| Multi-model support | `@model` + `@provider` decorators with lazy instantiation |
| Agent routing | Two-tier: keyword fast-path → LLM tool-call fallback |
| Task decomposition | LLM-driven planner with structured JSON output |
| Parallel execution | DAG-based dependency resolution with semaphore-bounded concurrency |
| Result synthesis | LLM-based merger with conflict resolution |
| Workflow templates | YAML/JSON declarative workflow definitions |
| Scheduled execution | Cron/interval/condition-based task scheduler |
| State persistence | LangGraph checkpointing (MemorySaver default, PostgresSaver injectable) |

---

## 2. Getting Started

### 2.1 Installation

```bash
pip install langdeep
```

For optional provider SDKs:

```bash
pip install "langdeep[all]"      # anthropic, google-genai, ollama, tavily
pip install "langdeep[persist]"  # SQLAlchemy for PostgresSaver
```

### 2.2 Minimal Example

Create a directory with three subdirectories:

```
my_project/
├── agent_test.py       # entry point
├── models/
│   └── my_model.py     # model registration
├── agents/
│   └── my_agent.py     # agent registration
└── tools/
    └── my_tools.py     # tool registration
```

**models/my_model.py**

```python
import os
from langdeep import model

@model(
    name="gpt4o",
    provider="openai",
    model_name="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)
def register_gpt4o():
    pass
```

**agents/my_agent.py**

```python
from langchain_core.messages import AIMessage
from langdeep import agent

@agent(
    name="assistant",
    description="General-purpose assistant",
    capabilities=["conversation", "qa"],
    routing_keywords=["help", "question", "what", "how"],
    model="gpt4o",
)
def assistant_factory():
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    llm = ChatOpenAI(model="gpt-4o")
    return create_react_agent(llm, tools=[])
```

**agent_test.py**

```python
import asyncio
from langchain_core.messages import AIMessage
from langdeep import FlowOrchestrator

orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    enable_checkpoint=False,
)

async def ask(question: str):
    result = await orchestrator.ainvoke(question)
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content)
            break

asyncio.run(ask("Hello!"))
```

Run it:

```bash
export OPENAI_API_KEY="sk-..."
python agent_test.py -q "Hello!"
```

### 2.3 CLI Arguments

The example `agent_test.py` shipped with the project supports:

```
python agent_test.py -q "question"     # single question, non-streaming
python agent_test.py -q "question" -s  # single question, streaming
python agent_test.py                   # interactive mode
python agent_test.py -s                # interactive streaming mode
```

---

## 3. Architecture

### 3.1 Graph Topology

The orchestrator compiles a LangGraph `StateGraph` with this structure:

```
┌─────────┐
│  START   │
└────┬─────┘
     │
     ▼
┌──────────┐    keyword match    ┌──────────────┐    ┌─────┐
│supervisor├────────────────────►│  agent_node   ├───►│     │
└────┬─────┘                     └──────────────┘    │     │
     │                                                │     │
     │ LLM fallback                                   │     │
     ▼                                                │  A  │
┌─────────┐     ┌──────────┐     ┌────────────┐      │  G  │
│ planner ├────►│ executor ├────►│ aggregator  ├─────►│  G  │
└─────────┘     └──────────┘     └────────────┘      │  R  │
                                                      │  E  │
┌──────────────┐                                      │  G  │
│ custom_node  ├─────────────────────────────────────►│  A  │
└──────────────┘                                      │  T  │
                                                      │  O  │
                                                      │  R  │
                                                      └──┬──┘
                                                         │
                                                         ▼
                                                      ┌─────┐
                                                      │ END │
                                                      └─────┘
```

**Two execution paths:**

1. **Direct route** — Supervisor matches an agent (keyword or LLM) → agent runs → aggregator → END
2. **Plan-then-execute** — Supervisor routes to planner → planner generates task list → executor runs tasks in dependency order → aggregator synthesizes → END

### 3.2 State Schema

Every node reads and writes a shared `TypedDict` state:

| Field | Type | Reducer | Purpose |
|---|---|---|---|
| `messages` | `Sequence[BaseMessage]` | `add_messages` | Accumulates conversation across all nodes |
| `next` | `str` | replace | Supervisor's routing decision |
| `current_task` | `str` | replace | Currently executing task ID |
| `task_context` | `Dict[str, Any]` | replace | Arbitrary shared context across steps |
| `agent_results` | `Dict[str, Any]` | `{**x, **y}` merge | Keyed results from each agent |
| `workflow_plan` | `List[Dict]` | replace | Multi-step task plan |
| `error_count` | `int` | replace | Retry counter |
| `max_retries` | `int` | replace | Retry limit |
| `aggregation_done` | `bool` | replace | Guards against double aggregation |

### 3.3 Initialization Sequence

When you create `FlowOrchestrator(...)`, it executes this sequence:

1. Creates `MarkdownPromptLoader` (built-in + optional external prompt dir)
2. Initializes checkpointer (`MemorySaver` by default)
3. **Auto-imports** all `.py` files from `component_dirs` (default: `models/`, `tools/`, `agents/`)
4. Builds `DefaultRouter` with routing strategy
5. Builds `Planner` with plan generator
6. Builds `Executor` with task runner and execution policy
7. Builds `Aggregator` with result merger
8. For each registered agent, creates a graph node via `make_agent_node()`
9. Compiles `StateGraph` with checkpointer

### 3.4 Component Lifecycle

```
import time ──────────────────────────────────────────►
  @model / @agent / @regist_tool / @provider evaluate
  → singleton registries populated
  → FlowOrchestrator.__init__ builds graph
     → get_model() / get_agent() called lazily at runtime
     → instances cached in registry
```

---

## 4. Models & Providers

### 4.1 Model Registration

Use the `@model` decorator on any function or class:

```python
from langdeep import model

@model(
    name="my_model",           # Registry key (defaults to function name)
    provider="openai",         # Must match a registered provider
    model_name="gpt-4o",       # Actual API model name
    base_url=None,             # Override API base URL
    api_key=None,              # Override API key
    temperature=0.7,
    max_tokens=None,
    **extra_params,            # Passed through to the provider factory
)
def register_my_model():
    pass
```

**How `extra_params` works:**

Any keyword argument not explicitly declared in the `@model` signature is captured into `ModelConfig.extra_params` and passed to the provider factory as `**extra`. The provider factory should pop known extra keys before forwarding to the LLM constructor.

Example with custom timeout:

```python
@model(
    name="gpt4o_fast",
    provider="openai",
    model_name="gpt-4o",
    request_timeout=30,        # Ends up in extra_params
)
def register_gpt4o_fast():
    pass
```

### 4.2 Built-in Providers

Eight providers are registered automatically:

| Provider | Factory | Required Packages |
|---|---|---|
| `openai` | `ChatOpenAI` | `langchain-openai` |
| `anthropic` | `ChatAnthropic` | `langchain-anthropic` |
| `azure_openai` | `AzureChatOpenAI` | `langchain-openai` |
| `ollama` | `ChatOllama` | `langchain-ollama` |
| `vertexai` | `ChatVertexAI` | `langchain-google-vertexai` |
| `google_genai` | `ChatGoogleGenerativeAI` | `langchain-google-genai` |
| `deepseek` | `ChatOpenAI` (base_url=https://api.deepseek.com) | `langchain-openai` |
| `mock` | `MockLLM` (in-process) | none |

The built-in provider for each key is registered in `ProviderRegistry._register_builtin_providers()`.

### 4.3 Custom Providers

Use the `@provider` decorator to register a factory function:

```python
from langdeep import provider
from langdeep.core.registry.model_registry import ModelConfig
from langchain_core.language_models import BaseChatModel

@provider(name="deepseek")
def create_deepseek_model(config: ModelConfig) -> BaseChatModel:
    import os
    from langchain_openai import ChatOpenAI

    extra = dict(config.extra_params or {})
    request_timeout = extra.pop("request_timeout", 90)

    api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY")

    return ChatOpenAI(
        model=config.model_name,
        base_url=config.base_url or "https://api.deepseek.com/v1",
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        request_timeout=request_timeout,
        **extra,
    )
```

**Important:** Registering a provider with the same name as a built-in one **overrides** it. This is how you customize behavior for a provider without touching framework code.

**Predefined shortcut decorators** are also available:

```python
from langdeep.core.decorators.provider import openai_provider, anthropic_provider

@openai_provider   # equivalent to @provider(name="openai")
def my_openai_factory(config):
    ...
```

### 4.4 ModelConfig Reference

```python
@dataclass
class ModelConfig:
    provider: str                     # Provider name
    model_name: str                   # Actual API model identifier
    base_url: Optional[str] = None    # API endpoint
    api_key: Optional[str] = None     # Auth key
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = {} # Catch-all for extra kwargs
```

### 4.5 Multiple Models with One Provider

Register as many models as you want under the same provider:

```python
@model(name="deepseek_chat", provider="deepseek", model_name="deepseek-chat", temperature=0.7)
def _(): pass

@model(name="deepseek_v4", provider="deepseek", model_name="deepseek-v4-flash", temperature=0.3)
def _(): pass
```

---

## 5. Agents

### 5.1 Agent Registration

Use the `@agent` decorator:

```python
from langdeep import agent

@agent(
    name="web_researcher",                      # Registry key
    description="Searches the web and summarizes findings",
    capabilities=["web_search", "summarization"],
    routing_keywords=["search", "find", "lookup", "research"],
    model="gpt4o",                              # Default model for this agent
    tools=["web_search", "read_url"],           # Tool names
    system_prompt=None,                         # Custom system prompt
    priority=1,                                 # Execution priority (higher = first)
)
def web_researcher_factory():
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langdeep.core.registry.tool_registry import tool_registry

    llm = ChatOpenAI(model="gpt-4o")
    tools = [
        tool_registry.get_tool("web_search"),
        tool_registry.get_tool("read_url"),
    ]
    return create_react_agent(llm, tools=tools)
```

### 5.2 AgentMetadata Reference

```python
@dataclass
class AgentMetadata:
    name: str
    description: str
    capabilities: List[str] = []          # Tags for capability-based lookup
    routing_keywords: List[str] = []      # Triggers for keyword routing
    model_name: str = "default"
    tools: List[str] = []                 # Tool names this agent uses
    system_prompt: Optional[str] = None
    priority: int = 1                     # Higher = executed first by priority_queue
```

### 5.3 Agent Contract

Agents are **duck-typed**. The factory must return an object that accepts a dict and returns a dict:

```python
# Input
{
    "messages": [HumanMessage(content="...")],
    "task_context": {"key": "value", ...}
}

# Output
{
    "messages": [AIMessage(content="response text")]
}
```

This means agents can be:

- **LangGraph `create_react_agent`** (ReAct agent with tool loop)
- **Custom LangGraph `StateGraph`** (multi-step sub-agent)
- **LangChain `Runnable` chains**
- **Plain callable objects** implementing `invoke(input: dict) -> dict`

### 5.4 Agent Instance Lifecycle

Agents are **lazily instantiated singletons** — the factory is called once on first `get_agent()`, and the result is cached. This means agent state persists across invocations within the same process.

### 5.5 Capability-Based Lookup

```python
from langdeep.core.registry.agent_registry import agent_registry

# Find all agents with a specific capability
research_agents = agent_registry.get_agents_by_capability("web_search")
for name in research_agents:
    print(name)
```

---

## 6. Tools

### 6.1 Tool Registration

Use the `@regist_tool` decorator:

```python
from langdeep import regist_tool

@regist_tool(
    name="web_search",
    description="Search the web for information",
    category="search",
    tags=["web", "external"],
    requires_confirmation=False,
    timeout=30,
)
def web_search(query: str) -> str:
    """Search the web and return results."""
    # ... implementation
    return "results..."
```

### 6.2 ToolMetadata Reference

```python
@dataclass
class ToolMetadata:
    name: str
    description: str
    category: str = "general"
    tags: List[str] = []
    requires_confirmation: bool = False
    timeout: Optional[int] = None
```

### 6.3 Filtered Tool Retrieval

```python
from langdeep.core.registry.tool_registry import tool_registry

# Get specific tools by name
tools = tool_registry.get_tools(names=["web_search", "read_file"])

# Get by category
search_tools = tool_registry.get_tools(category="search")

# Get by tags (ALL tags must match)
external_tools = tool_registry.get_tools(tags=["external", "api"])

# Combine filters (AND logic)
tools = tool_registry.get_tools(
    names=["web_search", "image_search", "other"],
    category="search",
    tags=["verified"]
)
```

---

## 7. Routing

### 7.1 Two-Tier Routing

The `DefaultRouter` uses a two-tier strategy:

**Tier 1 — Keyword fast-path (no LLM call):**

```python
class KeywordRoutingStrategy(RoutingStrategy):
    def route(self, user_input: str, available_agents: List[Dict]) -> Optional[str]:
        lower = user_input.lower()
        for agent_info in available_agents:
            keywords = agent_registry.get_metadata(agent_info["name"]).routing_keywords
            if any(kw.lower() in lower for kw in keywords):
                return agent_info["name"]
        return None  # Fall through to LLM
```

**Tier 2 — LLM tool-call routing:**

When no keyword matches, the router:
1. Builds a system message listing all agents and valid targets
2. Binds a `route_to_node` tool to the supervisor model
3. Calls the LLM, expecting a tool call with `{"next_node": "agent_name"}`
4. Falls back to parsing text content if no tool call is returned
5. Defaults to `"end"` if parsing fails

### 7.2 Custom Routing Strategy

Implement the `RoutingStrategy` ABC:

```python
from langdeep import RoutingStrategy
from typing import List, Dict, Optional

class ContextAwareRouter(RoutingStrategy):
    def route(self, user_input: str, available_agents: List[Dict]) -> Optional[str]:
        # Your custom logic
        if "urgent" in user_input.lower():
            return "priority_agent"
        return None  # Fall through to LLM

orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    routing_strategy=ContextAwareRouter(),
)
```

### 7.3 LLM Routing Prompt

The router sends this to the LLM:

```
Call route_to_node to select the next step.
Agents:
- agent_1: description
- agent_2: description
Valid targets: planner, end, agent_1, agent_2
Rule: simple tasks → best agent; complex multi-step tasks → planner.
```

You can customize the supervisor's behavior by loading a custom prompt. Create a `supervisor.md` file and pass `prompt_dir` to the orchestrator.

---

## 8. Workflows & Planning

### 8.1 Dynamic Planning (LLM-Generated)

When the supervisor routes to `"planner"`, the `Planner` node generates a task plan using the supervisor model:

```json
{
  "tasks": [
    {
      "id": "task_1",
      "name": "Search for recent news",
      "agent": "web_researcher",
      "tools": ["web_search"],
      "depends_on": [],
      "parallel": true,
      "status": "pending"
    },
    {
      "id": "task_2",
      "name": "Analyze findings",
      "agent": "analyst",
      "tools": [],
      "depends_on": ["task_1"],
      "parallel": false,
      "status": "pending"
    }
  ]
}
```

The prompt template is at `resources/prompts/planner.md`. It instructs the LLM to:
- Decompose complex requests into smaller tasks
- Assign tasks to appropriate agents
- Identify inter-task dependencies
- Mark parallel-safe tasks

### 8.2 Fallback Plan

If the LLM fails to generate a plan (network error, bad JSON), `FallbackPlanGenerator` produces a single-task plan using the first registered agent:

```python
class FallbackPlanGenerator(PlanGenerator):
    def generate(self, user_request, available_agent_names):
        return [{
            "id": "fallback_task",
            "name": "Process user request",
            "agent": available_agent_names[0] if available_agent_names else "default_agent",
            "depends_on": [],
            "parallel": False,
            "status": "pending",
        }]
```

### 8.3 Declarative Workflow Templates

Define static workflows in YAML or JSON:

```yaml
# workflows/competitor_analysis.yaml
id: competitor_analysis
name: "Competitor Analysis Pipeline"
description: "Search and analyze competitors"
steps:
  - id: task_search
    type: agent
    name: web_research_agent
    tools:
      - tavily_web_search
    config:
      query: "{{ user_input }}"
    priority: 1

  - id: task_analyze
    type: agent
    name: default_agent
    depends_on:
      - task_search
    config:
      instruction: "Analyze the search results and produce a summary"
    priority: 2
```

Load and use a template:

```python
result = await orchestrator.ainvoke(
    "Analyze our competitors",
    template_name="competitor_analysis",
)
```

The `TemplateLoader` replaces `{{ user_input }}` in the template with the actual user input.

### 8.4 WorkflowPlanner (Declarative API)

For programmatic workflow definition:

```python
from langdeep import WorkflowPlanner, WorkflowNode, NodeType

nodes = [
    WorkflowNode(
        id="search",
        type=NodeType.AGENT,
        name="web_researcher",
        priority=1,
    ),
    WorkflowNode(
        id="analyze",
        type=NodeType.AGENT,
        name="analyst",
        depends_on=["search"],
        priority=2,
    ),
]

planner = WorkflowPlanner(workflow_dir="workflows")
plan_dicts = planner.to_plan_dicts(nodes, user_input="Analyze competitors")
# Pass plan_dicts as workflow_plan to orchestrator.ainvoke()
```

**NodeType enum:**

| Value | Description |
|---|---|
| `AGENT` | Route to a registered agent |
| `TOOL` | Call a registered tool directly |
| `CONDITION` | Conditional branching |
| `PARALLEL` | Parallel execution group |
| `LOOP` | Iterative loop |
| `HUMAN` | Human-in-the-loop pause (declarative only) |
| `CUSTOM` | Custom node handler |

### 8.5 Dependency Resolution

The executor resolves dependencies automatically:

1. Tasks with no `depends_on` (or empty) run first
2. After each batch completes, check which remaining tasks have all dependencies satisfied
3. Repeat until all tasks complete or no progress is possible
4. Tasks with unmet dependencies after `len(pending) + 1` rounds are skipped with an error

```
Round 1: [task_1, task_2]  ← no dependencies, run in parallel
Round 2: [task_3]           ← depends on task_2
Round 3: [task_4, task_5]   ← depend on task_3
```

---

## 9. Execution Policies

### 9.1 Policy Configuration

```python
from langdeep import ExecutionPolicy

policy = ExecutionPolicy(
    max_concurrency=5,        # Max parallel tasks (≥ 1)
    strategy="gather",        # "gather" | "sequential" | "priority_queue"
    retry_on=[],              # Future: error types to retry
)
```

### 9.2 Strategies

| Strategy | Behavior |
|---|---|
| `gather` | Runs ready tasks concurrently via `asyncio.gather` with a `Semaphore(max_concurrency)` |
| `sequential` | Runs ready tasks one at a time in a simple loop |
| `priority_queue` | Sorts ready tasks by `priority` (descending), then runs concurrently |

### 9.3 From JSON File

```json
// execution_policy.json
{
  "max_concurrency": 3,
  "strategy": "priority_queue"
}
```

```python
policy = ExecutionPolicy.from_file("execution_policy.json")
```

### 9.4 Per-Task Configuration

In plan dicts or YAML templates:

```python
{
    "id": "critical_task",
    "agent": "high_priority_agent",
    "priority": 10,         # Higher = earlier in priority_queue strategy
    "depends_on": [...],
}
```

---

## 10. Prompts

### 10.1 Prompt File Format

Prompts are Markdown files with YAML frontmatter:

```markdown
---
name: planner
version: 1.0
description: "Plan generation prompt"
model: gpt4o
variables: [user_request, available_agents]
---

# System
You are a planning assistant. Available agents: {available_agents}

# Human
Please create a plan for: {user_request}
```

**Frontmatter fields:**

| Field | Required | Purpose |
|---|---|---|
| `name` | Yes | Unique prompt identifier |
| `version` | Yes | Version string for cache invalidation |
| `description` | No | Human-readable description |
| `model` | No | Suggested model for this prompt |
| `variables` | No | Expected interpolation variables (for documentation) |

**Section headers** map to message roles:

| Header | Maps to |
|---|---|
| `# System` | `SystemMessagePromptTemplate` |
| `# Human` or `# User` | `HumanMessagePromptTemplate` |
| `# AI` or `# Assistant` | `AIMessagePromptTemplate` |

### 10.2 Loading Prompts

```python
from langdeep.core.prompt.prompt_loader import MarkdownPromptLoader

loader = MarkdownPromptLoader(prompt_dir="custom_prompts")

# Load a specific prompt
template = loader.load_prompt("planner")
# Returns a ChatPromptTemplate with {variable} placeholders

# Use it with an LLM
messages = template.format_messages(
    user_request="Analyze competitors",
    available_agents="- agent1\n- agent2",
)
```

### 10.3 Custom Prompt Directory

```python
orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    prompt_dir="my_prompts",  # Override built-in prompts
)
```

Place files like `planner.md`, `aggregator.md`, `supervisor.md` in `my_prompts/` to override the built-in versions. If a prompt is not found in the custom directory, the built-in one is used.

### 10.4 Prompt Reloading

```python
loader.reload("planner")   # Reload a specific prompt
loader.reload()            # Reload all cached prompts
```

---

## 11. Checkpointing & State

### 11.1 Memory Checkpointer (Default)

By default, the orchestrator uses `MemorySaver()` — an in-memory checkpointer that persists state across graph nodes within a single invocation but loses state on process restart:

```python
orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    enable_checkpoint=True,  # default
)
```

### 11.2 Disabling Checkpointing

For stateless, single-turn use cases:

```python
orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    enable_checkpoint=False,
)
```

### 11.3 External Checkpointers

Inject any LangGraph-compatible checkpointer (requires `langdeep[persist]`):

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
await checkpointer.setup()

orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    checkpointer=checkpointer,
)
```

### 11.4 State Fields Deep Dive

**`messages` — Accumulated conversation**

Uses LangGraph's `add_messages` reducer. Messages are appended, not replaced. This means all node outputs contribute to the growing conversation history.

**`agent_results` — Keyed results merge**

Uses a custom reducer `lambda x, y: {**x, **y}`. Each agent's output is stored under its name and merged into one dict:

```python
# After agent_1 runs:
{"agent_results": {"agent_1": "result A"}}

# After agent_2 runs:
{"agent_results": {"agent_1": "result A", "agent_2": "result B"}}
```

**`task_context` — Shared mutable context**

Pass arbitrary data between tasks:

```python
result = await orchestrator.ainvoke(
    "Process order #123",
    context={"order_id": "123", "user_tier": "premium"},
)
```

Accessible inside agents via `task_context`.

### 11.5 Pre-Loading a Workflow Plan

Skip dynamic planning by providing a pre-built plan:

```python
prebuilt_plan = [
    {"id": "step1", "agent": "fetcher", "depends_on": [], "priority": 1},
    {"id": "step2", "agent": "processor", "depends_on": ["step1"], "priority": 2},
]

result = await orchestrator.ainvoke(
    "Run the pipeline",
    workflow_plan=prebuilt_plan,
)
```

---

## 12. Task Scheduler

### 12.1 Overview

The `TaskScheduler` runs orchestrator invocations on a schedule. It runs in a daemon thread and supports five trigger types.

### 12.2 Configuration

```python
from langdeep.core.scheduling import TaskScheduler, ScheduledTask, TriggerType

scheduler = TaskScheduler(orchestrator)

# Cron-based: every weekday at 9 AM
scheduler.register_task(ScheduledTask(
    id="daily_report",
    name="Daily Report Generation",
    trigger_type=TriggerType.CRON,
    trigger_config={"cron": "0 9 * * 1-5"},
    workflow="daily_report",
    params={"user_input": "Generate today's report"},
    timeout=600,
    retry_count=3,
    retry_delay=60,
))

# Interval-based: every 30 minutes
scheduler.register_task(ScheduledTask(
    id="health_check",
    name="System Health Check",
    trigger_type=TriggerType.INTERVAL,
    trigger_config={"interval_seconds": 1800},
    workflow="health_check",
))

# Condition-based: run when a condition is met
def inventory_alert_check(ctx):
    return ctx.variables.get("inventory_low", False)

scheduler.register_condition_checker("inventory_alert", inventory_alert_check)

scheduler.register_task(ScheduledTask(
    id="inventory_alert",
    name="Low Inventory Alert",
    trigger_type=TriggerType.CONDITION,
    trigger_config={"condition_name": "inventory_alert"},
    workflow="inventory_alert",
))

# One-time: at a specific datetime
scheduler.register_task(ScheduledTask(
    id="monthly_cleanup",
    name="Monthly Data Cleanup",
    trigger_type=TriggerType.ONCE,
    trigger_config={"run_at": "2026-05-01T02:00:00"},
    workflow="cleanup",
))

scheduler.start()
```

### 12.3 Trigger Types

| Trigger | `trigger_config` | Behavior |
|---|---|---|
| `CRON` | `{"cron": "0 9 * * 1-5"}` | Standard 5-field cron |
| `INTERVAL` | `{"interval_seconds": 1800}` | Fixed interval |
| `CONDITION` | `{"condition_name": "checker_name"}` | Calls registered checker |
| `ONCE` | `{"run_at": "2026-05-01T02:00:00"}` | One-time at ISO datetime |
| `EVENT` | `{"event_name": "..."}` | Reserved for future use |

### 12.4 ScheduledTask Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `id` | `str` | required | Unique task identifier |
| `name` | `str` | required | Human-readable name |
| `trigger_type` | `TriggerType` | required | Scheduling mechanism |
| `trigger_config` | `Dict` | required | Trigger-specific parameters |
| `workflow` | `str` | required | Template name or workflow ID |
| `params` | `Dict` | `{}` | Passed to orchestrator.ainvoke |
| `enabled` | `bool` | `True` | Toggle on/off |
| `timeout` | `int` | `300` | Max execution seconds |
| `retry_count` | `int` | `3` | Remaining retries |
| `retry_delay` | `int` | `60` | Seconds between retries |

### 12.5 Manual Execution

```python
await scheduler.aexecute_now("daily_report")
```

### 12.6 Lifecycle

```python
scheduler.start()   # Start daemon thread
scheduler.stop()    # Stop daemon thread
```

---

## 13. Error Handling

### 13.1 Error Hierarchy

All exceptions inherit from `LangDeepError`:

```
LangDeepError
├── ConfigurationError
│   └── InvalidPolicyError
├── ModelError
│   ├── ModelNotFoundError
│   ├── ProviderNotFoundError
│   ├── ProviderImportError
│   ├── ModelInvocationError
│   │   └── ModelTimeoutError
├── AgentError
│   ├── AgentNotFoundError
│   ├── AgentInvocationError
│   └── AgentRetryExhaustedError
├── ToolError
│   └── ToolNotFoundError
├── ExecutionError
│   ├── TaskExecutionError
│   └── CircularDependencyError
├── OrchestrationError
│   ├── RoutingError
│   ├── PlannerError
│   └── AggregatorError
└── TemplateError
    ├── TemplateNotFoundError
    └── PromptNotFoundError
```

### 13.2 Error Format

Every error carries structured context:

```python
try:
    result = orchestrator.invoke("test")
except OrchestrationError as e:
    print(e.code)       # "ORCHESTRATION_ERROR"
    print(e.detail)     # Human-readable message
    print(e.context)    # {"user_input": "test", "trace_id": "abc123"}
    print(e.cause)      # Original exception if re-raised
    print(e.to_dict())  # Full serializable representation
```

Example error output:

```
[ORCHESTRATION_ERROR] Async workflow execution failed
  | context={'user_input': 'test', 'trace_id': 'abc123'}
  | cause=ModelNotFoundError("[MODEL_NOT_FOUND] Model 'gpt5' is not registered ...")
```

### 13.3 Retry Behavior

**Agent nodes** (direct routing): Exponential backoff retry, default 3 attempts:

```python
# agent_node.py — make_agent_node
for attempt in range(1, max_retries + 1):
    try:
        result = agent.invoke(input)
        return result
    except Exception:
        if attempt == max_retries:
            return error_response
        time.sleep(2 ** (attempt - 1))
```

**Executor tasks**: Same exponential backoff pattern via `RetryTaskRunner`:

```python
orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    max_retries=5,  # Per-task retry limit
)
```

**Scheduled tasks**: Linear retry with configurable delay:

```python
ScheduledTask(
    ...,
    retry_count=3,     # Max retries
    retry_delay=60,    # Seconds between retries
)
```

### 13.4 Graceful Degradation

| Component | Failure behavior |
|---|---|
| LLMPlanGenerator | Falls back to `FallbackPlanGenerator` (single-task plan) |
| LLMMerger | Falls back to `ConcatMerger` (plain text join) |
| Task runner | Returns `err(msg)` dict, excluded from aggregation |
| Router LLM call | Defaults to `"end"` node |
| Aggregator (no successes) | Returns a fallback "unable to process" message |

---

## 14. Logging & Tracing

### 14.1 Log Format

All logs use structured `key=value` format:

```
level=INFO logger=langdeep.core.orchestrator.orchestrator trace_id=abc123 req_id=def456 input_preview=Hello msg="Orchestrator ainvoke start"
```

### 14.2 Configuring Log Level

```python
from langdeep.core.logging import configure
import logging

# Set minimum level
configure(level=logging.WARNING)

# Add a custom handler
import sys
handler = logging.StreamHandler(sys.stderr)
configure(level=logging.INFO, handler=handler)
```

### 14.3 Getting a Logger

```python
from langdeep import get_logger

logger = get_logger(__name__)
logger.info("Something happened", extra={"key": "value"})
```

Extra dict values appear in the log output as `key=value` pairs.

### 14.4 Tracing

Trace IDs and request IDs are automatically set per invocation:

```python
from langdeep import get_trace_id, set_trace_context, clear_trace_context

# Automatic — set by orchestrator.invoke()
trace_id = get_trace_id()    # e.g., "abc123def456"

# Manual — for custom scripts
trace_id = set_trace_context(trace_id="my_trace_id")
# ... do work ...
clear_trace_context()
```

---

## 15. API Reference

### 15.1 FlowOrchestrator

```python
class FlowOrchestrator:
    def __init__(
        self,
        supervisor_model: str = "gpt4o",
        max_retries: int = 3,
        enable_checkpoint: bool = True,
        prompt_dir: Optional[str] = None,
        component_dirs: Optional[List[str]] = None,  # Default: ["models", "tools", "agents"]
        llm_timeout: float = 30.0,
        checkpointer=None,
        routing_strategy: Optional[RoutingStrategy] = None,
        workflow_templates_dir: Optional[str] = None,
        execution_policy: Optional[ExecutionPolicy] = None,
        custom_nodes: Optional[Dict[str, Callable]] = None,
        plan_generator: Optional[PlanGenerator] = None,
        task_runner: Optional[TaskRunner] = None,
        result_merger: Optional[ResultMerger] = None,
    )

    # Synchronous execution
    def invoke(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        workflow_plan: Optional[List[Dict[str, Any]]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]

    # Asynchronous execution
    async def ainvoke(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        workflow_plan: Optional[List[Dict[str, Any]]] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]

    # Streaming execution (async generator)
    async def astream(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    )

    @property
    def graph(self)  # Compiled LangGraph StateGraph
```

### 15.2 Decorators

```python
@model(
    name: Optional[str] = None,
    provider: str = "openai",
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **extra_params,
)
```

```python
@agent(
    name: Optional[str] = None,
    description: str = "",
    capabilities: Optional[List[str]] = None,
    routing_keywords: Optional[List[str]] = None,
    model: str = "default",
    tools: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    priority: int = 1,
)
```

```python
@provider(name: Optional[str] = None)
# Decorated function must have signature: (config: ModelConfig) -> BaseChatModel
```

```python
@regist_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    tags: Optional[List[str]] = None,
    requires_confirmation: bool = False,
    timeout: Optional[int] = None,
)
```

### 15.3 RoutingStrategy (ABC)

```python
class RoutingStrategy(ABC):
    @abstractmethod
    def route(
        self,
        user_input: str,
        available_agents: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Return agent name or None to fall through to LLM routing."""
        ...
```

### 15.4 PlanGenerator (ABC)

```python
class PlanGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        user_request: str,
        available_agent_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Return list of task dicts with keys: id, name, agent, tools, depends_on, parallel, status."""
        ...
```

### 15.5 TaskRunner (ABC)

```python
class TaskRunner(ABC):
    @abstractmethod
    def run(
        self,
        task: Dict[str, Any],
        clean_messages: List[BaseMessage],
        state: Dict[str, Any],
        previous_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return ok(data) or err(msg) dict."""
        ...

    @abstractmethod
    async def arun(
        self,
        task: Dict[str, Any],
        clean_messages: List[BaseMessage],
        state: Dict[str, Any],
        previous_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Async version."""
        ...
```

### 15.6 ResultMerger (ABC)

```python
class ResultMerger(ABC):
    @abstractmethod
    def merge(
        self,
        user_request: str,
        agent_results: Dict[str, str],
    ) -> str:
        """Synthesize multiple agent results into one response."""
        ...
```

### 15.7 ExecutionPolicy

```python
@dataclass
class ExecutionPolicy:
    max_concurrency: int = 5
    strategy: str = "gather"  # "gather" | "sequential" | "priority_queue"
    retry_on: List[str] = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPolicy": ...

    @classmethod
    def from_file(cls, path: str) -> "ExecutionPolicy": ...

    def to_dict(self) -> Dict[str, Any]: ...
```

### 15.8 WorkflowPlanner

```python
class WorkflowPlanner:
    def __init__(self, workflow_dir: str = "workflows")

    def load_workflow(self, name: str) -> List[WorkflowNode]
    def list_templates(self) -> List[str]
    def to_plan_dicts(self, nodes: List[WorkflowNode], user_input: str = "") -> List[Dict]
    def topological_sort(self, nodes: List[WorkflowNode]) -> List[List[WorkflowNode]]
    def estimate_duration(self, nodes: List[WorkflowNode]) -> int
```

### 15.9 WorkflowNode

```python
@dataclass
class WorkflowNode:
    id: str
    type: NodeType
    name: str
    config: Dict[str, Any] = {}
    depends_on: List[str] = []
    condition: Optional[str] = None
    retry: int = 0
    timeout: int = 60
    priority: int = 0
```

### 15.10 Registries

```python
# Model registry (module-level singleton)
from langdeep.core.registry.model_registry import model_registry
model_registry.register(name: str, config: ModelConfig) -> None
model_registry.get_model(name: str) -> BaseChatModel       # Raises ModelNotFoundError
model_registry.list_models() -> List[str]

# Agent registry (module-level singleton)
from langdeep.core.registry.agent_registry import agent_registry
agent_registry.register(name: str, factory: Callable, metadata: AgentMetadata) -> None
agent_registry.get_agent(name: str) -> Any                  # Raises AgentNotFoundError
agent_registry.list_agents() -> List[str]
agent_registry.get_metadata(name: str) -> Optional[AgentMetadata]
agent_registry.get_agents_by_capability(capability: str) -> List[str]

# Tool registry (module-level singleton)
from langdeep.core.registry.tool_registry import tool_registry
tool_registry.register(tool_obj, metadata: ToolMetadata = None) -> None
tool_registry.get_tool(name: str)                            # Raises ToolNotFoundError
tool_registry.get_tools(names=None, category=None, tags=None)
tool_registry.list_tools() -> List[str]
tool_registry.get_metadata(name: str) -> Optional[ToolMetadata]

# Provider registry (module-level singleton)
from langdeep.core.registry.model_registry import provider_registry
provider_registry.register(provider_name: str, factory: Callable) -> None
provider_registry.get_provider(provider_name: str) -> Callable  # Raises ProviderNotFoundError
provider_registry.list_providers() -> List[str]
```

### 15.11 Result Helpers

```python
from langdeep.core.orchestrator.executor import ok, err

ok("task output")  # {"success": True, "data": "task output", "error": ""}
err("failed")      # {"success": False, "data": "", "error": "failed"}
```

---

## 16. Extension Points

### 16.1 Custom Graph Nodes

Add arbitrary LangGraph nodes to the orchestrator's graph:

```python
def my_audit_node(state):
    """Logs execution details and passes through."""
    logger.info("Audit", extra={"agents_called": list(state.get("agent_results", {}).keys())})
    return {"messages": []}  # Must return state-mergeable dict

orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    custom_nodes={"audit": my_audit_node},
)
```

Custom nodes are connected: `supervisor --conditional--> custom_node` and `custom_node --> aggregator`.

### 16.2 Custom PlanGenerator

```python
from langdeep import PlanGenerator

class TemplateBasedGenerator(PlanGenerator):
    def __init__(self, templates: Dict[str, List[Dict]]):
        self._templates = templates

    def generate(self, user_request, available_agent_names):
        # Match request to template by keyword
        for keyword, plan in self._templates.items():
            if keyword in user_request.lower():
                return plan
        # Fallback
        return [{"id": "default", "agent": available_agent_names[0], "depends_on": []}]

orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    plan_generator=TemplateBasedGenerator({
        "report": [
            {"id": "collect", "agent": "data_collector", "depends_on": []},
            {"id": "summarize", "agent": "summarizer", "depends_on": ["collect"]},
        ],
    }),
)
```

### 16.3 Custom TaskRunner

```python
from langdeep import TaskRunner, ok, err

class LoggingTaskRunner(TaskRunner):
    def __init__(self, wrapped: TaskRunner):
        self._wrapped = wrapped

    def run(self, task, clean_messages, state, previous_results):
        logger.info("Task start", extra={"task": task["id"]})
        result = self._wrapped.run(task, clean_messages, state, previous_results)
        logger.info("Task end", extra={"task": task["id"], "success": result.get("success")})
        return result

    async def arun(self, task, clean_messages, state, previous_results):
        logger.info("Task start", extra={"task": task["id"]})
        result = await self._wrapped.arun(task, clean_messages, state, previous_results)
        logger.info("Task end", extra={"task": task["id"], "success": result.get("success")})
        return result

orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    task_runner=LoggingTaskRunner(RetryTaskRunner(max_retries=3)),
)
```

### 16.4 Custom ResultMerger

```python
from langdeep import ResultMerger

class WeightedMerger(ResultMerger):
    """Merge results with configurable agent weights."""
    def __init__(self, weights: Dict[str, float]):
        self._weights = weights

    def merge(self, user_request, agent_results):
        # Sort by weight, prioritize high-weight agent outputs
        sorted_results = sorted(
            agent_results.items(),
            key=lambda x: self._weights.get(x[0], 0),
            reverse=True,
        )
        # Build weighted summary
        parts = []
        for name, text in sorted_results:
            weight = self._weights.get(name, 1.0)
            parts.append(f"[{name} (weight={weight})]\n{text}")
        return "\n\n".join(parts)

orchestrator = FlowOrchestrator(
    supervisor_model="gpt4o",
    result_merger=WeightedMerger({"analyst": 2.0, "researcher": 1.0}),
)
```

---

## 17. Best Practices

### 17.1 Project Structure

```
my_ai_project/
├── main.py                    # Entry point
├── agent_test.py              # CLI interface
├── models/                    # Model registrations
│   ├── openai_models.py
│   └── deepseek_models.py
├── agents/                    # Agent registrations
│   ├── assistant.py
│   ├── researcher.py
│   └── analyst.py
├── tools/                     # Tool registrations
│   ├── file_tools.py
│   ├── search_tools.py
│   └── data_tools.py
├── prompts/                   # Custom prompt overrides (optional)
│   ├── planner.md
│   └── supervisor.md
├── workflows/                 # Declarative workflow templates
│   ├── daily_report.yaml
│   └── competitor_analysis.yaml
└── execution_policy.json      # Optional policy file
```

### 17.2 Model Naming Conventions

- Use a consistent naming scheme: `provider_model_variant` (e.g., `deepseek_v4_pro`, `openai_gpt4o`)
- Keep `name` (registry key) and `model_name` (API identifier) distinct
- Put API keys in environment variables, never hardcode them in `@model` decorators
- Use `extra_params` for provider-specific options (timeout, api_version, etc.)

### 17.3 Agent Design

- **Single responsibility:** Each agent should do one thing well
- **Descriptive routing keywords:** Include common user phrasings that should trigger this agent
- **Capability tags:** Use consistent capability names across agents for `get_agents_by_capability()`
- **Priority levels:** Higher numbers run first in `priority_queue` strategy
- **Factory functions should be lightweight:** The factory is called once and cached. Don't put heavy initialization outside the factory.

### 17.4 Plan Design

- **Keep tasks coarse-grained:** Each task should be a meaningful unit of work, not a single API call
- **Minimize dependencies:** Only declare `depends_on` when truly required. The more dependencies, the less parallelism.
- **Use parallel markers:** Mark truly independent tasks with `"parallel": true`
- **Provide fallback plans:** For production, register a well-designed fallback plan instead of relying on `FallbackPlanGenerator`

### 17.5 Provider Factories

- **Always pop custom keys from `extra_params`** before passing to the LLM constructor to avoid duplicate keyword errors
- **Make a copy of `extra_params`** with `dict()` to avoid mutating the config's original dict
- **Use `ProviderImportError`** for missing optional packages
- **Validate required credentials** early and raise clear `ValueError`

### 17.6 Error Handling

- Catch `OrchestrationError` at the top level of your application
- Use `e.to_dict()` for logging structured error data
- Set `max_retries` based on your tolerance for latency vs reliability
- Monitor `error_count` in state for custom retry logic

---

## 18. Troubleshooting

### 18.1 Model Not Registered

**Symptom:**
```
ModelNotFoundError: [MODEL_NOT_FOUND] Model 'my_model' is not registered
| context={'available': ['deepseek_chat']}
```

**Causes & fixes:**

1. **Model file not in `component_dirs`.** Ensure your model `.py` file is in `models/` (or whatever directories you passed as `component_dirs`).
2. **File starts with `_`.** Auto-import skips files starting with underscore. Rename it.
3. **Import error in the file.** Check logs for `Module load failed` messages — a bad import prevents the whole module from loading.
4. **Wrong model name in code.** Verify the name matches exactly: `@model(name="my_model")` → `supervisor_model="my_model"`.

### 18.2 Module Load Failed

**Symptom:**
```
ERROR ... component_module=models.my_model error=No module named 'langdeep.registry'
```

**Cause:** Import error in the registered module file.

**Fix:** Check the import paths. Remember the correct paths:
- `from langdeep.core.registry.model_registry import ModelConfig` (not `langdeep.registry`)
- `from langdeep import model, agent, provider, regist_tool` (top-level exports)

### 18.3 Duplicate Keyword Argument

**Symptom:**
```
TypeError: ChatOpenAI() got multiple values for keyword argument 'request_timeout'
```

**Cause:** A parameter is passed both as an explicit keyword and via `**extra_params`.

**Fix:** In your provider factory, pop the key from `extra` before unpacking:

```python
extra = dict(config.extra_params or {})
request_timeout = extra.pop("request_timeout", 90)
# Now **extra no longer contains request_timeout
```

### 18.4 Tool Call Not Supported

**Symptom:**
```
Error code: 400 - {'error': {'message': 'deepseek-reasoner does not support this tool_choice'}}
```

**Cause:** The model being used doesn't support `tool_choice` (e.g., DeepSeek V4 when thinking mode is active, or DeepSeek R1 which doesn't support function calling at all).

**Fixes:**
1. Use a model that supports tool calls (e.g., `deepseek-chat` / V3, `gpt-4o`, `claude-sonnet-4-6`)
2. If using DeepSeek V4, ensure non-thinking mode is active when making tool calls
3. As a workaround, remove `tool_choice="required"` from the router (the `_parse_tool_call` function handles text responses as fallback)

### 18.5 Missing API Key

**Symptom:**
```
ValueError: Missing DEEPSEEK_API_KEY. Set the environment variable or pass api_key in @model config.
```

**Fix:** Either:
```bash
export DEEPSEEK_API_KEY="sk-..."
```
Or pass it in the decorator (not recommended for production):
```python
@model(name="my_model", provider="deepseek", api_key="sk-...")
```

### 18.6 Agent Returns "No Answer"

**Symptom:**
```
✅ Answer:
未获取到回答
```

**Causes:**
1. Router LLM call failed and defaulted to `"end"` node — check for routing errors in logs
2. No agent was matched — add more routing keywords or check if the user input is handled
3. Agent ran but returned empty output — check the agent's implementation

### 18.7 Tasks Not Running in Parallel

**Symptom:** Tasks that should run in parallel are executing sequentially.

**Check:**
1. `ExecutionPolicy.strategy` — must be `"gather"` or `"priority_queue"`, not `"sequential"`
2. `ExecutionPolicy.max_concurrency` — must be `> 1`
3. Task `depends_on` — over-specifying dependencies prevents parallel execution
4. The plan's `"parallel"` markers — the planner might have set them to `false`

### 18.8 Getting Help

Check logs for structured error context:
```
level=ERROR ... trace_id=abc123 req_id=def456 error=... msg="..."
```

Use the trace_id to correlate errors across multiple log lines. All error types provide `.to_dict()` for serialization.
