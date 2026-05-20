# Concepts

LangDeep is a registry-based framework for LangChain and LangGraph multi-agent
workflows.

## Registries

Decorators write metadata into process-local registries for models, providers,
agents, tools, memory, cache, IM channels, and sandbox backends.

Registries are not distributed configuration stores. Production services should
import all component modules during startup and run diagnostics before serving
traffic.

## Runtime Graph

`FlowOrchestrator` builds a graph with supervisor, planner, executor,
aggregator, one node per registered agent, and optional custom nodes.

## State

The graph state carries messages, routing target, workflow plan, current task,
task context, agent results, retry counts, and optional memory metadata.

## Extension Points

Use local abstractions when integrating: `RoutingStrategy`, `PlanGenerator`,
`TaskRunner`, `ResultMerger`, and backend interfaces for memory, cache, and
sandbox.
