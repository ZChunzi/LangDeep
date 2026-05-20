# Workflow Plans

Workflow plans let callers provide deterministic multi-step execution instead of
letting an LLM planner create tasks.

## Shape

Plans are lists of task dictionaries. Common fields include:

- `id`
- `agent`
- `task`
- `depends_on`
- `requires_confirmation`
- `status`

Use `validate_workflow_plan()` to validate externally supplied plans.

## Dependencies

`depends_on` declares task ordering. The executor can run independent tasks
concurrently depending on the active execution policy.
