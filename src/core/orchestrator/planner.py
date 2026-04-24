"""Workflow planning — template-based and LLM-driven plan generation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from ..logging import get_logger
from ..errors import PlannerError, TemplateNotFoundError
from ..registry.agent_registry import agent_registry

logger = get_logger(__name__)


# ── Extension point ──────────────────────────────────────────────────────────────

class PlanGenerator(ABC):
    """Pluggable plan generator.

    Subclass to provide custom planning logic (e.g. hardcoded plans,
    domain-specific planners, or external plan sources).
    """

    @abstractmethod
    def generate(
        self,
        user_request: str,
        available_agent_names: List[str],
    ) -> List[Dict[str, Any]]:
        ...


class FallbackPlanGenerator(PlanGenerator):
    """Returns a single-task plan using the first registered agent."""

    def generate(
        self,
        user_request: str,
        available_agent_names: List[str],
    ) -> List[Dict[str, Any]]:
        agent = available_agent_names[0] if available_agent_names else "default_agent"
        return [{
            "id": "task_1",
            "name": "Process user request",
            "agent": agent,
            "tools": [],
            "depends_on": [],
            "status": "pending",
        }]


class LLMPlanGenerator(PlanGenerator):
    """Uses an LLM to generate a multi-step workflow plan."""

    def __init__(self, model_name: str, prompt_loader=None):
        self._model_name = model_name
        self._prompt_loader = prompt_loader

    def generate(
        self,
        user_request: str,
        available_agent_names: List[str],
    ) -> List[Dict[str, Any]]:
        from ..registry.model_registry import model_registry

        llm = model_registry.get_model(self._model_name)

        try:
            if self._prompt_loader:
                planner_prompt = self._prompt_loader.load_prompt("planner")
                prompt_msgs = planner_prompt.format_messages(
                    user_request=user_request,
                    available_agents=available_agent_names,
                )
            else:
                raise RuntimeError("No prompt loader configured")
        except Exception:
            prompt_msgs = [HumanMessage(content=(
                f"Create a JSON plan for: {user_request}\n"
                f"Available agents: {available_agent_names}\n"
                f"Output only JSON — a list of tasks with id/agent/status fields."
            ))]

        try:
            response = llm.invoke(prompt_msgs)
            return parse_plan_content(str(response.content))
        except Exception as exc:
            logger.error("LLM plan generation failed", extra={"error": str(exc)})
            return FallbackPlanGenerator().generate(user_request, available_agent_names)


# ── Planner node logic ───────────────────────────────────────────────────────────

class Planner:
    """Workflow planning node.

    Parameters:
        model_name: Registered model for LLM planning.
        plan_generator: Optional custom PlanGenerator instance.
        prompt_loader: Optional prompt loader for the planner prompt.
    """

    def __init__(
        self,
        model_name: str,
        plan_generator: Optional[PlanGenerator] = None,
        prompt_loader=None,
    ):
        self._generator = plan_generator or LLMPlanGenerator(model_name, prompt_loader)
        self._fallback = FallbackPlanGenerator()

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a workflow_plan for the given state, reusing any existing plan."""
        existing = state.get("workflow_plan")
        if existing and len(existing) > 0:
            logger.info("Reusing existing workflow plan", extra={"plan_size": len(existing)})
            return {"workflow_plan": existing}

        user_request = _last_human(state["messages"])
        agents = agent_registry.list_agents()

        try:
            plan = self._generator.generate(user_request, agents)
        except Exception as exc:
            logger.error("Plan generation failed, using fallback", extra={"error": str(exc)})
            plan = self._fallback.generate(user_request, agents)

        logger.info("Plan created", extra={"task_count": len(plan)})
        return {"workflow_plan": plan}


# ── Template support (kept separate — only loaded when a template dir is given) ──

class TemplateLoader:
    """Loads and applies predefined workflow templates from YAML/JSON files."""

    def __init__(self, templates_dir: Optional[str] = None):
        self._templates: Dict[str, List[Dict[str, Any]]] = {}
        if templates_dir:
            self._load(templates_dir)

    def _load(self, directory: str) -> None:
        import json
        import yaml
        from pathlib import Path

        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning("Template directory not found", extra={"path": directory})
            return

        for fpath in sorted(dir_path.iterdir()):
            try:
                if fpath.suffix in (".yaml", ".yml"):
                    with open(fpath, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                elif fpath.suffix == ".json":
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    continue
                tpl_id = data.get("id", fpath.stem)
                steps = data.get("steps", data.get("nodes", []))
                self._templates[tpl_id] = steps
                logger.info("Template loaded", extra={"template_id": tpl_id, "file": fpath.name})
            except Exception as exc:
                logger.error("Template load failed", extra={"file": str(fpath), "error": str(exc)})

    @property
    def template_names(self) -> List[str]:
        return list(self._templates.keys())

    def apply(self, template_name: str, user_input: str) -> List[Dict[str, Any]]:
        steps = self._templates.get(template_name)
        if not steps:
            raise TemplateNotFoundError(
                f"Template '{template_name}' not found",
                context={"available": list(self._templates.keys())},
            )
        import json
        plan = json.loads(json.dumps(steps))  # deep copy
        for task in plan:
            for key, value in list(task.items()):
                if isinstance(value, str) and "{{ user_input }}" in value:
                    task[key] = value.replace("{{ user_input }}", user_input)
                if key == "config" and isinstance(value, dict):
                    for ck, cv in list(value.items()):
                        if isinstance(cv, str) and "{{ user_input }}" in cv:
                            value[ck] = cv.replace("{{ user_input }}", user_input)
        return plan


# ── JSON helpers ─────────────────────────────────────────────────────────────────

def parse_plan_content(content: str) -> List[Dict[str, Any]]:
    """Parse LLM response into a task list, with fallback."""
    import json
    try:
        text = content.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        plan = json.loads(text)
        if isinstance(plan, list):
            return plan
        if isinstance(plan, dict) and "tasks" in plan:
            return plan["tasks"]
        logger.warning("Unexpected plan format", extra={"type": str(type(plan))})
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Plan JSON parse failed", extra={"error": str(exc)})
    return FallbackPlanGenerator().generate("", [])


def update_plan_status(plan: List[Dict], results: Dict[str, Any]) -> List[Dict]:
    """Mark tasks as completed by matching task_id keys in results."""
    for task in plan:
        tid = task.get("id")
        if tid and tid in results:
            task["status"] = "completed"
    return plan


def _last_human(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            return str(m.content)
    return ""
