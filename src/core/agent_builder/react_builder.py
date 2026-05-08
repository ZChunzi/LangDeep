"""LangGraph ReAct agent builder."""

from typing import Any, Optional

from .base import BaseAgentBuilder
from ..registry.agent_registry import AgentMetadata
from ..registry.model_registry import model_registry
from ..registry.tool_registry import tool_registry


class ReActAgentBuilder(BaseAgentBuilder):
    """Build agents with langgraph.prebuilt.create_react_agent."""

    def build(self, metadata: AgentMetadata) -> Any:
        from langgraph.prebuilt import create_react_agent

        llm = model_registry.get_model(metadata.model_name)
        tools = tool_registry.get_tools(names=metadata.tools)
        prompt = self._resolve_prompt(metadata)

        kwargs = {"model": llm, "tools": tools}
        if prompt:
            kwargs["prompt"] = prompt
        return create_react_agent(**kwargs)

    def _resolve_prompt(self, metadata: AgentMetadata) -> Optional[Any]:
        if metadata.system_prompt:
            return metadata.system_prompt
        if not metadata.prompt_path:
            return None

        from pathlib import Path

        path = Path(metadata.prompt_path)
        if not path.is_file():
            from ..errors import PromptNotFoundError

            raise PromptNotFoundError(
                f"Agent prompt file not found: {metadata.prompt_path}",
                context={"agent": metadata.name, "prompt_path": metadata.prompt_path},
            )
        return path.read_text(encoding="utf-8")
