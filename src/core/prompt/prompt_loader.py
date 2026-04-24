"""Markdown prompt loader with frontmatter parsing and caching."""

import importlib.resources
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ..logging import get_logger
from ..errors import PromptNotFoundError, ConfigurationError

logger = get_logger(__name__)


@dataclass
class PromptConfig:
    """Parsed frontmatter metadata for a prompt."""
    name: str
    version: str
    description: str
    model: Optional[str] = None
    variables: List[str] = field(default_factory=list)


class MarkdownPromptLoader:
    """Loads and caches ChatPromptTemplate instances from markdown files.

    File format::

        ---
        name: math_solver
        version: 1.0
        description: Math problem solver
        model: gpt4o
        variables: [question, context]
        ---

        # System

        You are a math expert...

        # Human

        Please solve: {question}
    """

    def __init__(self, prompt_dir: Optional[str] = None):
        self._prompt_dir = Path(prompt_dir) if prompt_dir else None
        try:
            self._builtin_dir = importlib.resources.files("langdeep") / "resources" / "prompts"
        except Exception:
            self._builtin_dir = None
        self._cache: Dict[str, ChatPromptTemplate] = {}

    def load_prompt(self, name: str) -> ChatPromptTemplate:
        if name in self._cache:
            return self._cache[name]

        prompt_file = self._find_prompt_file(name)
        with open(prompt_file, encoding="utf-8") as f:
            content = f.read()

        config, body = self._parse_frontmatter(content)
        template = self._parse_markdown_prompt(body)
        self._cache[name] = template
        logger.debug("Prompt loaded", extra={"name": name, "version": config.get("version", "unknown")})
        return template

    def reload(self, name: Optional[str] = None):
        if name:
            self._cache.pop(name, None)
            logger.debug("Prompt cache entry cleared", extra={"name": name})
        else:
            self._cache.clear()
            logger.debug("All prompt cache entries cleared")

    def load_all_prompts(self) -> Dict[str, ChatPromptTemplate]:
        prompts = {}
        if self._prompt_dir is not None:
            for file in self._prompt_dir.glob("*.md"):
                name = file.stem.replace(".prompt", "")
                prompts[name] = self.load_prompt(name)
        return prompts

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_prompt_file(self, name: str) -> Path:
        # External dir takes precedence
        if self._prompt_dir is not None:
            for suffix in (".md", ".prompt.md"):
                ext_file = self._prompt_dir / f"{name}{suffix}"
                if ext_file.exists():
                    return ext_file
        # Fall back to built-in
        if self._builtin_dir is not None:
            for suffix in (".md", ".prompt.md"):
                builtin_file = self._builtin_dir / f"{name}{suffix}"
                if builtin_file.exists():
                    return builtin_file
        search = []
        if self._prompt_dir:
            search.append(str(self._prompt_dir))
        if self._builtin_dir:
            search.append(str(self._builtin_dir))
        raise PromptNotFoundError(
            f"Prompt file '{name}' not found",
            context={"search_paths": search},
        )

    def _parse_frontmatter(self, content: str) -> tuple:
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(pattern, content, re.DOTALL)
        if not match:
            return {}, content
        config = {}
        for line in match.group(1).split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key, value = key.strip(), value.strip()
                if value.startswith("[") and value.endswith("]"):
                    value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                elif value.startswith("\"") and value.endswith("\""):
                    value = value[1:-1]
                config[key] = value
        return config, match.group(2)

    def _parse_markdown_prompt(self, content: str) -> ChatPromptTemplate:
        messages = []
        current_role = None
        current_lines = []

        for line in content.split("\n"):
            if line.startswith("# System"):
                if current_role and current_lines:
                    messages.append(self._make_message(current_role, "\n".join(current_lines)))
                current_role, current_lines = "system", []
            elif line.startswith("# Human") or line.startswith("# User"):
                if current_role and current_lines:
                    messages.append(self._make_message(current_role, "\n".join(current_lines)))
                current_role, current_lines = "human", []
            elif line.startswith("# AI") or line.startswith("# Assistant"):
                if current_role and current_lines:
                    messages.append(self._make_message(current_role, "\n".join(current_lines)))
                current_role, current_lines = "ai", []
            else:
                if current_role:
                    current_lines.append(line)

        if current_role and current_lines:
            messages.append(self._make_message(current_role, "\n".join(current_lines)))

        return ChatPromptTemplate(messages=messages)

    @staticmethod
    def _make_message(role: str, content: str):
        if role == "system":
            return SystemMessagePromptTemplate.from_template(content)
        elif role == "human":
            return HumanMessagePromptTemplate.from_template(content)
        elif role == "ai":
            return AIMessagePromptTemplate.from_template(content)
        else:
            raise ConfigurationError(
                f"Unknown role type: {role}",
                context={"valid_roles": ["system", "human", "ai"]},
            )


# Global singleton (no external dir)
prompt_loader = MarkdownPromptLoader()
