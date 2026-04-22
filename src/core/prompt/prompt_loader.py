"""Markdown format prompt loader."""
import os
import re
import importlib.resources
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)


@dataclass
class PromptConfig:
    """Prompt configuration"""
    name: str
    version: str
    description: str
    model: Optional[str] = None
    variables: List[str] = field(default_factory=list)


class MarkdownPromptLoader:
    """
    Markdown format prompt loader

    Markdown file format:
        ---
        name: math_solver
        version: 1.0
        description: Math problem solver prompt
        model: gpt4o
        variables: [question, context]
        ---

        # System

        You are a math expert, skilled at solving various math problems.

        ## Context

        {context}

        # Human

        Please help me solve the following problem:

        {question}

        ## Requirements

        1. Provide detailed solution steps
        2. Use LaTeX format for final answer
    """

    def __init__(self, prompt_dir: Optional[str] = None):
        """
        Initialize the prompt loader.

        Args:
            prompt_dir: Optional external directory containing user prompt files.
                        If provided, files in this directory take precedence over
                        built-in prompts. If None, only built-in prompts are used.
        """
        # 用户指定的外部目录（可选）
        if prompt_dir is not None:
            self.prompt_dir = Path(prompt_dir)
        else:
            self.prompt_dir = None

        # 内置资源目录（打包在 langdeep 包内）
        try:
            self._builtin_dir = importlib.resources.files("langdeep") / "resources" / "prompts"
        except Exception:
            # 如果包资源不可用（例如未安装），回退到 None
            self._builtin_dir = None

        self._cache: Dict[str, ChatPromptTemplate] = {}

    def _find_prompt_file(self, name: str) -> Path:
        """
        Find the prompt file by name.
        
        Search order:
        1. External user directory (if specified and file exists)
        2. Built-in package resource directory

        Args:
            name: Prompt name (without extension)

        Returns:
            Path to the found prompt file.

        Raises:
            FileNotFoundError: If no matching file is found.
        """
        # 1. 外部目录优先
        if self.prompt_dir is not None:
            external_file = self.prompt_dir / f"{name}.md"
            if external_file.exists():
                return external_file
            external_file2 = self.prompt_dir / f"{name}.prompt.md"
            if external_file2.exists():
                return external_file2

        # 2. 回退到内置资源
        if self._builtin_dir is not None:
            builtin_file = self._builtin_dir / f"{name}.md"
            if builtin_file.exists():
                return builtin_file
            builtin_file2 = self._builtin_dir / f"{name}.prompt.md"
            if builtin_file2.exists():
                return builtin_file2

        # 未找到
        search_paths = []
        if self.prompt_dir:
            search_paths.append(str(self.prompt_dir))
        if self._builtin_dir:
            search_paths.append(str(self._builtin_dir))
        raise FileNotFoundError(f"Prompt file '{name}' not found in {search_paths}")

    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter"""
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            return {}, content

        frontmatter_str = match.group(1)
        body = match.group(2)

        # Simple YAML parsing (use pyyaml in production)
        config = {}
        for line in frontmatter_str.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Parse list
                if value.startswith('[') and value.endswith(']'):
                    value = [v.strip() for v in value[1:-1].split(',') if v.strip()]
                # Remove quotes
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                config[key] = value

        return config, body

    def _parse_markdown_prompt(self, content: str) -> ChatPromptTemplate:
        """Parse Markdown format prompt"""
        messages = []
        current_role = None
        current_content = []

        lines = content.split('\n')

        for line in lines:
            # Detect role headers
            if line.startswith('# System'):
                if current_role and current_content:
                    messages.append(self._create_message(current_role, '\n'.join(current_content)))
                current_role = 'system'
                current_content = []
            elif line.startswith('# Human') or line.startswith('# User'):
                if current_role and current_content:
                    messages.append(self._create_message(current_role, '\n'.join(current_content)))
                current_role = 'human'
                current_content = []
            elif line.startswith('# AI') or line.startswith('# Assistant'):
                if current_role and current_content:
                    messages.append(self._create_message(current_role, '\n'.join(current_content)))
                current_role = 'ai'
                current_content = []
            else:
                if current_role:
                    current_content.append(line)

        # Process last message block
        if current_role and current_content:
            messages.append(self._create_message(current_role, '\n'.join(current_content)))

        return ChatPromptTemplate(messages=messages)

    def _create_message(self, role: str, content: str):
        """Create message template"""
        if role == 'system':
            return SystemMessagePromptTemplate.from_template(content)
        elif role == 'human':
            return HumanMessagePromptTemplate.from_template(content)
        elif role == 'ai':
            return AIMessagePromptTemplate.from_template(content)
        else:
            raise ValueError(f"Unknown role type: {role}")

    def load_prompt(self, name: str) -> ChatPromptTemplate:
        """Load specified prompt template"""
        if name in self._cache:
            return self._cache[name]

        # Find prompt file using search strategy
        prompt_file = self._find_prompt_file(name)

        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        config, body = self._parse_frontmatter(content)
        prompt_template = self._parse_markdown_prompt(body)

        self._cache[name] = prompt_template
        return prompt_template

    def load_all_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Load all prompt templates"""
        prompts = {}
        # 注意：此方法仅扫描外部目录（如果存在），不会递归内置目录
        if self.prompt_dir is not None:
            for file in self.prompt_dir.glob("*.md"):
                name = file.stem.replace('.prompt', '')
                prompts[name] = self.load_prompt(name)
        # 若需加载内置所有 Prompt，需额外实现（暂保持原行为）
        return prompts

    def reload(self, name: Optional[str] = None):
        """Reload prompt (supports hot update)"""
        if name:
            if name in self._cache:
                del self._cache[name]
        else:
            self._cache.clear()


# Global singleton - 默认使用内置 Prompt（无外部目录）
prompt_loader = MarkdownPromptLoader()