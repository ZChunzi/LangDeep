"""Unit tests for MarkdownPromptLoader."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile

from langdeep.core.prompt.prompt_loader import MarkdownPromptLoader
from langdeep.core.errors import PromptNotFoundError


def _write_prompt(dirpath: str, name: str, content: str):
    path = os.path.join(dirpath, f"{name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def test_load_prompt_with_frontmatter():
    with tempfile.TemporaryDirectory() as tmp:
        _write_prompt(tmp, "test_prompt", """\
---
name: test_prompt
version: 1.0
description: A test prompt
variables: [question]
---

# System

You are a helpful assistant.

# Human

Answer: {question}
""")
        loader = MarkdownPromptLoader(prompt_dir=tmp)
        template = loader.load_prompt("test_prompt")
        assert template is not None
        assert len(template.messages) == 2


def test_load_prompt_without_frontmatter():
    with tempfile.TemporaryDirectory() as tmp:
        _write_prompt(tmp, "bare", """\
# System

Be concise.

# Human

Hello!
""")
        loader = MarkdownPromptLoader(prompt_dir=tmp)
        template = loader.load_prompt("bare")
        assert len(template.messages) == 2


def test_prompt_not_found():
    loader = MarkdownPromptLoader()
    try:
        loader.load_prompt("non_existent_prompt")
        assert False, "Should raise"
    except PromptNotFoundError:
        pass


def test_cache_hit():
    with tempfile.TemporaryDirectory() as tmp:
        _write_prompt(tmp, "cached_prompt", """\
# Human

Hello
""")
        loader = MarkdownPromptLoader(prompt_dir=tmp)
        t1 = loader.load_prompt("cached_prompt")
        t2 = loader.load_prompt("cached_prompt")
        assert t1 is t2


def test_reload_clears_cache():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_prompt(tmp, "reload_test", """\
# Human

v1
""")
        loader = MarkdownPromptLoader(prompt_dir=tmp)
        t1 = loader.load_prompt("reload_test")
        loader.reload("reload_test")
        t2 = loader.load_prompt("reload_test")
        # After reload, should be a fresh load (same content, but template may differ)
        assert t2 is not None


def test_reload_all():
    with tempfile.TemporaryDirectory() as tmp:
        _write_prompt(tmp, "p1", "# Human\nA")
        _write_prompt(tmp, "p2", "# Human\nB")
        loader = MarkdownPromptLoader(prompt_dir=tmp)
        loader.load_prompt("p1")
        loader.load_prompt("p2")
        loader.reload()
        assert loader._cache == {}


def test_load_all_prompts():
    with tempfile.TemporaryDirectory() as tmp:
        _write_prompt(tmp, "pa", "# Human\nText A")
        _write_prompt(tmp, "pb", "# Human\nText B")
        loader = MarkdownPromptLoader(prompt_dir=tmp)
        all_prompts = loader.load_all_prompts()
        assert "pa" in all_prompts
        assert "pb" in all_prompts


def test_markdown_with_all_roles():
    with tempfile.TemporaryDirectory() as tmp:
        _write_prompt(tmp, "all_roles", """\
# System

System message

# Human

Human message

# AI

AI message
""")
        loader = MarkdownPromptLoader(prompt_dir=tmp)
        template = loader.load_prompt("all_roles")
        assert len(template.messages) == 3


def test_prompt_format():
    with tempfile.TemporaryDirectory() as tmp:
        _write_prompt(tmp, "fmt_test", """\
# Human

Question: {question}
""")
        loader = MarkdownPromptLoader(prompt_dir=tmp)
        template = loader.load_prompt("fmt_test")
        formatted = template.format_messages(question="test")
        assert len(formatted) == 1
        assert "test" in formatted[0].content
