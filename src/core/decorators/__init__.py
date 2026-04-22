"""Decorators for annotation-based registration."""
from .model import model
from .tool import regist_tool
from .agent import agent
from .provider import provider

__all__ = ["model", "regist_tool", "agent", "provider"]