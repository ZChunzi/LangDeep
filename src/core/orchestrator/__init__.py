"""Modular workflow orchestrator with pluggable extension points."""

from .orchestrator import FlowOrchestrator

# Extension point ABCs — import and subclass to customise behaviour
from .router import RoutingStrategy, KeywordRoutingStrategy
from .planner import PlanGenerator, LLMPlanGenerator, FallbackPlanGenerator
from .executor import TaskRunner, RetryTaskRunner, ok, err
from .aggregator import ResultMerger, LLMMerger, ConcatMerger

__all__ = [
    "FlowOrchestrator",
    # Extension points
    "RoutingStrategy",
    "KeywordRoutingStrategy",
    "PlanGenerator",
    "LLMPlanGenerator",
    "FallbackPlanGenerator",
    "TaskRunner",
    "RetryTaskRunner",
    "ResultMerger",
    "LLMMerger",
    "ConcatMerger",
    # Utility
    "ok",
    "err",
]
