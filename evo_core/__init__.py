"""Minimal autonomous research kernel with pluggable modules."""

from .config import CoreConfig, load_config
from .engine import ResearchKernel, RunResult
from .protocol import HookContext, ResearchPlugin, RunState, ToolDef, TurnDecision

__all__ = [
    "CoreConfig",
    "HookContext",
    "ResearchKernel",
    "ResearchPlugin",
    "RunResult",
    "RunState",
    "ToolDef",
    "TurnDecision",
    "load_config",
]
