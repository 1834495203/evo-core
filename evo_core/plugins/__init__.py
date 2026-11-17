"""Pluggable module registry for evo-core."""

from __future__ import annotations

from ..config import CoreConfig
from ..protocol import ResearchPlugin
from .budget import BudgetPlugin
from .compactor import ContextCompactorPlugin
from .harness import ResearchHarnessPlugin
from .literature import LiteratureSearchPlugin
from .memory import MemoryPlugin
from .sanity_gate import SanityGatePlugin
from .skills import SkillsPlugin
from .subagents import SubagentsPlugin


def load_plugins(config: CoreConfig) -> list[ResearchPlugin]:
    """Instantiate the ordered set of configurable plugins from *config*."""
    return [
        BudgetPlugin(),
        ContextCompactorPlugin(),
        LiteratureSearchPlugin(),
        ResearchHarnessPlugin(),
        SkillsPlugin(),
        MemoryPlugin(),
        SubagentsPlugin(),
        SanityGatePlugin(),
    ]


__all__ = [
    "BudgetPlugin",
    "ContextCompactorPlugin",
    "LiteratureSearchPlugin",
    "MemoryPlugin",
    "ResearchHarnessPlugin",
    "SanityGatePlugin",
    "SkillsPlugin",
    "SubagentsPlugin",
    "load_plugins",
]
