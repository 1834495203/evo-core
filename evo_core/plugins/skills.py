"""Opt-in SKILL.md discovery plugin (zero prompt bloat when no skills are configured)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..protocol import BasePlugin, HookContext, RunState, ToolDef

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def _discover_skills(dirs: list[str]) -> dict[str, tuple[str, Path]]:
    discovered: dict[str, tuple[str, Path]] = {}
    for raw_dir in dirs:
        base = Path(raw_dir).expanduser().resolve()
        if not base.is_dir():
            continue
        for skill_md in sorted(base.glob("*/SKILL.md")):
            text = skill_md.read_text(encoding="utf-8", errors="replace")
            match = _FRONTMATTER_RE.match(text)
            meta: dict[str, Any] = {}
            if match:
                try:
                    loaded = yaml.safe_load(match.group(1))
                    if isinstance(loaded, dict):
                        meta = loaded
                except yaml.YAMLError:
                    pass
            name = str(meta.get("name") or skill_md.parent.name)
            desc = str(meta.get("description") or "").strip()
            discovered[name] = (desc, skill_md)
    return discovered


class SkillsPlugin(BasePlugin):
    """Loads SKILL.md metadata from configured directories and exposes `read_skill` on demand."""

    name = "skills"

    def enabled(self, cfg: Any) -> bool:
        return bool(cfg.plugins.skills.enabled)

    def system_prompt_section(self, state: RunState, ctx: HookContext) -> str | None:
        skills = _discover_skills(ctx.config.plugins.skills.dirs)
        if not skills:
            return None
        lines = ["# Available Skills", "Call `read_skill(name)` when a skill applies:"]
        for name, (desc, _) in skills.items():
            lines.append(f"- `{name}`: {desc}")
        return "\n".join(lines)

    def tools(self, ctx: HookContext) -> list[ToolDef]:
        skills = _discover_skills(ctx.config.plugins.skills.dirs)
        if not skills:
            return []

        def read_skill(name: str) -> str:
            entry = skills.get(name)
            if entry is None:
                return f"Unknown skill '{name}'. Available: {', '.join(skills)}"
            return entry[1].read_text(encoding="utf-8", errors="replace")[:24_000]

        return [
            ToolDef(
                name="read_skill",
                description="Read the full SKILL.md instructions for an installed skill.",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                handler=read_skill,
            )
        ]
