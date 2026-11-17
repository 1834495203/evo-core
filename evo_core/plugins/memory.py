"""Pluggable persistent observation memory (zero prompt bloat when store is empty)."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from ..protocol import BasePlugin, HookContext, RunState, ToolDef


class MemoryPlugin(BasePlugin):
    """Stores durable cross-run insights; avoids forcing preflight searches on empty stores."""

    name = "memory"

    def enabled(self, cfg: Any) -> bool:
        mcfg = cfg.plugins.memory
        return bool(mcfg.enabled) and mcfg.mode in {"lightweight", "full"}

    def _obs_path(self, ctx: HookContext) -> Path:
        custom = ctx.config.plugins.memory.memory_dir
        base = Path(custom).resolve() if custom else (ctx.run_dir / "memory")
        base.mkdir(parents=True, exist_ok=True)
        return base / "observations.jsonl"

    def _load_observations(self, ctx: HookContext) -> list[dict[str, Any]]:
        path = self._obs_path(ctx)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def system_prompt_section(self, state: RunState, ctx: HookContext) -> str | None:
        mode = ctx.config.plugins.memory.mode
        rows = self._load_observations(ctx)
        if not rows and mode == "lightweight":
            # Key optimization over EvoScientist: zero bloat on cold start!
            return None

        parts: list[str] = ["# Persistent Research Memory"]
        if rows:
            index_lines = [
                f"- {r['id']} [{r.get('memory_type', 'semantic')}]: {r.get('summary', '')}"
                for r in rows[-15:]
            ]
            parts.append("Relevant prior observations exist — inspect with `search_observations` / `read_observation` if applicable:")
            parts.extend(index_lines)
        if mode == "full":
            parts.append(
                "Use `record_observation(summary, detail, memory_type)` ONLY for durable, "
                "non-obvious, evidence-backed environment workarounds or verified research insights."
            )
        return "\n".join(parts)

    def tools(self, ctx: HookContext) -> list[ToolDef]:
        mode = ctx.config.plugins.memory.mode
        rows = self._load_observations(ctx)
        tools: list[ToolDef] = []

        if rows or mode == "full":
            def search_observations(query: str, limit: int = 8) -> str:
                current = self._load_observations(ctx)
                if not current:
                    return "No observations recorded yet."
                terms = [t for t in re.split(r"\W+", query.lower()) if t]
                scored: list[tuple[int, dict[str, Any]]] = []
                for row in current:
                    blob = f"{row.get('summary', '')} {row.get('detail', '')}".lower()
                    score = sum(blob.count(t) for t in terms) if terms else 1
                    if score > 0:
                        scored.append((score, row))
                scored.sort(key=lambda x: x[0], reverse=True)
                hits = scored[: max(1, min(20, int(limit)))]
                if not hits:
                    return f"No matching observations for '{query}'."
                return "\n".join(
                    f"- {r['id']} ({r.get('memory_type', 'semantic')}): {r.get('summary', '')}"
                    for _, r in hits
                )

            def read_observation(observation_id: str) -> str:
                for row in self._load_observations(ctx):
                    if row.get("id") == observation_id:
                        return json.dumps(row, ensure_ascii=False, indent=2)
                return f"Observation not found: {observation_id}"

            tools.extend(
                [
                    ToolDef(
                        name="search_observations",
                        description="Search prior durable research observations by keyword.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                        handler=search_observations,
                    ),
                    ToolDef(
                        name="read_observation",
                        description="Read full details of a prior observation by ID.",
                        parameters={
                            "type": "object",
                            "properties": {"observation_id": {"type": "string"}},
                            "required": ["observation_id"],
                        },
                        handler=read_observation,
                    ),
                ]
            )

        if mode == "full":
            def record_observation(
                summary: str,
                detail: str,
                memory_type: str = "semantic",
            ) -> str:
                path = self._obs_path(ctx)
                entry = {
                    "id": f"obs_{uuid.uuid4().hex[:8]}",
                    "memory_type": memory_type,
                    "summary": summary.strip(),
                    "detail": detail.strip(),
                }
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                return f"Recorded observation {entry['id']}."

            tools.append(
                ToolDef(
                    name="record_observation",
                    description="Save a durable, reusable non-obvious research finding or environment workaround.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "One-line summary."},
                            "detail": {"type": "string", "description": "Evidence-backed detail."},
                            "memory_type": {
                                "type": "string",
                                "enum": ["semantic", "procedural", "episodic"],
                            },
                        },
                        "required": ["summary", "detail"],
                    },
                    handler=record_observation,
                )
            )

        return tools
