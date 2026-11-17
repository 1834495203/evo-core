"""Configuration loading for evo-core (pure dataclasses, no global state)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from dotenv import dotenv_values


@dataclass
class BudgetPluginConfig:
    enabled: bool = True
    max_calls: int | None = None
    max_wall_seconds: float | None = None
    soft_converge_ratio: float = 0.75
    hard_report_ratio: float = 0.90


@dataclass
class CompactorPluginConfig:
    enabled: bool = True
    context_limit_tokens: int = 96_000
    trigger_ratio: float = 0.72
    keep_recent_messages: int = 10
    max_tool_result_chars: int = 12_000


@dataclass
class LiteraturePluginConfig:
    enabled: bool = True
    providers: list[str] = field(
        default_factory=lambda: ["arxiv", "openalex", "github", "semantic_scholar", "webpage"]
    )
    timeout: float = 25.0


@dataclass
class HarnessPluginConfig:
    enabled: bool = True
    max_experiments: int | None = 16
    experiment_timeout_seconds: int | None = 180
    max_parallel_workers: int = 2
    experiments_between_reviews: int = 3
    executor: str = "local"  # local | docker
    mlflow_tracking_uri: str | None = None


@dataclass
class SanityGatePluginConfig:
    enabled: bool = True
    check_python_syntax: bool = True
    enforce_test_before_sweep: bool = True


@dataclass
class SubagentsPluginConfig:
    enabled: bool = True
    max_child_calls: int | None = 25
    roles: list[str] = field(
        default_factory=lambda: [
            "planner",
            "research",
            "code",
            "debug",
            "analysis",
            "writing",
        ]
    )


@dataclass
class MemoryPluginConfig:
    enabled: bool = True
    mode: str = "lightweight"  # off | lightweight | full
    memory_dir: str | None = None


@dataclass
class SkillsPluginConfig:
    enabled: bool = False
    dirs: list[str] = field(default_factory=list)


@dataclass
class PluginsConfig:
    budget: BudgetPluginConfig = field(default_factory=BudgetPluginConfig)
    compactor: CompactorPluginConfig = field(default_factory=CompactorPluginConfig)
    literature: LiteraturePluginConfig = field(default_factory=LiteraturePluginConfig)
    harness: HarnessPluginConfig = field(default_factory=HarnessPluginConfig)
    sanity_gate: SanityGatePluginConfig = field(default_factory=SanityGatePluginConfig)
    subagents: SubagentsPluginConfig = field(default_factory=SubagentsPluginConfig)
    memory: MemoryPluginConfig = field(default_factory=MemoryPluginConfig)
    skills: SkillsPluginConfig = field(default_factory=SkillsPluginConfig)


@dataclass
class CoreConfig:
    """Explicit runtime configuration for ResearchKernel and its plugins."""

    model: str = "deepseek-flash"
    modules: dict[str, Any] = field(default_factory=dict)
    strict_roles: bool | None = None
    role_permissions: dict[str, list[str]] = field(default_factory=dict)
    role_delegations: dict[str, list[str]] = field(default_factory=dict)
    auxiliary_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    thinking: str = "enabled"  # enabled | disabled | provider_default
    reasoning_effort: str = "high"
    temperature: float = 0.2
    max_output_tokens: int = 24_000
    request_timeout: float = 180.0
    max_retries: int = 4

    workspace: str = "./workspace"
    run_dir: str = "./run_meta"
    science_python: str | None = None
    dangerous_mode: bool = False
    command_timeout: int = 180
    max_output_chars: int = 24_000

    plugins: PluginsConfig = field(default_factory=PluginsConfig)

    def resolved_api_credentials(self, env_file: Path | None = None) -> tuple[str, str]:
        """Resolve (api_key, base_url) from config, environment, or .env without mutating env."""
        env_values: dict[str, str] = {}
        candidates = [env_file] if env_file else [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for candidate in candidates:
            if candidate and candidate.is_file():
                raw = dotenv_values(candidate, encoding="utf-8-sig")
                env_values = {k: v for k, v in raw.items() if v is not None}
                break

        api_key = (
            self.api_key
            or os.environ.get("CUSTOM_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or env_values.get("OPENAI_API_KEY")
        )
        base_url = (
            self.base_url
            or os.environ.get("CUSTOM_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or env_values.get("OPENAI_BASE_URL")
        )
        if not api_key or not base_url:
            raise ValueError(
                "Missing API credentials: configure OPENAI_API_KEY and OPENAI_BASE_URL "
                "in .env, environment variables, or config."
            )
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Invalid base_url: {base_url!r}")
        return api_key, base_url.rstrip("/")

    def effective_aux_model(self) -> str:
        return self.auxiliary_model or self.model


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, val in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if isinstance(val, dict) and hasattr(current, "__dataclass_fields__"):
            _merge_dataclass(current, val)
        else:
            setattr(instance, key, val)
    return instance


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> CoreConfig:
    """Load CoreConfig from YAML file (if given) and apply explicit overrides."""
    cfg = CoreConfig()
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a YAML mapping.")
        _merge_dataclass(cfg, raw)
    if overrides:
        _merge_dataclass(cfg, overrides)
    return cfg
