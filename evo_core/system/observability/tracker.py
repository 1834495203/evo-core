"""Externalized experiment ledger & execution span tracker (JSONL + optional MLflow)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from ..domain import now


class ExperimentTracker:
    """Appends structured experiment metrics/validity and LLM/tool spans to durable JSONL (+ optional MLflow)."""

    def __init__(
        self,
        tracking_dir: Path,
        *,
        mlflow_tracking_uri: str | None = None,
        experiment_name: str = "evo-core",
    ) -> None:
        self.tracking_dir = Path(tracking_dir).resolve()
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.tracking_dir / "experiments.jsonl"
        self.spans_file = self.tracking_dir / "spans.jsonl"
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name

    def log_job(self, job: Any) -> None:
        data = asdict(job) if is_dataclass(job) else dict(job)
        result = data.get("result") or {}
        metrics = result.get("metrics") or {}
        entry = {
            "logged_at": now(),
            "job_id": data.get("id"),
            "study_id": data.get("study_id"),
            "plan_version": data.get("plan_version"),
            "status": data.get("status"),
            "validity": data.get("validity", "unverified"),
            "invalidated_reason": data.get("invalidated_reason", ""),
            "spec": data.get("spec", {}),
            "manifest": data.get("manifest", {}),
            "metrics": metrics,
            "error": result.get("error", ""),
            "checks": data.get("checks", []),
        }
        with self.experiments_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.mlflow_tracking_uri:
            self._try_log_mlflow(entry)

    def _try_log_mlflow(self, entry: dict[str, Any]) -> None:
        try:
            import mlflow  # type: ignore[import-not-found]
        except Exception:
            return
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=str(entry.get("job_id") or "exp")):
                spec = entry.get("spec") or {}
                mlflow.log_params(
                    {
                        "entrypoint": str(spec.get("entrypoint", "")),
                        "protocol": str(spec.get("protocol", "")),
                        "data_version": str(spec.get("data_version", "")),
                        "seed": str(spec.get("seed", 0)),
                        "plan_version": str(entry.get("plan_version", 1)),
                    }
                )
                mlflow.set_tags(
                    {
                        "job_id": str(entry.get("job_id", "")),
                        "study_id": str(entry.get("study_id", "")),
                        "status": str(entry.get("status", "")),
                        "validity": str(entry.get("validity", "unverified")),
                        "purpose": str(spec.get("purpose", ""))[:200],
                    }
                )
                numeric = {
                    k: float(v)
                    for k, v in (entry.get("metrics") or {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                if numeric:
                    mlflow.log_metrics(numeric)
        except Exception:
            pass

    def log_span(self, kind: str, payload: dict[str, Any]) -> None:
        entry = {"time": now(), "kind": kind, **payload}
        with self.spans_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def query(self, after: int = 0, limit: int = 100, actor: str = "", run_id: str = "") -> dict:
        cursor = max(0, after); rows = []
        if not self.spans_file.exists(): return {"events": [], "next_cursor": cursor}
        with self.spans_file.open(encoding="utf-8") as stream:
            for sequence, line in enumerate(stream, 1):
                if sequence <= after: continue
                try: row = json.loads(line)
                except json.JSONDecodeError: break  # don't consume a partially written final line
                cursor = sequence
                identity = row.get("identity") or {}
                if actor and (identity.get("actor_id") or row.get("actor")) != actor: continue
                if run_id and (identity.get("run_id") or row.get("run_id")) != run_id: continue
                rows.append({**row, "sequence": sequence})
                if len(rows) >= max(1, min(500, limit)): break
        return {"events": rows, "next_cursor": cursor}

    def log_llm_call(
        self,
        actor: str,
        call_num: int,
        request_messages: list[dict[str, Any]],
        response: Any,
        *,
        parent_span_id: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        tool_calls = getattr(response, "tool_calls", None) or []
        payload: dict[str, Any] = {
            "actor": actor,
            "call": call_num,
            "parent_span_id": parent_span_id,
            "duration_ms": duration_ms,
            "msg_count": len(request_messages),
            "finish_reason": getattr(response, "finish_reason", ""),
            "input_tokens": int(getattr(response, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(response, "output_tokens", 0) or 0),
            "cached_tokens": int(getattr(response, "cached_tokens", 0) or 0),
            "reasoning_chars": len(getattr(response, "reasoning_content", "") or ""),
            "tool_calls": [getattr(tc, "name", str(tc)) for tc in tool_calls],
        }
        self.log_span("llm_call", payload)

    def list_logged_jobs(self) -> list[dict[str, Any]]:
        if not self.experiments_file.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.experiments_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows
