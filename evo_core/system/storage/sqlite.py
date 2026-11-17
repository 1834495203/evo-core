"""Single transactional write boundary for studies, evidence, jobs and commands."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from ..domain import Job, ResearchRecord, Study, document, encode, job_from, now, study_from


class ConflictError(ValueError):
    """A stale proposal or mismatched idempotency key must be reconsidered."""


class SqliteResearchStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES studies(id), body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES studies(id),
                    request_key TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL,
                    UNIQUE(study_id, request_key));
                CREATE TABLE IF NOT EXISTS commands (
                    id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES studies(id),
                    status TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT NOT NULL REFERENCES studies(id), kind TEXT NOT NULL,
                    actor TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_by_study ON events(study_id, sequence);
            """)

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if db.in_transaction:
                db.commit()
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, study_id: str, kind: str, payload: dict, actor: str = "controller") -> None:
        db.execute("INSERT INTO events(study_id,kind,actor,payload,created_at) VALUES(?,?,?,?,?)",
                   (study_id, kind, actor, encode(payload), now()))

    @staticmethod
    def study_in(db, study_id: str) -> Study:
        row = db.execute("SELECT body FROM studies WHERE id=?", (study_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown study: {study_id}")
        return study_from(json.loads(row[0]))

    @staticmethod
    def update_in(db, study: Study) -> Study:
        updated = replace(study, revision=study.revision + 1)
        count = db.execute("UPDATE studies SET revision=?,body=? WHERE id=? AND revision=?",
                           (updated.revision, encode(document(updated)), study.id, study.revision)).rowcount
        if count != 1:
            raise ConflictError("Study changed; refresh the snapshot")
        return updated

    def create(self, study: Study) -> None:
        with self.connection(write=True) as db:
            db.execute("INSERT INTO studies VALUES(?,?,?)", (study.id, study.revision, encode(document(study))))
            self.event(db, study.id, "study.created", document(study))

    def get(self, study_id: str) -> Study:
        with self.connection() as db:
            return self.study_in(db, study_id)

    def append(self, record: ResearchRecord, *, expected_revision: int | None = None) -> Study:
        with self.connection(write=True) as db:
            study = self.study_in(db, record.study_id)
            if expected_revision is not None and study.revision != expected_revision:
                raise ConflictError("Evidence arrived or the plan changed; refresh the snapshot")
            for ref in record.refs:
                row = db.execute("SELECT study_id FROM records WHERE id=? UNION ALL SELECT study_id FROM jobs WHERE id=?",
                                 (ref, ref)).fetchall()
                if not row or any(r[0] != study.id for r in row):
                    raise ValueError(f"Reference is missing or belongs to another study: {ref}")
            db.execute("INSERT INTO records VALUES(?,?,?)", (record.id, study.id, encode(document(record))))
            if record.kind == "decision":
                study = replace(study, next_action=record.details["next_action"])
            updated = self.update_in(db, study)
            self.event(db, study.id, f"research.{record.kind}", document(record))
            return updated

    def revise(self, study: Study, kind: str) -> Study:
        with self.connection(write=True) as db:
            updated = self.update_in(db, study)
            self.event(db, study.id, kind, document(updated))
            return updated

    def enqueue(self, job: Job, *, expected_revision: int | None = None) -> Job:
        with self.connection(write=True) as db:
            study = self.study_in(db, job.study_id)
            old = db.execute("SELECT body FROM jobs WHERE study_id=? AND request_key=?",
                             (study.id, job.request_key)).fetchone()
            if old:
                previous = job_from(json.loads(old[0]))
                if previous.fingerprint != job.fingerprint:
                    raise ConflictError("Idempotency key was already used for different inputs")
                return previous
            if expected_revision is not None and study.revision != expected_revision:
                raise ConflictError("Stale experiment proposal")
            if study.status != "running" or study.plan_version != job.plan_version:
                raise ConflictError("Study is paused or experiment uses an obsolete plan")
            count = db.execute("SELECT count(*) FROM jobs WHERE study_id=?", (study.id,)).fetchone()[0]
            if study.max_experiments is not None and count >= study.max_experiments:
                raise ValueError("Experiment budget exhausted; analyze existing evidence and report")
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?)",
                       (job.id, study.id, job.request_key, job.status, encode(document(job))))
            self.update_in(db, replace(study,
                next_action=f"Collect and interpret experiment {job.id}: {job.spec['purpose']}"))
            self.event(db, study.id, "experiment.queued", {"job_id": job.id, "purpose": job.spec["purpose"]}, "harness")
            return job

    def get_job(self, job_id: str) -> Job:
        with self.connection() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown experiment: {job_id}")
            return job_from(json.loads(row[0]))

    def claim(self, worker_id: str, study_id: str | None = None) -> Job | None:
        with self.connection(write=True) as db:
            rows = db.execute("SELECT body FROM jobs WHERE status='queued' AND (? IS NULL OR study_id=?) ORDER BY rowid",
                              (study_id, study_id)).fetchall()
            for row in rows:
                job = job_from(json.loads(row[0]))
                study = self.study_in(db, job.study_id)
                if study.status != "running":
                    continue
                pending = db.execute("SELECT body FROM commands WHERE study_id=? AND status='pending'",
                                     (study.id,)).fetchall()
                if any(json.loads(c[0])["kind"] in {"pause", "change_direction"} for c in pending):
                    continue
                # A plan change must never silently start an old queued experiment.
                if job.plan_version != study.plan_version:
                    stale = replace(job, status="superseded")
                    db.execute("UPDATE jobs SET status=?,body=? WHERE id=?",
                               (stale.status, encode(document(stale)), stale.id))
                    self.update_in(db, study)
                    self.event(db, study.id, "experiment.superseded", {"job_id": job.id}, "harness")
                    continue
                claimed = replace(job, status="running", worker_id=worker_id)
                db.execute("UPDATE jobs SET status=?,body=? WHERE id=?",
                           (claimed.status, encode(document(claimed)), claimed.id))
                self.update_in(db, study)
                self.event(db, study.id, "experiment.started", {"job_id": job.id}, worker_id)
                return claimed
            return None

    def finish_job(self, job_id: str, worker_id: str, result: dict[str, Any]) -> Job:
        if result.get("status") not in {"succeeded", "failed"}:
            raise ValueError("Worker result must be succeeded or failed")
        with self.connection(write=True) as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            job = job_from(json.loads(row[0]))
            if job.worker_id != worker_id or job.status != "running":
                raise ConflictError("Result is not from the worker that claimed this running job")
            finished = replace(job, status=result["status"], result=result)
            db.execute("UPDATE jobs SET status=?,body=? WHERE id=?",
                       (finished.status, encode(document(finished)), finished.id))
            current = self.study_in(db, job.study_id)
            if current.plan_version == job.plan_version:
                current = replace(current, next_action=f"Experiment {job.id} finished; its results are available.")
            self.update_in(db, current)
            self.event(db, job.study_id, "experiment.finished", {"job_id": job.id, "result": result}, worker_id)
            return finished

    def verify_job(self, job_id: str, check_record: dict[str, Any], worker_id: str = "checker") -> Job:
        with self.connection(write=True) as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown experiment: {job_id}")
            job = job_from(json.loads(row[0]))
            if job.status not in {"succeeded", "failed"}:
                raise ConflictError("Only finished experiments can be verified")
            checks = [*job.checks, check_record]
            passed = bool(check_record.get("passed")) and job.status == "succeeded" and all(
                bool(c.get("passed", False)) for c in checks
            )
            validity = "verified" if passed else "invalid"
            reason = job.invalidated_reason
            if not passed and not reason:
                errs = check_record.get("errors") or []
                reason = "; ".join(str(e) for e in errs) or "Verification check failed"
            updated = replace(job, checks=checks, validity=validity, invalidated_reason=reason)
            db.execute("UPDATE jobs SET body=? WHERE id=?", (encode(document(updated)), updated.id))
            self.update_in(db, self.study_in(db, job.study_id))
            evt_kind = "experiment.verified" if passed else "experiment.check_failed"
            self.event(db, job.study_id, evt_kind, {"job_id": job.id, "validity": validity, "check": check_record}, worker_id)
            return updated

    def invalidate_job(self, job_id: str, reason: str, actor: str = "controller") -> Job:
        if not reason or not reason.strip():
            raise ValueError("Invalidation requires an explicit reason")
        with self.connection(write=True) as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown experiment: {job_id}")
            job = job_from(json.loads(row[0]))
            updated = replace(job, validity="invalid", invalidated_reason=reason.strip())
            db.execute("UPDATE jobs SET body=? WHERE id=?", (encode(document(updated)), updated.id))
            self.update_in(db, self.study_in(db, job.study_id))
            self.event(db, job.study_id, "experiment.invalidated", {"job_id": job.id, "reason": updated.invalidated_reason}, actor)
            return updated

    def list_studies(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT body FROM studies ORDER BY rowid DESC").fetchall()
            return [json.loads(r[0]) for r in rows]

    def snapshot(self, study_id: str) -> dict[str, Any]:
        with self.connection() as db:
            db.execute("BEGIN")
            study = self.study_in(db, study_id)
            data: dict[str, Any] = {"study": document(study), "as_of": now()}
            for table in ("records", "jobs", "commands"):
                data[table] = [json.loads(r[0]) for r in db.execute(
                    f"SELECT body FROM {table} WHERE study_id=? ORDER BY rowid", (study_id,))]
            data["event_cursor"] = db.execute(
                "SELECT coalesce(max(sequence),0) FROM events WHERE study_id=?", (study_id,)).fetchone()[0]
            return data

    def events(self, study_id: str, after: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        if after < 0 or not 1 <= limit <= 1000:
            raise ValueError("Invalid event cursor or page size")
        with self.connection() as db:
            self.study_in(db, study_id)
            rows = db.execute("SELECT * FROM events WHERE study_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                              (study_id, after, limit)).fetchall()
            return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]
