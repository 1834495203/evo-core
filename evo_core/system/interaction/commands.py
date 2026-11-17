"""Commands are applied transactionally at decision boundaries, not into chat history."""

from __future__ import annotations

from dataclasses import replace
import json

from ..domain import encode, identity, now
from ..storage.sqlite import SqliteResearchStore


class UserCommandInbox:
    def __init__(self, store: SqliteResearchStore):
        self.store = store

    def send(self, study_id: str, kind: str, text: str = "", *, expected_plan_version: int | None = None) -> str:
        if kind not in {"suggestion", "change_direction", "pause", "resume"}:
            raise ValueError("Unknown user command")
        if kind in {"suggestion", "change_direction"} and not text.strip():
            raise ValueError("A suggestion or new direction requires text")
        command_id = identity("cmd")
        body = {"id": command_id, "study_id": study_id, "kind": kind, "text": text,
                "expected_plan_version": expected_plan_version, "status": "pending", "created_at": now()}
        with self.store.connection(write=True) as db:
            self.store.study_in(db, study_id)
            db.execute("INSERT INTO commands VALUES(?,?,?,?)", (command_id, study_id, "pending", encode(body)))
            self.store.event(db, study_id, "user.command_received", body, "user")
        return command_id

    def apply_pending(self, study_id: str) -> list[dict]:
        handled = []
        with self.store.connection(write=True) as db:
            study = self.store.study_in(db, study_id)
            rows = db.execute("SELECT body FROM commands WHERE study_id=? AND status='pending' ORDER BY rowid",
                              (study_id,)).fetchall()
            for row in rows:
                command = json.loads(row[0])
                expected = command["expected_plan_version"]
                if expected is not None and expected != study.plan_version:
                    command.update(status="rejected", reason="Plan changed; review the new direction")
                else:
                    kind = command["kind"]
                    if kind == "change_direction":
                        study = replace(study, direction=command["text"], plan_version=study.plan_version + 1,
                                        next_action="Consider the updated direction.")
                    elif kind in {"pause", "resume"}:
                        study = replace(study, status="paused" if kind == "pause" else "running")
                    # A suggestion is delivered for consideration; it is not automatically accepted.
                    command.update(status="delivered" if kind == "suggestion" else "applied",
                                   applied_plan_version=study.plan_version)
                    study = self.store.update_in(db, study)
                command["handled_at"] = now()
                db.execute("UPDATE commands SET status=?,body=? WHERE id=?",
                           (command["status"], encode(command), command["id"]))
                self.store.event(db, study_id, "user.command_handled", command)
                handled.append(command)
        return handled
