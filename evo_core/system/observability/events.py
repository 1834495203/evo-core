"""Transport-independent event feed; no busy polling thread or web server."""

from __future__ import annotations

from ..storage.sqlite import SqliteResearchStore


class ResearchEvents:
    def __init__(self, store: SqliteResearchStore):
        self._store = store

    def read(self, study_id: str, after: int = 0, limit: int = 100) -> dict:
        events = self._store.events(study_id, after, limit)
        return {"events": events, "next_cursor": events[-1]["sequence"] if events else after}
