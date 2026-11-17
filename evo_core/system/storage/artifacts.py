"""Archive bytes by hash, and freeze explicitly selected project files."""

from __future__ import annotations

from contextlib import closing
import os
from pathlib import Path, PurePosixPath
import sqlite3
from uuid import uuid4

from ..domain import digest


def relative_path(name: str) -> Path:
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative POSIX file path: {name!r}")
    if not path.parts or path.as_posix() != name:
        raise ValueError("Use one canonical relative file path without ./ or repeated separators")
    return Path(*path.parts)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._publication_lock = self.root / ".publication.sqlite3"
        with closing(sqlite3.connect(self._publication_lock, timeout=10)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS publication_guard (id INTEGER PRIMARY KEY)")
            db.commit()

    def put(self, data: bytes) -> str:
        key = digest(data)
        target = self.root / key
        if target.exists():
            self.read(key)
            return key
        # A cross-process lock avoids replacing a concurrent publisher's object.
        # SQLite works here without requiring filesystem hard-link privileges.
        with closing(sqlite3.connect(self._publication_lock, timeout=10)) as db:
            db.execute("BEGIN IMMEDIATE")
            if target.exists():
                self.read(key)
            else:
                pending = self.root / f".tmp-{uuid4().hex[:12]}"
                try:
                    pending.write_bytes(data)
                    os.replace(pending, target)
                finally:
                    pending.unlink(missing_ok=True)
            db.commit()
        return key

    def read(self, key: str) -> bytes:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact digest")
        data = (self.root / key).read_bytes()
        if digest(data) != key:
            raise ValueError("Archived artifact hash mismatch")
        return data

    def freeze(self, project: Path, files: tuple[str, ...]) -> dict[str, str]:
        project = project.resolve()
        if not files or len(files) != len(set(files)):
            raise ValueError("Declare a nonempty, unique source file list")
        manifest = {}
        total = 0
        for name in files:
            target = project / relative_path(name)
            if not target.resolve().is_relative_to(project) or target.is_symlink():
                raise ValueError("Source file escapes project or is a symlink")
            data = target.read_bytes()
            total += len(data)
            if total > 40_000_000:
                raise ValueError("Development snapshot exceeds 40 MB")
            manifest[name] = self.put(data)
        return manifest

    def materialize(self, manifest: dict[str, str], destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        for name, key in manifest.items():
            target = destination / relative_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.read(key))
