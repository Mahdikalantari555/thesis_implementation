"""Disposable SQLite metadata/result cache."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
              artifact_id TEXT PRIMARY KEY, root_name TEXT NOT NULL, relative_path TEXT NOT NULL,
              fingerprint TEXT NOT NULL, record_json TEXT NOT NULL, last_scan_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scans (
              scan_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, artifact_count INTEGER NOT NULL,
              changed_count INTEGER NOT NULL, error_count INTEGER NOT NULL, manifest_path TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT record_json FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_by_fingerprint(self, artifact_id: str, fingerprint: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT record_json FROM artifacts WHERE artifact_id=? AND fingerprint=?", (artifact_id, fingerprint)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_artifact(self, record: dict[str, Any], scan_id: str) -> None:
        self.connection.execute(
            "INSERT INTO artifacts(artifact_id,root_name,relative_path,fingerprint,record_json,last_scan_id) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(artifact_id) DO UPDATE SET "
            "root_name=excluded.root_name, relative_path=excluded.relative_path, fingerprint=excluded.fingerprint, "
            "record_json=excluded.record_json, last_scan_id=excluded.last_scan_id",
            (record["id"], record["root"], record["relative_path"], record["fingerprint"],
             json.dumps(record, ensure_ascii=False, allow_nan=False), scan_id),
        )

    def list_artifacts(self, scan_id: str | None = None) -> list[dict[str, Any]]:
        if scan_id is None:
            rows = self.connection.execute("SELECT record_json FROM artifacts ORDER BY relative_path")
        else:
            rows = self.connection.execute("SELECT record_json FROM artifacts WHERE last_scan_id=? ORDER BY relative_path", (scan_id,))
        return [json.loads(row[0]) for row in rows]

    def put_scan(self, scan: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO scans(scan_id,created_at,artifact_count,changed_count,error_count,manifest_path) VALUES(?,?,?,?,?,?)",
            (scan["scan_id"], scan["created_at"], scan["artifact_count"], scan["changed_count"],
             scan["error_count"], scan["manifest_path"]),
        )

    def list_scans(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM scans ORDER BY created_at DESC")
        return [dict(row) for row in rows]

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
