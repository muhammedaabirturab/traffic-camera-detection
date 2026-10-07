"""Detection history stored in a local SQLite database.

Each analysis gets one row with the summary fields shown in the History table; the
full JSON result (detections, violations, evidence paths) is stored alongside so a
previous analysis can be reopened exactly as it was shown.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    kind            TEXT NOT NULL,          -- 'image' | 'video'
    filename        TEXT NOT NULL,
    status          TEXT NOT NULL,          -- 'processing' | 'completed' | 'failed'
    vehicles        INTEGER DEFAULT 0,
    violations      INTEGER DEFAULT 0,
    violation_types TEXT DEFAULT '[]',      -- JSON list of rule ids
    confidence      REAL,                   -- highest violation confidence (NULL if none)
    processing_time REAL,
    thumbnail       TEXT,
    result_json     TEXT                    -- full API response
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
"""


class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    # ------------------------------------------------------------------ writes
    def create(self, analysis_id: str, kind: str, filename: str, created_at: str) -> None:
        with self._lock, self._connect() as con:
            con.execute("INSERT INTO analyses (id, created_at, kind, filename, status) VALUES (?, ?, ?, ?, 'processing')",
                        (analysis_id, created_at, kind, filename))

    def complete(self, analysis_id: str, result: dict) -> None:
        summary = result.get("summary", {})
        violations = result.get("violations", [])
        with self._lock, self._connect() as con:
            con.execute(
                """UPDATE analyses SET status='completed', vehicles=?, violations=?, violation_types=?,
                   confidence=?, processing_time=?, thumbnail=?, result_json=? WHERE id=?""",
                (
                    summary.get("vehicles", 0),
                    len(violations),
                    json.dumps(sorted(v["rule_id"] for v in violations)),  # one entry per violation
                    max((v["confidence"] for v in violations), default=None),
                    result.get("processing_time"),
                    result.get("media", {}).get("thumbnail"),
                    json.dumps(result),
                    analysis_id,
                ),
            )

    def fail(self, analysis_id: str, error: str) -> None:
        with self._lock, self._connect() as con:
            con.execute("UPDATE analyses SET status='failed', result_json=? WHERE id=?",
                        (json.dumps({"status": "error", "error": error}), analysis_id))

    def delete(self, analysis_id: str) -> bool:
        with self._lock, self._connect() as con:
            cur = con.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d.pop("result_json", None)
        d["violation_types"] = json.loads(d.get("violation_types") or "[]")
        return d

    def list(self, limit: int = 100, offset: int = 0, kind: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM analyses"
        params: list[Any] = []
        if kind:
            q += " WHERE kind=?"
            params.append(kind)
        q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._connect() as con:
            return [self._row(r) for r in con.execute(q, params).fetchall()]

    def get(self, analysis_id: str) -> Optional[dict]:
        with self._connect() as con:
            row = con.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        if row is None:
            return None
        d = self._row(row)
        d["result"] = json.loads(row["result_json"]) if row["result_json"] else None
        return d

    def stats(self) -> dict:
        with self._connect() as con:
            rows = con.execute("SELECT kind, vehicles, violations, violation_types, processing_time, created_at "
                               "FROM analyses WHERE status='completed'").fetchall()
        by_type: dict[str, int] = {}
        for r in rows:
            for t in json.loads(r["violation_types"] or "[]"):
                by_type[t] = by_type.get(t, 0) + 1
        times = [r["processing_time"] for r in rows if r["processing_time"] is not None]
        by_day: dict[str, int] = {}
        for r in rows:
            day = r["created_at"][:10]
            by_day[day] = by_day.get(day, 0) + r["violations"]
        return {
            "analyses": len(rows),
            "images": sum(1 for r in rows if r["kind"] == "image"),
            "videos": sum(1 for r in rows if r["kind"] == "video"),
            "vehicles": sum(r["vehicles"] or 0 for r in rows),
            "violations": sum(r["violations"] or 0 for r in rows),
            "analyses_with_violations": sum(1 for r in rows if (r["violations"] or 0) > 0),
            "violations_by_type": by_type,
            "violations_by_day": dict(sorted(by_day.items())[-14:]),
            "avg_processing_time": round(sum(times) / len(times), 3) if times else None,
        }
