"""Detection history in SQLite (local file, no server needed)."""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    filename TEXT NOT NULL,
    vehicles INTEGER NOT NULL,
    violations INTEGER NOT NULL,
    violation_types TEXT NOT NULL,
    confidence REAL NOT NULL,
    processing_time REAL NOT NULL,
    risk_level TEXT,
    result_json TEXT NOT NULL
);
"""


class HistoryStore:
    def __init__(self, db_file: Path, media_root: Path):
        self.db_file, self.media_root = db_file, media_root
        self.lock = threading.Lock()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_file, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, result: dict) -> None:
        s = result.get("summary", {})
        row = (
            result["id"], result["created_at"], result["kind"], result["filename"],
            int(s.get("unique_vehicles", s.get("vehicles_detected", 0))), int(s.get("possible_violations", 0)),
            json.dumps(s.get("violation_types", [])), float(result.get("confidence", 0.0)),
            float(result.get("processing_time", 0.0)), result.get("intelligence", {}).get("risk_level"),
            json.dumps(result),
        )
        with self.lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)

    def list(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,created_at,kind,filename,vehicles,violations,violation_types,confidence,processing_time,risk_level "
                "FROM analyses ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [{**dict(r), "violation_types": json.loads(r["violation_types"])} for r in rows]

    def get(self, aid: str) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute("SELECT result_json FROM analyses WHERE id=?", (aid,)).fetchone()
        return json.loads(r["result_json"]) if r else None

    def delete(self, aid: str) -> bool:
        with self.lock, self._conn() as c:
            cur = c.execute("DELETE FROM analyses WHERE id=?", (aid,))
        shutil.rmtree(self.media_root / aid, ignore_errors=True)
        return cur.rowcount > 0

    def stats(self) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT COUNT(*) n, COALESCE(SUM(violations),0) v, COALESCE(SUM(vehicles),0) veh, "
                          "COALESCE(AVG(confidence),0) conf FROM analyses").fetchone()
            types = c.execute("SELECT violation_types FROM analyses WHERE violations>0").fetchall()
        by_type: dict[str, int] = {}
        for t in types:
            for k in json.loads(t["violation_types"]):
                by_type[k] = by_type.get(k, 0) + 1
        return {"analyses": r["n"], "violations": r["v"], "vehicles": r["veh"],
                "average_confidence": round(r["conf"], 4), "violations_by_type": by_type}
