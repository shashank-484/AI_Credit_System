"""Append-only SQLite decision log with a verifiable hash chain."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "audit.sqlite"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def session():
    db = connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize() -> None:
    with session() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            values_json TEXT NOT NULL,
            probability REAL NOT NULL,
            predicted_bad INTEGER NOT NULL,
            threshold REAL NOT NULL,
            model_version TEXT NOT NULL,
            explanations_json TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT NOT NULL REFERENCES decisions(id),
            created_at TEXT NOT NULL,
            action TEXT NOT NULL,
            note TEXT NOT NULL,
            previous_hash TEXT NOT NULL,
            event_hash TEXT NOT NULL
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions(created_at DESC)")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_event(db: sqlite3.Connection, decision_id: str, action: str, note: str) -> None:
    latest = db.execute("SELECT event_hash FROM audit_events ORDER BY id DESC LIMIT 1").fetchone()
    previous = latest["event_hash"] if latest else "0" * 64
    created = now_iso()
    payload = json.dumps(
        {"decision_id": decision_id, "created_at": created, "action": action, "note": note},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256((previous + payload).encode("utf-8")).hexdigest()
    db.execute(
        "INSERT INTO audit_events(decision_id, created_at, action, note, previous_hash, event_hash) VALUES (?, ?, ?, ?, ?, ?)",
        (decision_id, created, action, note, previous, digest),
    )


def save_decision(values: dict[str, Any], probability: float, predicted_bad: bool, threshold: float,
                  model_version: str, explanations: list[dict[str, Any]]) -> str:
    decision_id = str(uuid4())
    with session() as db:
        db.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (decision_id, now_iso(), json.dumps(values, ensure_ascii=False, sort_keys=True), probability,
             int(predicted_bad), threshold, model_version, json.dumps(explanations, ensure_ascii=False)),
        )
        _append_event(db, decision_id, "prediction_created", f"model={model_version}; predicted_bad={int(predicted_bad)}")
    return decision_id


def add_review(decision_id: str, action: str, note: str) -> None:
    with session() as db:
        exists = db.execute("SELECT 1 FROM decisions WHERE id = ?", (decision_id,)).fetchone()
        if not exists:
            raise KeyError(decision_id)
        _append_event(db, decision_id, action, note)


def recent_decisions(limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with session() as db:
        rows = db.execute(
            "SELECT d.*, (SELECT e.action FROM audit_events e WHERE e.decision_id=d.id ORDER BY e.id DESC LIMIT 1) AS latest_action "
            "FROM decisions d ORDER BY d.created_at DESC LIMIT ?", (limit,),
        ).fetchall()
        return [{
            "id": row["id"], "created_at": row["created_at"], "probability": row["probability"],
            "predicted_bad": bool(row["predicted_bad"]), "threshold": row["threshold"],
            "model_version": row["model_version"], "review_status": row["latest_action"],
            "values": json.loads(row["values_json"]), "explanations": json.loads(row["explanations_json"]),
        } for row in rows]


def recent_inputs(limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    with session() as db:
        rows = db.execute("SELECT values_json FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(row["values_json"]) for row in rows]


def verify_integrity() -> dict[str, Any]:
    expected_previous = "0" * 64
    verified = 0
    with session() as db:
        rows = db.execute("SELECT * FROM audit_events ORDER BY id").fetchall()
    for row in rows:
        payload = json.dumps(
            {"decision_id": row["decision_id"], "created_at": row["created_at"],
             "action": row["action"], "note": row["note"]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        calculated = hashlib.sha256((expected_previous + payload).encode("utf-8")).hexdigest()
        if row["previous_hash"] != expected_previous or row["event_hash"] != calculated:
            return {"valid": False, "verified_events": verified, "failed_event_id": row["id"]}
        expected_previous = row["event_hash"]
        verified += 1
    return {"valid": True, "verified_events": verified, "head_hash": expected_previous}
