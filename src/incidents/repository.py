from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from src.incidents.models import Incident


class IncidentRepository:
    def __init__(self, db_path: str | None = None) -> None:
        raw_path = db_path or os.getenv("INCIDENT_DB_PATH", "./data/incidents.db")
        self.db_path = Path(raw_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    observed_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    observed_sku TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    store_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_observed_at ON incidents(observed_at_utc)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status, review_status)"
            )

    def upsert_incident(self, incident: Incident) -> None:
        payload = incident.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO incidents (
                    incident_id, observed_at_utc, status, review_status,
                    observed_sku, confidence, store_id, camera_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(incident_id) DO UPDATE SET
                    observed_at_utc=excluded.observed_at_utc,
                    status=excluded.status,
                    review_status=excluded.review_status,
                    observed_sku=excluded.observed_sku,
                    confidence=excluded.confidence,
                    store_id=excluded.store_id,
                    camera_id=excluded.camera_id,
                    payload_json=excluded.payload_json
                """,
                (
                    incident.incident_id,
                    incident.observed_at_utc,
                    incident.status.value,
                    incident.review_status.value,
                    incident.observed_sku,
                    incident.confidence,
                    incident.store_id,
                    incident.camera_id,
                    payload,
                ),
            )

    def list_incidents(self) -> list[Incident]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM incidents ORDER BY observed_at_utc ASC"
            ).fetchall()
        incidents: list[Incident] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            incidents.append(Incident.model_validate(payload))
        return incidents

    def get_incident(self, incident_id: str) -> Incident | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM incidents WHERE incident_id = ?",
                (incident_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        return Incident.model_validate(payload)

    def delete_older_than(self, cutoff_iso_utc: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM incidents WHERE observed_at_utc < ?",
                (cutoff_iso_utc,),
            )
            return int(cur.rowcount or 0)
