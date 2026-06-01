"""
db_service.py — SQLite-backed patient history & prediction logging.
Uses only stdlib sqlite3, no extra dependencies required.
"""
import os
import sqlite3
import json
from datetime import datetime

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_BASE, "data", "predictions.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                disease     TEXT    NOT NULL,
                patient_id  TEXT,
                inputs      TEXT    NOT NULL,   -- JSON
                prediction  INTEGER NOT NULL,
                probability REAL    NOT NULL,
                risk_level  TEXT    NOT NULL,
                explanation TEXT    NOT NULL,   -- JSON
                created_at  TEXT    NOT NULL
            )
        """)
        conn.commit()


def log_prediction(
    disease: str,
    patient_id: str,
    inputs: dict,
    prediction: int,
    probability: float,
    risk_level: str,
    explanation: list,
) -> int:
    """Insert one prediction record and return its row id."""
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (disease, patient_id, inputs, prediction, probability, risk_level, explanation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                disease,
                patient_id or "anonymous",
                json.dumps(inputs),
                prediction,
                round(probability, 6),
                risk_level,
                json.dumps(explanation),
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_history(disease: str | None = None, limit: int = 50) -> list[dict]:
    """Return recent predictions, optionally filtered by disease."""
    # BUG FIX #7: sqlite3.Row objects become invalid after the connection closes.
    # Fetch and convert to plain dicts INSIDE the with block, before conn closes.
    with _get_conn() as conn:
        if disease:
            raw_rows = conn.execute(
                "SELECT * FROM predictions WHERE disease = ? ORDER BY id DESC LIMIT ?",
                (disease, limit),
            ).fetchall()
        else:
            raw_rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        # Convert to plain dicts while connection is still open
        result = []
        for row in raw_rows:
            d = dict(row)
            d["inputs"] = json.loads(d["inputs"])
            d["explanation"] = json.loads(d["explanation"])
            result.append(d)

    return result


def get_stats() -> dict:
    """Return aggregate counts per disease and risk level."""
    # BUG FIX #7: same fix — read all data inside the with block
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT disease, risk_level, COUNT(*) as cnt
            FROM predictions
            GROUP BY disease, risk_level
            """
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        stats: dict = {"total": total, "by_disease": {}}
        for row in rows:
            d = row["disease"]
            if d not in stats["by_disease"]:
                stats["by_disease"][d] = {}
            stats["by_disease"][d][row["risk_level"]] = row["cnt"]

    return stats


# Initialise on import
init_db()
