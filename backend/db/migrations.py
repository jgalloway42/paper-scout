"""Run-once migration runner. Executes schema.sql against the configured DB path."""

import sqlite3
from pathlib import Path


def run(db_path: str | None = None) -> None:
    """Apply schema.sql to the database at db_path.

    If db_path is None, reads PAPER_SCOUT_DB_PATH env var (falls back to
    'paper_scout.db'). Safe to call repeatedly — all statements use
    CREATE TABLE IF NOT EXISTS.
    """
    if db_path is None:
        import os
        db_path = os.environ.get("PAPER_SCOUT_DB_PATH", "paper_scout.db")

    schema_path = Path(__file__).parent / "schema.sql"
    ddl = schema_path.read_text()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(ddl)
        conn.commit()
    finally:
        conn.close()
