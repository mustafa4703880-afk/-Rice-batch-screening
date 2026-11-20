import sqlite3
from contextlib import closing
from pathlib import Path

DB = Path(__file__).resolve().parent / "data" / "batch_history.sqlite3"


def save_batch(batch, decision, notes, report_csv):
    with closing(sqlite3.connect(DB)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY,
                lot TEXT NOT NULL, supplier TEXT NOT NULL, batch_date TEXT NOT NULL,
                variety TEXT NOT NULL, profile TEXT NOT NULL, threshold INTEGER NOT NULL,
                status TEXT NOT NULL, decision TEXT NOT NULL, notes TEXT NOT NULL, report TEXT NOT NULL,
                UNIQUE (lot, supplier, batch_date)
            )
        """)
        db.execute("""
            INSERT INTO batches (lot, supplier, batch_date, variety, profile, threshold,
                                 status, decision, notes, report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (batch["lot"], batch["supplier"], str(batch["date"]), batch["declared"],
              batch["profile"], batch["threshold"], batch["status"], decision, notes, report_csv))


def read_batches():
    if not DB.exists():
        return []
    with closing(sqlite3.connect(DB)) as db:
        db.row_factory = sqlite3.Row
        return db.execute("""
            SELECT id, lot, supplier, batch_date, variety, profile, threshold, status, decision, notes
            FROM batches ORDER BY batch_date DESC, id DESC
        """).fetchall()


def read_report(batch_id):
    with closing(sqlite3.connect(DB)) as db:
        return db.execute("SELECT report FROM batches WHERE id = ?", (batch_id,)).fetchone()[0]
