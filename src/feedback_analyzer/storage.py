"""Short-lived SQLite transactions and persistent claims before any paid work."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from pydantic import ValidationError

from feedback_analyzer.batch_contracts import BatchRow, Outcome, utc_now


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    action: str
    owner: str | None = None
    outcome: Outcome | None = None


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS outcomes (
        identity TEXT PRIMARY KEY, ticket_key TEXT NOT NULL, ticket_id TEXT,
        input_hash TEXT NOT NULL, config_hash TEXT NOT NULL, pipeline_version TEXT NOT NULL,
        original_message TEXT, normalized_message TEXT, raw_record TEXT,
        status TEXT NOT NULL CHECK(status IN ('processed','ignored','error')),
        category TEXT, priority TEXT, sentiment TEXT, product_feature TEXT, suggested_reply TEXT,
        rejection_reason TEXT, error_stage TEXT, error_code TEXT, error_message TEXT,
        input_truncated INTEGER NOT NULL, model_metadata_json TEXT NOT NULL, outcome_json TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        CHECK(status != 'processed' OR (category IS NOT NULL AND category != 'Non-Qualifying'
          AND priority IS NOT NULL AND sentiment IS NOT NULL AND suggested_reply IS NOT NULL
          AND error_message IS NULL)),
        CHECK(status != 'ignored' OR rejection_reason IS NOT NULL),
        CHECK(status != 'error' OR error_message IS NOT NULL))""",
    "CREATE INDEX IF NOT EXISTS outcomes_ticket_key ON outcomes(ticket_key)",
    """CREATE TABLE IF NOT EXISTS claims (
        identity TEXT PRIMARY KEY, ticket_key TEXT NOT NULL, owner TEXT NOT NULL, started_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS batch_runs (
        run_id TEXT PRIMARY KEY, source_name TEXT NOT NULL, source_hash TEXT NOT NULL,
        config_hash TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
        file_errors_json TEXT NOT NULL, persistence_errors_json TEXT NOT NULL DEFAULT '[]',
        total_count INTEGER, processed_count INTEGER, ignored_count INTEGER, error_count INTEGER)""",
    """CREATE TABLE IF NOT EXISTS batch_rows (
        run_id TEXT NOT NULL REFERENCES batch_runs(run_id), ordinal INTEGER NOT NULL,
        source_row INTEGER NOT NULL, source_line_end INTEGER NOT NULL, identity TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('processed','ignored','error')),
        reused INTEGER NOT NULL, recorded_at TEXT NOT NULL, row_json TEXT NOT NULL,
        PRIMARY KEY(run_id,ordinal))""",
)


class OutcomeStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as db:
                current = db.execute("PRAGMA user_version").fetchone()[0]
                tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                names = {row[0] for row in tables}
                if (current not in (0, 1) or (current == 0 and tables)
                        or (current == 1 and not {"outcomes", "claims", "batch_runs", "batch_rows"}.issubset(names))):
                    raise StorageError("Unsupported database schema. Choose a new database or restore a compatible backup.")
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("BEGIN IMMEDIATE")
                for statement in SCHEMA:
                    db.execute(statement)
                db.execute("PRAGMA user_version=1")
                db.commit()
        except OSError:
            raise StorageError("Database directory is unavailable. Check the output path and permissions.") from None

    @contextmanager
    def _connection(self):
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db
        except sqlite3.Error:
            raise StorageError("SQLite could not safely read/write this operation. Check access, disk space and database integrity.") from None
        finally:
            if db is not None:
                # Closing also rolls back an unfinished transaction. Never share connections between threads.
                db.close()

    def start_run(self, source_name, source_hash, config_hash, file_errors) -> tuple[str, str]:
        run_id, started = uuid4().hex, utc_now()
        with self._connection() as db:
            db.execute("INSERT INTO batch_runs(run_id,source_name,source_hash,config_hash,started_at,file_errors_json) VALUES(?,?,?,?,?,?)",
                       (run_id, source_name, source_hash, config_hash, started, json.dumps(file_errors)))
        return run_id, started

    def reserve(self, identity, ticket_key, *, reprocess=False, retry_errors=False, recover_incomplete=False) -> Reservation:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            saved = db.execute("SELECT outcome_json FROM outcomes WHERE identity=?", (identity,)).fetchone()
            try:
                outcome = Outcome.model_validate_json(saved[0]) if saved else None
            except (ValidationError, ValueError):
                raise StorageError("Stored outcome failed validation. No automatic model/API retry was made.") from None
            if outcome and (outcome.status != "error" or not retry_errors):
                db.commit()
                return Reservation("reuse", outcome=outcome)
            claim = db.execute("SELECT owner FROM claims WHERE identity=?", (identity,)).fetchone()
            if claim and not recover_incomplete:
                db.commit()
                return Reservation("busy")
            # A previous identity for this validated ticket needs explicit authorization.
            other = db.execute("SELECT 1 FROM outcomes WHERE ticket_key=? AND identity!=? UNION ALL SELECT 1 FROM claims WHERE ticket_key=? AND identity!=? LIMIT 1",
                               (ticket_key, identity, ticket_key, identity)).fetchone()
            if ticket_key.startswith("ticket:") and other and not reprocess:
                db.commit()
                return Reservation("changed")
            if claim:
                db.execute("DELETE FROM claims WHERE identity=?", (identity,))
            owner = uuid4().hex
            db.execute("INSERT INTO claims(identity,ticket_key,owner,started_at) VALUES(?,?,?,?)", (identity, ticket_key, owner, utc_now()))
            db.commit()
            return Reservation("run", owner=owner)

    @staticmethod
    def _record_row(db, run_id, row):
        db.execute("INSERT INTO batch_rows(run_id,ordinal,source_row,source_line_end,identity,status,reused,recorded_at,row_json) VALUES(?,?,?,?,?,?,?,?,?)",
                   (run_id, row.ordinal, row.source_row, row.source_line_end, row.identity, row.outcome.status,
                    int(row.reused), row.recorded_at, row.model_dump_json()))

    def record_row(self, run_id: str, row: BatchRow):
        row = BatchRow.model_validate(row.model_dump())
        with self._connection() as db:
            self._record_row(db, run_id, row)

    def finish_row(self, run_id: str, row: BatchRow, owner: str, metadata: dict):
        row = BatchRow.model_validate(row.model_dump())
        outcome = row.outcome
        analysis = outcome.analysis
        category = analysis.category if analysis else None
        sentiment = analysis.sentiment if analysis else None
        generated = analysis.generated.details if analysis and analysis.generated else None
        error = outcome.error
        now = utc_now()
        values = {
            "identity": row.identity, "ticket_key": row.ticket_key, "ticket_id": outcome.ticket_id,
            "input_hash": row.input_hash, "config_hash": row.config_hash,
            "pipeline_version": metadata["pipeline_version"], "original_message": outcome.original_message,
            "normalized_message": outcome.normalized_message, "raw_record": outcome.raw_record, "status": outcome.status,
            "category": category.category if category else None, "priority": generated.priority if generated else None,
            "sentiment": sentiment.sentiment if sentiment else None,
            "product_feature": generated.product_feature if generated else None,
            "suggested_reply": generated.suggested_reply if generated else None,
            "rejection_reason": outcome.reason, "error_stage": error.stage if error else None,
            "error_code": error.code if error else None, "error_message": error.message if error else None,
            "input_truncated": int(bool((category and category.truncated) or (sentiment and sentiment.truncated))),
            "model_metadata_json": json.dumps(metadata, sort_keys=True), "outcome_json": outcome.model_dump_json(),
            "created_at": now, "updated_at": now,
        }
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            claim = db.execute("SELECT owner FROM claims WHERE identity=?", (row.identity,)).fetchone()
            if not claim or claim[0] != owner:
                raise StorageError("Work reservation changed. Result was not stored; do not retry automatically.")
            # Only column names owned by this module enter SQL; all customer values are bound parameters.
            columns = list(values)
            updates = ",".join(f"{name}=excluded.{name}" for name in columns if name not in ("identity", "created_at"))
            db.execute(f"INSERT INTO outcomes({','.join(columns)}) VALUES({','.join('?' for _ in columns)}) ON CONFLICT(identity) DO UPDATE SET {updates}", list(values.values()))
            self._record_row(db, run_id, row)
            db.execute("DELETE FROM claims WHERE identity=? AND owner=?", (row.identity, owner))
            db.commit()

    def finish_run(self, result):
        counts = result.counts
        with self._connection() as db:
            db.execute("UPDATE batch_runs SET finished_at=?,total_count=?,processed_count=?,ignored_count=?,error_count=?,persistence_errors_json=? WHERE run_id=?",
                       (result.finished_at, len(result.rows), counts["processed"], counts["ignored"], counts["error"],
                        json.dumps(result.persistence_errors), result.run_id))
