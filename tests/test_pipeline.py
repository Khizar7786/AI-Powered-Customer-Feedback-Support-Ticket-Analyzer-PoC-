"""Batch accounting and persistent reuse, with explicit model/API doubles."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import sqlite3
from threading import Event

import pytest

from feedback_analyzer.pipeline import analyze_csv
from feedback_analyzer.batch_contracts import Outcome
from pydantic import ValidationError
from feedback_analyzer.pipeline_identity import PipelineIdentity
from feedback_analyzer.storage import OutcomeStore, StorageError


def query(path, sql):
    with closing(sqlite3.connect(path)) as db, db:
        return db.execute(sql).fetchall()


def test_mixed_batch_accounts_for_each_record_original_text_and_partial_failures(tmp_path, batch_setup):
    s = batch_setup
    s.failures.add("api fails")
    data = b'ticket_id,message\nA,"  Hi,\ncheckout crashes  "\nB,Hi!\nC,social greeting\nD,api fails\nE,sentiment fails\nF,category fails\n,Missing ID\nZ,extra,field\n'
    path = tmp_path / "tickets.db"
    result = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    assert result.counts == {"processed": 1, "ignored": 2, "error": 5}
    assert sum(result.counts.values()) == len(result.rows) == 8
    assert result.rows[0].source_row == 2 and result.rows[0].source_line_end == 3
    assert result.rows[0].outcome.original_message == "  Hi,\ncheckout crashes  "
    assert s.calls["api"] == ["  Hi,\ncheckout crashes  ", "api fails"]
    api_error = result.rows[3].outcome
    assert api_error.analysis.category and api_error.analysis.sentiment and not api_error.analysis.generated
    assert result.fully_processed == 1 and all(row.persisted for row in result.rows)
    assert query(path, "SELECT count(*) FROM outcomes") == [(8,)]
    assert query(path, "SELECT total_count,processed_count,ignored_count,error_count FROM batch_runs") == [(8, 1, 2, 5)]
    saved = query(path, "SELECT category,sentiment,priority,error_code FROM outcomes WHERE ticket_id='D'")
    assert saved == [("Bug Report", "Negative", None, "refusal")]
    assert "private" not in result.model_dump_json()
    metadata = query(path, "SELECT input_hash,config_hash,pipeline_version,created_at,updated_at,model_metadata_json FROM outcomes WHERE ticket_id='A'")[0]
    assert len(metadata[0]) == len(metadata[1]) == 64 and metadata[2] == "unit-batch-v1"
    assert metadata[3] and metadata[4] and json.loads(metadata[5])["model"] == "explicit-mock"


def test_matching_success_and_ignore_survive_new_store_runtime_and_source_position(tmp_path, batch_setup):
    s, path = batch_setup, tmp_path / "tickets.db"
    first = analyze_csv(b"ticket_id,message\nA,App crashes\nB,Hello\n", s.runtime, OutcomeStore(path), s.identity)
    assert first.fully_processed == 1
    before = {key: list(value) for key, value in s.calls.items()}
    second = analyze_csv(b"message,ticket_id\nHello,B\nApp crashes,A\n", s.runtime, OutcomeStore(path), s.identity)
    assert second.reused_count == 2 and s.calls == before
    assert second.rows[0].outcome.ticket_id == "B" and second.rows[0].source_row == 2
    assert query(path, "SELECT count(*) FROM outcomes") == [(2,)]
    assert query(path, "SELECT count(*) FROM batch_rows") == [(4,)]  # Deliberate run audit, not duplicate analyses.
    assert query(path, "SELECT count(*) FROM claims") == [(0,)]


@pytest.mark.parametrize("change", ["message", "config", "whitespace"])
def test_changed_input_or_config_requires_explicit_reprocessing(tmp_path, batch_setup, change):
    s, path = batch_setup, tmp_path / "tickets.db"
    store = OutcomeStore(path)
    data, identity = b"ticket_id,message\nA,App crashes\n", s.identity
    analyze_csv(data, s.runtime, store, identity)
    if change == "config":
        identity = PipelineIdentity(metadata={"pipeline_version": "unit-batch-v2"})
    else:
        data = b"ticket_id,message\nA,New issue\n" if change == "message" else b"ticket_id,message\nA, App crashes \n"
    blocked = analyze_csv(data, s.runtime, store, identity)
    assert blocked.counts["error"] == 1 and blocked.rows[0].outcome.error.code == "reprocess_required"
    assert len(s.calls["api"]) == 1
    allowed = analyze_csv(data, s.runtime, store, identity, reprocess=True)
    assert allowed.fully_processed == 1 and len(s.calls["api"]) == 2
    repeated = analyze_csv(data, s.runtime, store, identity, reprocess=True, retry_errors=True)
    assert repeated.reused_count == 1 and len(s.calls["api"]) == 2
    assert query(path, "SELECT count(*) FROM outcomes") == [(2,)]


def test_error_retry_is_explicit_and_does_not_repeat_other_successes(tmp_path, batch_setup):
    s, path = batch_setup, tmp_path / "tickets.db"
    data = b"ticket_id,message\nA,api fails\nB,App crashes\n"
    s.failures.add("api fails")
    first = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    assert first.counts == {"processed": 1, "ignored": 0, "error": 1}
    s.failures.clear()
    again = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    assert again.reused_count == 2 and len(s.calls["api"]) == 2
    retried = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity, retry_errors=True)
    assert retried.fully_processed == 2 and retried.reused_count == 1 and len(s.calls["api"]) == 3
    assert query(path, "SELECT count(*) FROM outcomes") == [(2,)]


def test_duplicate_and_missing_ids_persist_separately_and_bypass_all_models(tmp_path, batch_setup):
    s, path = batch_setup, tmp_path / "tickets.db"
    data = b"ticket_id,message\nA,one\nA,two\n,three\n,four\n"
    result = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    assert result.counts["error"] == 4 and len({row.identity for row in result.rows}) == 4
    assert not any(s.calls.values())
    assert query(path, "SELECT count(*) FROM outcomes") == [(4,)]
    corrected = analyze_csv(b"ticket_id,message\nA,App crashes\n", s.runtime, OutcomeStore(path), s.identity)
    assert corrected.fully_processed == 1  # Invalid duplicate IDs do not claim a valid ticket's identity.


@pytest.mark.parametrize("data,rows", [(b"ticket_id,message\nA,App crashes\nB,\"broken\n", 2), (b"wrong,columns\nA,issue\n", 0), (b"\xff", 0)])
def test_file_errors_never_trigger_api_and_preserve_recoverable_record_accounting(tmp_path, batch_setup, data, rows):
    s, path = batch_setup, tmp_path / "tickets.db"
    result = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    assert result.file_errors and len(result.rows) == rows and result.counts["error"] == rows
    assert not any(s.calls.values())
    assert query(path, "SELECT count(*) FROM batch_rows") == [(rows,)]


def test_concurrent_same_input_has_one_paid_operation(tmp_path, batch_setup):
    s, path = batch_setup, tmp_path / "tickets.db"
    entered, release = Event(), Event()
    original = s.provider.generate

    def waiting(context):
        entered.set()
        assert release.wait(10)
        return original(context)

    s.provider.generate = waiting
    data = b"ticket_id,message\nA,App crashes\n"
    store = OutcomeStore(path)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(analyze_csv, data, s.runtime, store, s.identity)
        try:
            assert entered.wait(10)
            second = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
            assert second.rows[0].outcome.error.code == "in_progress"
        finally:
            release.set()
        assert first.result(10).fully_processed == 1
    assert len(s.calls["api"]) == 1 and query(path, "SELECT count(*) FROM outcomes") == [(1,)]


def test_commit_failure_rolls_back_result_keeps_claim_and_continues(tmp_path, batch_setup):
    s, path = batch_setup, tmp_path / "tickets.db"
    store = OutcomeStore(path)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TRIGGER fail_first BEFORE INSERT ON batch_rows WHEN NEW.ordinal=1 BEGIN SELECT RAISE(ABORT,'simulated disk problem'); END")
    data = b"ticket_id,message\nA,First issue\nB,Second issue\n"
    result = analyze_csv(data, s.runtime, store, s.identity)
    assert result.counts == {"processed": 1, "ignored": 0, "error": 1}
    assert not result.rows[0].persisted and result.rows[0].outcome.analysis.generated
    assert result.persistence_errors and len(s.calls["api"]) == 2
    assert query(path, "SELECT ticket_id FROM outcomes") == [("B",)]
    assert query(path, "SELECT count(*) FROM claims") == [(1,)]
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TRIGGER fail_first")
    again = analyze_csv(data, s.runtime, store, s.identity)
    assert again.rows[0].outcome.error.code == "in_progress" and len(s.calls["api"]) == 2
    recovered = analyze_csv(data, s.runtime, store, s.identity, recover_incomplete=True)
    assert recovered.fully_processed == 2 and len(s.calls["api"]) == 3


def test_sql_values_are_bound_and_connections_close(tmp_path, batch_setup):
    s, path = batch_setup, tmp_path / "tickets.db"
    data = b'ticket_id,message\n"A\'; DROP TABLE outcomes; --"," App crashes; DROP TABLE outcomes; -- "\n'
    analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    assert query(path, "SELECT original_message FROM outcomes") == [(" App crashes; DROP TABLE outcomes; -- ",)]
    # No cached/shared connection prevents exclusive access or moving the database.
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.execute("BEGIN EXCLUSIVE")
        db.execute("ROLLBACK")
    path.rename(tmp_path / "moved.db")


def test_corrupt_stored_success_fails_closed_without_paid_retry(tmp_path, batch_setup):
    s, path = batch_setup, tmp_path / "tickets.db"
    data = b"ticket_id,message\nA,App crashes\n"
    analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE outcomes SET outcome_json='{}'")
    result = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity, retry_errors=True)
    assert result.rows[0].outcome.error.stage == "storage" and len(s.calls["api"]) == 1


def test_unrelated_database_is_not_modified(tmp_path):
    path = tmp_path / "other.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE unrelated(value TEXT)")
    with pytest.raises(StorageError, match="Unsupported database"):
        OutcomeStore(path)
    assert query(path, "SELECT name FROM sqlite_master WHERE type='table'") == [("unrelated",)]


def test_valid_but_mismatched_cached_text_is_rejected_without_api_work(tmp_path, batch_setup):
    s, path = batch_setup, tmp_path / "tickets.db"
    data = b"ticket_id,message\nA,App crashes\n"
    analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    with closing(sqlite3.connect(path)) as db, db:
        value = json.loads(db.execute("SELECT outcome_json FROM outcomes").fetchone()[0])
        value["original_message"] = value["normalized_message"] = value["analysis"]["ticket"]["message"] = "Other issue"
        db.execute("UPDATE outcomes SET outcome_json=?", (json.dumps(value),))
    result = analyze_csv(data, s.runtime, OutcomeStore(path), s.identity)
    assert result.rows[0].outcome.error.stage == "storage" and len(s.calls["api"]) == 1


def test_false_complete_outcome_is_rejected_before_persistence():
    with pytest.raises(ValidationError, match="requires matching validated analysis"):
        Outcome(ticket_id="A", original_message="App crashes", normalized_message="App crashes", status="processed")


def test_provider_config_drift_is_an_error_before_request(tmp_path, batch_setup):
    s = batch_setup
    s.provider.public_options = {"model": "changed-model"}
    identity = PipelineIdentity(metadata={"pipeline_version": "test", "generation": {"model": "expected-model"}})
    result = analyze_csv(b"ticket_id,message\nA,App crashes\n", s.runtime, OutcomeStore(tmp_path / "tickets.db"), identity)
    assert result.rows[0].outcome.error.code == "configuration_changed"
    assert result.rows[0].outcome.analysis.sentiment and not s.calls["api"]
