"""Acceptance runner checks explicit references without using a real model, API or held-out set."""

from contextlib import closing
import json
import sqlite3

import pytest

from feedback_analyzer.acceptance import run_acceptance, verify_exports
from feedback_analyzer.config import load_config
from feedback_analyzer.exports import build_exports, save_exports
from feedback_analyzer.pipeline import analyze_csv
from feedback_analyzer.storage import OutcomeStore


def fixtures(tmp_path):
    folder = tmp_path / "fixtures"
    folder.mkdir()
    (folder / "assessment.csv").write_bytes(b"ticket_id,message\nA,App crashes\n")
    (folder / "mixed.csv").write_bytes(b"ticket_id,message\nB,api fails\nC,Hello\n,Missing ID\n")
    reference = {"authorship": "Explicit unit fixtures; not live evidence or human labels.",
        "assessment": {"expected_counts": {"processed": 1, "ignored": 0, "error": 0},
                       "rows": [{"ordinal": 1, "status": "processed", "category": "Bug Report", "priority": "Medium", "sentiment": "Negative"}]},
        "mixed": {"expected_counts": {"processed": 0, "ignored": 1, "error": 2},
                  "rows": [{"ordinal": 1, "status": "error", "error_stage": "generation"},
                           {"ordinal": 2, "status": "ignored"}, {"ordinal": 3, "status": "error", "error_stage": "input"}]}}
    (folder / "expectations.json").write_text(json.dumps(reference), encoding="utf-8")
    return folder, reference


def test_actual_results_are_compared_not_replaced_and_exports_match(tmp_path, batch_setup):
    s = batch_setup
    s.failures.add("api fails")
    folder, _ = fixtures(tmp_path)
    output = tmp_path / "reports"
    summary = run_acceptance(load_config(tmp_path, environ={}), folder, output, runtime=s.runtime, identity=s.identity)
    assert all(row["status"] == "checks_passed" for row in summary["batches"])
    assert summary["provider_operations"] == 2 and summary["successful_provider_operations"] == 1
    error = json.loads((output / "mixed/errors.json").read_bytes())["rows"][0]["outcome"]
    assert error["analysis"]["category"] and error["analysis"]["sentiment"] and not error["analysis"]["generated"]
    repeated = run_acceptance(load_config(tmp_path, environ={}), folder, output, runtime=s.runtime, identity=s.identity)
    assert repeated["provider_operations"] == 0 and len(s.calls["api"]) == 2
    assert len(list((output / "assessment/runs").iterdir())) == 2
    with closing(sqlite3.connect(output / "mixed/tickets.db")) as db:
        assert db.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 3


def test_mismatches_remain_visible_without_forcing_expected_priority(tmp_path, batch_setup):
    s = batch_setup
    s.failures.add("api fails")
    folder, refs = fixtures(tmp_path)
    refs["assessment"]["rows"][0]["priority"] = "Urgent"
    (folder / "expectations.json").write_text(json.dumps(refs))
    summary = run_acceptance(load_config(tmp_path, environ={}), folder, tmp_path / "reports", runtime=s.runtime, identity=s.identity)
    first = summary["batches"][0]
    assert first["status"] == "mismatches_or_errors" and first["mismatches"][0]["fields"] == ["priority"]
    actual = json.loads((tmp_path / "reports/assessment/batch.json").read_bytes())["rows"][0]["outcome"]
    assert actual["analysis"]["generated"]["details"]["priority"] == "Medium"


def test_all_fixture_counts_are_checked_before_any_paid_work(tmp_path, batch_setup):
    s = batch_setup
    folder, refs = fixtures(tmp_path)
    refs["mixed"]["rows"].pop()
    (folder / "expectations.json").write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="before model/API work"):
        run_acceptance(load_config(tmp_path, environ={}), folder, tmp_path / "reports", runtime=s.runtime, identity=s.identity)
    assert not any(s.calls.values())


def test_export_verification_detects_tampered_files(tmp_path, batch_setup):
    s = batch_setup
    result = analyze_csv(b"ticket_id,message\nA,App crashes\n", s.runtime, OutcomeStore(tmp_path / "tickets.db"), s.identity)
    bundle = build_exports(result)
    save_exports(bundle, tmp_path)
    (tmp_path / "results.csv").write_bytes(b"changed")
    checks = verify_exports(result, bundle, tmp_path)
    assert not checks["saved_bytes_equal_shared_exports"] and not checks["manifest_checksums_match"]
