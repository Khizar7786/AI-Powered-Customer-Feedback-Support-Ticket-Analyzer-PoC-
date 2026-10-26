"""Export fidelity, spreadsheet viewing protections and file failure boundaries."""

import csv
import hashlib
from io import StringIO
import json

import pytest

from feedback_analyzer import exports
from feedback_analyzer.exports import ExportError, build_exports, save_exports, spreadsheet_text
from feedback_analyzer.pipeline import analyze_csv
from feedback_analyzer.storage import OutcomeStore


def input_csv(rows):
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["ticket_id", "message"])
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def decode_csv(data):
    return list(csv.DictReader(StringIO(data.decode("utf-8-sig"), newline="")))


def test_full_exports_reconcile_and_preserve_quotes_unicode_newlines_and_partial_errors(tmp_path, batch_setup):
    s = batch_setup
    message = '  Hi,\n"Checkout" crashes — café  '
    s.failures.add("api fails")
    result = analyze_csv(input_csv([("A", message), ("B", "Hello"), ("C", "api fails"), ("", "Missing ID")]),
                         s.runtime, OutcomeStore(tmp_path / "tickets.db"), s.identity)
    bundle = build_exports(result, s.identity.metadata)
    processed, ignored = decode_csv(bundle.files["results.csv"]), decode_csv(bundle.files["ignored.csv"])
    payload, errors = json.loads(bundle.files["batch.json"]), json.loads(bundle.files["errors.json"])
    assert bundle.files["results.csv"].startswith(b"\xef\xbb\xbf")
    assert len(processed) == len(ignored) == 1 and len(errors["rows"]) == 2
    assert processed[0]["original_message"] == payload["rows"][0]["outcome"]["original_message"] == message
    assert payload["counts"] == {"processed": 1, "ignored": 1, "error": 2} and payload["total_rows"] == 4
    partial = errors["rows"][0]["outcome"]["analysis"]
    assert partial["category"] and partial["sentiment"] and partial["generated"] is None
    assert processed[0]["product_feature"] == "" and payload["rows"][0]["outcome"]["analysis"]["generated"]["details"]["product_feature"] is None
    assert processed[0]["input_hash"] == result.rows[0].input_hash
    assert result.rows[0].outcome.original_message == message  # No export mutation of shared data.


@pytest.mark.parametrize("value", ["=1+1", "+SUM(A1:A2)", "-10+20", "@SUM(A1)", "  =1+1", "\ufeff=1+1", "\u200b=1+1", "\tHello", "\rHello", "\nHello", "＝1+1", "＋1", "－1", "＠x"])
def test_formula_or_control_leading_csv_text_is_explicitly_protected(value):
    assert spreadsheet_text(value) == ("'" + value, True)


@pytest.mark.parametrize("value", ["", "App crashes", 'Text,"quoted"\nnext line', "Hi @team", "'already text", "contains = sign"])
def test_ordinary_text_is_unchanged(value):
    assert spreadsheet_text(value) == (value, False)


def test_csv_protection_applies_to_ids_and_messages_but_json_stays_exact(tmp_path, batch_setup):
    s = batch_setup
    original = '=HYPERLINK("https://example.invalid", "click")'
    result = analyze_csv(input_csv([("+dangerous-ID", original)]), s.runtime, OutcomeStore(tmp_path / "tickets.db"), s.identity)
    bundle = build_exports(result)
    row = decode_csv(bundle.files["results.csv"])[0]
    assert row["ticket_id"] == "'+dangerous-ID" and row["original_message"] == "'" + original
    assert {"ticket_id", "original_message", "normalized_message"}.issubset(row["spreadsheet_escaped_fields"].split(";"))
    assert json.loads(bundle.files["batch.json"])["rows"][0]["outcome"]["original_message"] == original
    assert bundle.escaped_cells >= 3


def test_empty_and_file_error_exports_have_headers_and_file_errors(tmp_path, batch_setup):
    s = batch_setup
    result = analyze_csv(b"wrong,columns\nA,issue\n", s.runtime, OutcomeStore(tmp_path / "tickets.db"), s.identity)
    bundle = build_exports(result)
    assert not decode_csv(bundle.files["results.csv"]) and not decode_csv(bundle.files["ignored.csv"])
    assert json.loads(bundle.files["errors.json"])["file_errors"] == result.file_errors


def test_files_and_manifest_match_memory_bytes_and_replacing_is_idempotent(tmp_path, batch_setup):
    s = batch_setup
    result = analyze_csv(b"ticket_id,message\nA,App crashes\n", s.runtime, OutcomeStore(tmp_path / "tickets.db"), s.identity)
    bundle = build_exports(result)
    save_exports(bundle, tmp_path)
    save_exports(bundle, tmp_path)
    manifest = json.loads((tmp_path / "exports_manifest.json").read_bytes())
    assert manifest["run_id"] == result.run_id
    for name, data in bundle.files.items():
        assert (tmp_path / name).read_bytes() == data
        assert manifest["sha256"][name] == hashlib.sha256(data).hexdigest()
    assert not list(tmp_path.glob(".export-*.tmp"))


def test_failed_file_replace_preserves_result_and_cleans_temps(tmp_path, batch_setup, monkeypatch):
    s = batch_setup
    result = analyze_csv(b"ticket_id,message\nA,App crashes\n", s.runtime, OutcomeStore(tmp_path / "tickets.db"), s.identity)
    bundle = build_exports(result)
    monkeypatch.setattr(exports.os, "replace", lambda *args: (_ for _ in ()).throw(PermissionError("private filesystem error")))
    with pytest.raises(ExportError, match="retry saving") as error:
        save_exports(bundle, tmp_path)
    assert "private" not in str(error.value) and result.fully_processed == 1
    assert not list(tmp_path.glob(".export-*.tmp")) and len(s.calls["api"]) == 1


def test_private_metadata_and_false_processed_results_cannot_be_exported(tmp_path, batch_setup):
    s = batch_setup
    result = analyze_csv(b"ticket_id,message\nA,App crashes\n", s.runtime, OutcomeStore(tmp_path / "tickets.db"), s.identity)
    with pytest.raises(ExportError, match="Private configuration"):
        build_exports(result, {"generation": {"api_key": "unit-secret"}})
    result.rows[0].persisted = False
    with pytest.raises(ExportError, match="validated, saved"):
        build_exports(result)
