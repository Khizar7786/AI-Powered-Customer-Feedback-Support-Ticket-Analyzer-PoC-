"""CLI calls the same pipeline, with explicit doubles and temporary SQLite files."""

import json
from contextlib import closing
import sqlite3

from scripts import analyze


def test_cli_mixed_input_and_persistent_replay(tmp_path, monkeypatch, capsys, batch_setup):
    s = batch_setup
    path = tmp_path / "sample.csv"
    path.write_text("ticket_id,message\nA,App crashes\nB,Hello\n,Missing ID\n")
    database = tmp_path / "tickets.db"
    monkeypatch.setattr(analyze.BatchRuntime, "local", lambda *args: s.runtime)
    monkeypatch.setattr(analyze, "build_identity", lambda *args: s.identity)
    args = ["--input", str(path), "--database", str(database)]
    assert analyze.main(args) == 1  # Intentional input error is not a successful exit.
    first = json.loads(capsys.readouterr().out)
    assert first["counts"] == {"processed": 1, "ignored": 1, "error": 1}
    assert first["errors"][0]["source_row"] == 4
    assert analyze.main(args) == 1
    second = json.loads(capsys.readouterr().out)
    assert second["reused"] == 3 and len(s.calls["api"]) == 1
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 3


def test_cli_missing_input_is_safe_setup_error(tmp_path, capsys):
    assert analyze.main(["--input", str(tmp_path / "missing.csv"), "--database", str(tmp_path / "tickets.db")]) == 2
    assert "no successful batch" in capsys.readouterr().out
    assert not (tmp_path / "tickets.db").exists()


def test_cli_flags_authorize_changed_input(tmp_path, monkeypatch, capsys, batch_setup):
    s = batch_setup
    path, database = tmp_path / "input.csv", tmp_path / "tickets.db"
    monkeypatch.setattr(analyze.BatchRuntime, "local", lambda *args: s.runtime)
    monkeypatch.setattr(analyze, "build_identity", lambda *args: s.identity)
    args = ["--input", str(path), "--database", str(database)]
    path.write_text("ticket_id,message\nA,App crashes\n")
    assert analyze.main(args) == 0
    capsys.readouterr()
    path.write_text("ticket_id,message\nA,Changed issue\n")
    assert analyze.main(args) == 1
    assert json.loads(capsys.readouterr().out)["errors"][0]["code"] == "reprocess_required"
    assert analyze.main([*args, "--reprocess"]) == 0
    assert len(s.calls["api"]) == 2
