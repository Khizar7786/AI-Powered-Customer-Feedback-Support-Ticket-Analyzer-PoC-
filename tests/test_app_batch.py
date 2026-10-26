"""Streamlit batch/persistence integration with explicit model and API doubles."""

import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from feedback_analyzer import batch_ui, config as config_module, readiness, pipeline
from feedback_analyzer.storage import StorageError


@pytest.fixture(autouse=True)
def clear_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def setup(tmp_path, monkeypatch, s, data=b"ticket_id,message\nA,App crashes\nB,Hello\n,Missing ID\n"):
    (tmp_path / ".env").write_text("LLM_PROVIDER=openai\nLLM_MODEL=test-model\nOPENAI_API_KEY=unit-ui-secret\n")
    config = config_module.load_config(tmp_path, environ={})
    for directory in (config.classifier_dir, config.sentiment_dir):
        directory.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "model.safetensors", "tokenizer.json"):
            (directory / name).write_text("explicit fake model bytes")
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(readiness, "inspect_device", lambda: readiness.DeviceStatus("CPU", "Mocked device"))
    monkeypatch.setattr(batch_ui, "build_identity", lambda *args: s.identity)
    monkeypatch.setattr(pipeline.BatchRuntime, "local", lambda *args, **kwargs: s.runtime)
    selected = {"data": data}
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: SimpleNamespace(name="test.csv", getvalue=lambda: selected["data"]))
    return AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run(), selected


def test_explicit_batch_action_reruns_and_repeat_submission(tmp_path, monkeypatch, batch_setup):
    s = batch_setup
    app, _ = setup(tmp_path, monkeypatch, s)
    assert not (tmp_path / "outputs/tickets.db").exists() and not any(s.calls.values())
    app.button("validate_csv").click().run()
    assert not app.exception and not app.button("analyze_tickets").disabled and not s.calls["api"]
    app.button("analyze_tickets").click().run()
    assert not app.exception
    result = app.session_state["batch_results"]
    assert result.counts == {"processed": 1, "ignored": 1, "error": 1} and len(s.calls["api"]) == 1
    app.run()
    with closing(sqlite3.connect(tmp_path / "outputs/tickets.db")) as db:
        assert db.execute("SELECT count(*) FROM batch_runs").fetchone()[0] == 1
    app.button("analyze_tickets").click().run()
    assert app.session_state["batch_results"].reused_count == 3 and len(s.calls["api"]) == 1
    assert "unit-ui-secret" not in str(app.session_state)


def test_changed_input_waits_for_reprocessing_checkbox(tmp_path, monkeypatch, batch_setup):
    s = batch_setup
    app, selected = setup(tmp_path, monkeypatch, s, b"ticket_id,message\nA,App crashes\n")
    app.button("validate_csv").click().run()
    app.button("analyze_tickets").click().run()
    selected["data"] = b"ticket_id,message\nA,Changed issue\n"
    app.button("validate_csv").click().run()
    assert "batch_results" not in app.session_state
    app.button("analyze_tickets").click().run()
    assert app.session_state["batch_results"].rows[0].outcome.error.code == "reprocess_required"
    assert len(s.calls["api"]) == 1
    app.checkbox("batch_reprocess").check().run()
    assert len(s.calls["api"]) == 1
    app.button("analyze_tickets").click().run()
    assert app.session_state["batch_results"].fully_processed == 1 and len(s.calls["api"]) == 2


def test_failed_ticket_keeps_partial_and_next_ticket_runs(tmp_path, monkeypatch, batch_setup):
    s = batch_setup
    s.failures.add("api fails")
    app, _ = setup(tmp_path, monkeypatch, s, b"ticket_id,message\nA,api fails\nB,App crashes\n")
    app.button("validate_csv").click().run()
    app.button("analyze_tickets").click().run()
    result = app.session_state["batch_results"]
    assert result.counts == {"processed": 1, "ignored": 0, "error": 1}
    assert result.rows[0].outcome.analysis.sentiment and not result.rows[0].outcome.analysis.generated
    s.failures.clear()
    app.checkbox("batch_retry_errors").check().run()
    app.button("analyze_tickets").click().run()
    assert app.session_state["batch_results"].fully_processed == 2 and len(s.calls["api"]) == 3


def test_storage_setup_failure_prevents_model_and_api_work(tmp_path, monkeypatch, batch_setup):
    s = batch_setup
    app, _ = setup(tmp_path, monkeypatch, s)
    app.button("validate_csv").click().run()
    monkeypatch.setattr(batch_ui, "OutcomeStore", lambda *args: (_ for _ in ()).throw(StorageError("mock database unavailable")))
    app.button("analyze_tickets").click().run()
    assert not app.exception and not any(s.calls.values())
    assert "batch_results" not in app.session_state and "storage failed" in app.session_state["batch_error"]
