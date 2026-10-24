"""Streamlit rendering checks with isolated configuration and a fake device."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from feedback_analyzer import config as config_module
from feedback_analyzer import ingestion as ingestion_module
from feedback_analyzer import readiness as readiness_module
from feedback_analyzer.readiness import DeviceStatus


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def clear_resource_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def visible_text(app):
    return "\n".join(
        item.value
        for elements in (app.markdown, app.caption, app.info, app.warning, app.error)
        for item in elements
    )


def test_missing_setup_renders_and_reruns_without_repeating_device_probe(tmp_path, monkeypatch):
    config = config_module.load_config(tmp_path, environ={})
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    probes = []

    def fake_probe():
        probes.append(True)
        return DeviceStatus("CPU", "Offline test device.")

    monkeypatch.setattr(readiness_module, "inspect_device", fake_probe)
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert app.title[0].value == "Feedback Triage AI"
    assert visible_text(app).count("**Missing**") == 2
    assert "**Not configured**" in visible_text(app)
    assert app.button("analyze_tickets").disabled
    assert app.button("validate_csv").disabled
    assert not app.metric

    app.run()
    assert not app.exception
    assert len(probes) == 1
    assert app.button("analyze_tickets").disabled


def test_configured_api_is_unverified_and_secrets_never_render(tmp_path, monkeypatch):
    key = "dummy-test-only-secret"
    model = "dummy-model-presence-only"
    config = config_module.load_config(
        tmp_path,
        environ={"LLM_PROVIDER": "openai", "LLM_MODEL": model, "OPENAI_API_KEY": key},
    )
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(
        readiness_module, "inspect_device", lambda: DeviceStatus("CPU", "Offline test device.")
    )
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    text = visible_text(app)
    assert "**Configured (unverified)**" in text
    assert key not in text
    assert model not in text
    assert app.button("analyze_tickets").disabled


def offline_app(tmp_path, monkeypatch):
    """Keep UI checks independent of local models, credentials, and hardware."""
    config = config_module.load_config(tmp_path, environ={})
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(
        readiness_module, "inspect_device", lambda: DeviceStatus("CPU", "Offline test device.")
    )
    return AppTest.from_file(str(APP_PATH), default_timeout=10)


def test_sample_preview_has_three_outcomes_and_reruns_do_not_reparse(tmp_path, monkeypatch):
    calls = []
    real_load = ingestion_module.load_ticket_csv

    def record_load(data):
        calls.append(data)
        return real_load(data)

    monkeypatch.setattr(ingestion_module, "load_ticket_csv", record_load)
    app = offline_app(tmp_path, monkeypatch).run()
    assert not calls

    app.button("load_sample").click().run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric} == {
        "Rows read": "12", "Candidates": "6", "Ignored": "2", "Input errors": "4",
    }
    assert [tab.label for tab in app.tabs] == ["Candidates", "Ignored", "Input errors"]
    candidates = app.tabs[0].dataframe[0].value
    assert candidates["ticket_id"].tolist() == ["T001", "T002", "T003", "T006", "T007", "T008"]
    assert candidates.loc[candidates["ticket_id"] == "T003", "original_message"].item() == (
        "Hello, hope you are having a nice day!"
    )
    assert app.tabs[1].dataframe[0].value["ticket_id"].tolist() == ["T004", "T005"]
    assert app.tabs[2].dataframe[0].value["source_row"].tolist() == [10, 11, 12, 13]
    assert "category" not in candidates.columns
    assert app.button("analyze_tickets").disabled

    app.run()
    assert not app.exception
    assert len(calls) == 1
    assert {metric.label: metric.value for metric in app.metric}["Candidates"] == "6"


def test_uploaded_csv_waits_for_validation_and_displays_source_lines(tmp_path, monkeypatch):
    data = (
        b'ticket_id,message\nA1,"  Hi,\ncheckout crashes  "\n'
        b',App crashes\nD1,Hello\nD1,Billing failed\nX1,Too,many\n'
    )
    upload = SimpleNamespace(name="uploaded_tickets.csv", getvalue=lambda: data)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app = offline_app(tmp_path, monkeypatch).run()
    assert not app.exception
    assert not app.metric
    assert not app.button("validate_csv").disabled

    app.button("validate_csv").click().run()
    assert not app.exception
    assert "Preview source: uploaded_tickets.csv" in visible_text(app)
    assert {metric.label: metric.value for metric in app.metric} == {
        "Rows read": "5", "Candidates": "1", "Ignored": "0", "Input errors": "4",
    }
    candidate = app.tabs[0].dataframe[0].value.iloc[0]
    assert candidate["source_row"] == 2
    assert candidate["source_line_end"] == 3
    assert candidate["original_message"] == "  Hi,\ncheckout crashes  "
    assert candidate["normalized_message"] == "Hi, checkout crashes"
    assert app.code[0].value == "  Hi,\ncheckout crashes  "
    errors = app.tabs[2].dataframe[0].value
    assert errors["source_row"].tolist() == [4, 5, 6, 7]
    assert errors["reason"].str.len().gt(0).all()
    assert errors.loc[errors["source_row"] == 7, "raw_record"].item() == "X1,Too,many\n"
    assert app.button("analyze_tickets").disabled


@pytest.mark.parametrize(
    "data",
    [
        b"example_id,message,label\nDEV001,App crashes,Bug Report\n",
        b"ticket_id,message\nA1,\xff\n",
        b'ticket_id,message\nA1,"unterminated message\n',
    ],
    ids=["training-schema", "invalid-encoding", "malformed-quoting"],
)
def test_upload_file_errors_are_visible_and_never_enable_analysis(data, tmp_path, monkeypatch):
    upload = SimpleNamespace(name="invalid.csv", getvalue=lambda: data)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app = offline_app(tmp_path, monkeypatch).run()
    app.button("validate_csv").click().run()

    assert not app.exception
    preview = app.session_state["csv_preview"]
    assert preview.file_errors
    assert all(error in visible_text(app) for error in preview.file_errors)
    assert all(row.status == "error" for row in preview.rows)
    assert app.button("analyze_tickets").disabled


def test_upload_change_clears_old_preview_until_explicit_validation(tmp_path, monkeypatch):
    selection = {"upload": None, "changed": False}

    def fake_uploader(*args, **kwargs):
        if selection["changed"]:
            kwargs["on_change"]()
            selection["changed"] = False
        return selection["upload"]

    monkeypatch.setattr(st, "file_uploader", fake_uploader)
    app = offline_app(tmp_path, monkeypatch).run()
    app.button("load_sample").click().run()
    assert app.metric

    selection["upload"] = SimpleNamespace(
        name="replacement.csv", getvalue=lambda: b"ticket_id,message\nNEW,App crashes\n",
    )
    selection["changed"] = True
    app.run()
    assert not app.exception
    assert "csv_preview" not in app.session_state
    assert not app.metric
    assert "sample_tickets.csv" not in visible_text(app)

    app.button("validate_csv").click().run()
    assert not app.exception
    assert app.tabs[0].dataframe[0].value["ticket_id"].tolist() == ["NEW"]
    assert app.button("analyze_tickets").disabled
