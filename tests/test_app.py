"""Streamlit rendering checks with isolated configuration and a fake device."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from feedback_analyzer import config as config_module
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
        for elements in (app.markdown, app.caption, app.info, app.warning)
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
    assert app.button[0].disabled

    app.run()
    assert not app.exception
    assert len(probes) == 1
    assert app.button[0].disabled


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
    assert app.button[0].disabled
