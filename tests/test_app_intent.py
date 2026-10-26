"""Stage 5 UI contracts with explicit fake inference, not model verification."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from feedback_analyzer import config as config_module
from feedback_analyzer import readiness as readiness_module
from feedback_analyzer.intent import IntentClassifier, IntentPrediction, IntentSetupError
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


def create_fake_checkpoint(directory):
    """Only file detection is real. Tests must mock loading these dummy bytes."""
    directory.mkdir(parents=True)
    for name in (
        "config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "vocab.txt", "training_summary.json",
        "label_mappings.json", "model.safetensors",
    ):
        (directory / name).write_text("fake test artifact", encoding="utf-8")


class FakeClassifier:
    """Predetermined test outputs; never a substitute for live inference."""

    device = "cpu"
    device_warning = "Mock CPU device for deterministic UI tests."

    def __init__(self):
        self.messages = []
        self.fail_messages = set()

    def predict(self, message):
        self.messages.append(message)
        if message in self.fail_messages:
            raise RuntimeError("mock inference failed")
        category = {
            "Hello, hope you are having a nice day!": "Non-Qualifying",
            "Hi team, it would be awesome if you could add a dark mode to the mobile app.": "Feature Request",
            "My invoice at https://example.com/invoice will not open.": "Billing Issue",
        }.get(message, "Bug Report")
        truncated = message.startswith("long ")
        tokens = 200 if truncated else 12
        return IntentPrediction(
            category=category, confidence=0.2, input_tokens=tokens,
            used_tokens=min(tokens, 128), max_length=128,
            truncated=truncated, device=self.device,
        )


def offline_app(tmp_path, monkeypatch, *, checkpoint=True, classifier=None, load_error=None):
    config = config_module.load_config(tmp_path, environ={})
    if checkpoint:
        create_fake_checkpoint(config.classifier_dir)
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(
        readiness_module, "inspect_device", lambda: DeviceStatus("CPU", "Mock device.")
    )
    fake = classifier or FakeClassifier()
    loads = []

    def load(directory, device="auto"):
        loads.append((Path(directory), device))
        if load_error:
            raise load_error
        return fake

    monkeypatch.setattr(IntentClassifier, "from_checkpoint", staticmethod(load))
    return AppTest.from_file(str(APP_PATH), default_timeout=15), fake, loads


def tab(app, label):
    return next(item for item in app.tabs if item.label == label)


def test_category_action_is_explicit_and_results_survive_reruns(tmp_path, monkeypatch):
    app, classifier, loads = offline_app(tmp_path, monkeypatch)
    app.run()
    assert not app.exception
    assert app.button("analyze_categories").disabled
    assert not loads and not classifier.messages

    app.button("load_sample").click().run()
    assert not app.exception
    assert not app.button("analyze_categories").disabled
    assert not loads and not classifier.messages

    app.button("analyze_categories").click().run()
    assert not app.exception
    assert len(loads) == 1
    assert len(classifier.messages) == 6
    assert "Hello!" not in classifier.messages
    assert "" not in classifier.messages
    assert "The export button is broken." not in classifier.messages  # Invalid duplicate ID.
    assert app.button("analyze_tickets").disabled
    rows = app.session_state["category_results"].rows
    assert len(rows) == 12
    assert sum(row.status == "partial" for row in rows) == 5
    assert sum(row.status == "ignored" for row in rows) == 3
    assert sum(row.status == "error" for row in rows) == 4
    result_metrics = {metric.label: metric.value for metric in app.metric}
    assert result_metrics["Fully processed"] == "0"
    assert result_metrics["Partial analysis"] == "5"
    assert result_metrics["Ignored after category checks"] == "3"

    partial = tab(app, "Partial analysis").dataframe[0].value
    assert set(partial["category"]) == {"Bug Report", "Feature Request", "Billing Issue"}
    assert partial["confidence"].eq(0.2).all()
    ignored = tab(app, "Ignored after category checks").dataframe[0].value
    greeting = ignored.loc[ignored["ticket_id"] == "T003"].iloc[0]
    assert greeting["category"] == "Non-Qualifying"
    assert "Non-Qualifying" in greeting["reason"]
    assert "partial" in visible_text(app).lower()
    assert classifier.device_warning in visible_text(app)

    app.run()
    assert not app.exception
    assert len(loads) == 1 and len(classifier.messages) == 6
    source_selector = next(item for item in app.selectbox if item.label == "Inspect a source row")
    source_selector.select_index(1).run()
    assert not app.exception
    assert len(loads) == 1 and len(classifier.messages) == 6

    # A second explicit action may rerun inference but must reuse the resource.
    app.button("analyze_categories").click().run()
    assert not app.exception
    assert len(loads) == 1 and len(classifier.messages) == 12


def test_missing_checkpoint_shows_training_command_and_never_loads(tmp_path, monkeypatch):
    app, classifier, loads = offline_app(tmp_path, monkeypatch, checkpoint=False)
    app.run()
    app.button("load_sample").click().run()
    assert not app.exception
    assert app.button("analyze_categories").disabled
    assert "python -m scripts.train_intent" in "\n".join(item.value for item in app.code)
    assert not loads and not classifier.messages
    assert "category_results" not in app.session_state


def test_load_failure_stays_visible_without_any_prediction(tmp_path, monkeypatch):
    app, classifier, loads = offline_app(
        tmp_path, monkeypatch, load_error=IntentSetupError("Invalid trained checkpoint. Run python -m scripts.train_intent.")
    )
    app.run()
    app.button("load_sample").click().run()
    app.button("analyze_categories").click().run()
    assert not app.exception
    assert len(loads) == 1 and not classifier.messages
    assert "classifier could not be loaded or run" in visible_text(app)
    assert "python -m scripts.train_intent --verify-only" in "\n".join(item.value for item in app.code)
    assert "category_results" not in app.session_state
    assert app.button("analyze_tickets").disabled
    app.run()
    assert not app.exception
    assert len(loads) == 1
    assert "classifier could not be loaded or run" in visible_text(app)


def test_failed_candidate_is_error_and_other_candidates_continue(tmp_path, monkeypatch):
    fake = FakeClassifier()
    fake.fail_messages.add("App crashes")
    app, classifier, loads = offline_app(tmp_path, monkeypatch, classifier=fake)
    app.run()
    app.button("load_sample").click().run()
    app.button("analyze_categories").click().run()
    assert not app.exception
    assert len(loads) == 1 and len(classifier.messages) == 6
    rows = app.session_state["category_results"].rows
    failed = next(row for row in rows if row.ticket_id == "T007")
    assert failed.status == "error"
    assert failed.prediction is None
    assert sum(row.status == "partial" for row in rows) == 4
    assert sum(row.status == "ignored" for row in rows) == 3
    errors = tab(app, "Category errors").dataframe[0].value
    assert "T007" in errors["ticket_id"].tolist()
    assert {metric.label: metric.value for metric in app.metric}["Fully processed"] == "0"


def test_upload_preserves_original_and_displays_input_truncation(tmp_path, monkeypatch):
    original = "  long " + ("message " * 150) + "\n tail  "
    data = ('ticket_id,message\nLONG,"' + original + '"\n').encode("utf-8")
    upload = SimpleNamespace(name="long.csv", getvalue=lambda: data)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app, classifier, loads = offline_app(tmp_path, monkeypatch)
    app.run()
    assert not loads and not classifier.messages
    app.button("validate_csv").click().run()
    assert not loads and not classifier.messages
    app.button("analyze_categories").click().run()
    assert not app.exception
    row = app.session_state["category_results"].rows[0]
    assert row.original_message == original
    assert classifier.messages == [" ".join(original.split())]
    assert row.prediction.truncated
    assert row.prediction.input_tokens == 200
    assert row.prediction.used_tokens == 128
    table = tab(app, "Partial analysis").dataframe[0].value.iloc[0]
    assert table["original_message"] == original
    assert table["classifier_input_truncated"]
    assert table["input_tokens"] == 200
    assert table["used_tokens"] == 128
    assert table["max_length"] == 128
    assert "Only the beginning was classified" in visible_text(app)


def test_new_input_clears_previous_results_and_requires_explicit_action(tmp_path, monkeypatch):
    selection = {"upload": None, "changed": False}

    def uploader(*args, **kwargs):
        if selection["changed"]:
            kwargs["on_change"]()
            selection["changed"] = False
        return selection["upload"]

    monkeypatch.setattr(st, "file_uploader", uploader)
    app, classifier, loads = offline_app(tmp_path, monkeypatch)
    app.run()
    app.button("load_sample").click().run()
    app.button("analyze_categories").click().run()
    assert "category_results" in app.session_state

    selection["upload"] = SimpleNamespace(name="new.csv", getvalue=lambda: b"ticket_id,message\nNEW,Hi checkout crashes\n")
    selection["changed"] = True
    app.run()
    assert not app.exception
    assert "category_results" not in app.session_state
    assert "csv_preview" not in app.session_state
    assert len(classifier.messages) == 6 and len(loads) == 1
    assert app.button("analyze_categories").disabled
    app.button("validate_csv").click().run()
    assert "category_results" not in app.session_state
    assert len(classifier.messages) == 6
    app.button("analyze_categories").click().run()
    assert not app.exception
    assert classifier.messages[-1] == "Hi checkout crashes"
    assert len(loads) == 1  # Same checkpoint resource reused for a new batch.


def test_device_change_clears_results_without_automatically_loading_model(tmp_path, monkeypatch):
    app, classifier, loads = offline_app(tmp_path, monkeypatch)
    app.run()
    app.button("load_sample").click().run()
    app.button("analyze_categories").click().run()
    widgets = list(app.radio) + list(app.selectbox)
    selector = next(item for item in widgets if item.key == "intent_device")
    selector.set_value("CPU").run()
    assert not app.exception
    assert "category_results" not in app.session_state
    assert len(loads) == 1 and len(classifier.messages) == 6
    app.button("analyze_categories").click().run()
    assert not app.exception
    assert len(loads) == 2
    assert loads[-1][1] == "cpu"


def test_checkpoint_changes_clear_results_and_reload_only_on_action(tmp_path, monkeypatch):
    app, classifier, loads = offline_app(tmp_path, monkeypatch)
    app.run()
    app.button("load_sample").click().run()
    app.button("analyze_categories").click().run()
    assert len(loads) == 1
    checkpoint = tmp_path / "models" / "ticket_classifier" / "model.safetensors"
    checkpoint.write_text("changed fake test artifact", encoding="utf-8")
    app.run()
    assert not app.exception
    assert "category_results" not in app.session_state
    assert len(loads) == 1 and len(classifier.messages) == 6
    app.button("analyze_categories").click().run()
    assert not app.exception
    assert len(loads) == 2 and len(classifier.messages) == 12


def test_separate_sessions_share_cached_model_but_keep_results_separate(tmp_path, monkeypatch):
    first, classifier, loads = offline_app(tmp_path, monkeypatch)
    first.run()
    first.button("load_sample").click().run()
    first.button("analyze_categories").click().run()
    second = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert not second.exception
    assert "category_results" not in second.session_state
    assert len(loads) == 1 and len(classifier.messages) == 6
    second.button("load_sample").click().run()
    assert len(classifier.messages) == 6
    second.button("analyze_categories").click().run()
    assert not second.exception
    assert len(loads) == 1 and len(classifier.messages) == 12
    assert "category_results" in first.session_state
    assert "category_results" in second.session_state


@pytest.mark.parametrize("data", [
    b"ticket_id,message\nEMPTY,Hello!\n",
    b"example_id,message,label\nDEV001,App crashes,Bug Report\n",
    b'ticket_id,message\nX,"unterminated\n',
])
def test_no_candidates_or_file_errors_do_not_enable_category_action(data, tmp_path, monkeypatch):
    upload = SimpleNamespace(name="invalid_or_ignored.csv", getvalue=lambda: data)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app, classifier, loads = offline_app(tmp_path, monkeypatch)
    app.run()
    app.button("validate_csv").click().run()
    assert not app.exception
    assert app.button("analyze_categories").disabled
    assert not loads and not classifier.messages
