"""Stage 6 UI contracts with explicit fake classifiers, never live weights."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from feedback_analyzer import config as config_module
from feedback_analyzer import readiness as readiness_module
from feedback_analyzer.intent import IntentClassifier, IntentPrediction
from feedback_analyzer.readiness import DeviceStatus
from feedback_analyzer.sentiment import (
    SentimentClassifier, SentimentPrediction, SentimentSetupError, preprocess_message,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def clear_resources():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def visible_text(app):
    return "\n".join(
        element.value for elements in (app.markdown, app.caption, app.info, app.warning, app.error)
        for element in elements
    )


def table(app, label):
    return next(tab for tab in app.tabs if tab.label == label).dataframe[0].value


def fake_files(directory, sentiment=False):
    directory.mkdir(parents=True)
    names = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    names += ["pytorch_model.bin", "sentiment_source.json"] if sentiment else ["model.safetensors", "training_summary.json"]
    for name in names:
        (directory / name).write_text("explicit fake artifact", encoding="utf-8")


class FakeIntent:
    device = "cpu"
    device_warning = None

    def __init__(self):
        self.messages = []
        self.all_ignored = False

    def predict(self, message):
        self.messages.append(message)
        category = "Non-Qualifying" if self.all_ignored or message == "Hello, hope you are having a nice day!" else "Bug Report"
        if not self.all_ignored and "dark mode" in message:
            category = "Feature Request"
        truncated = message.startswith("long ")
        return IntentPrediction(
            category=category, confidence=0.2, input_tokens=700 if truncated else 12,
            used_tokens=128 if truncated else 12, max_length=128, truncated=truncated, device="cpu",
        )


class FakeSentiment:
    """Labels deliberately predetermined only in tests, including a neutral feature request."""

    device = "cpu"
    device_warning = "Mock sentiment CPU fallback warning."
    revision = "a" * 40

    def __init__(self):
        self.messages = []
        self.fail_messages = set()
        self.invalid_messages = set()

    def predict(self, message):
        self.messages.append(message)
        if message in self.fail_messages:
            raise RuntimeError("explicit mock sentiment failure")
        if message in self.invalid_messages:
            return {"sentiment": "Happy", "confidence": 0.9}
        truncated = message.startswith("long ")
        return SentimentPrediction(
            sentiment="Neutral" if "dark mode" in message or "invoice" in message else "Negative",
            confidence=0.2, input_tokens=700 if truncated else 12,
            used_tokens=512 if truncated else 12, max_length=512,
            truncated=truncated, device="cpu", model_input=preprocess_message(message),
        )


def offline_app(tmp_path, monkeypatch, *, model_exists=True, load_error=None):
    config = config_module.load_config(tmp_path, environ={})
    fake_files(config.classifier_dir)
    if model_exists:
        fake_files(config.sentiment_dir, sentiment=True)
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(readiness_module, "inspect_device", lambda: DeviceStatus("CPU", "Mock device."))
    category = FakeIntent()
    sentiment = FakeSentiment()
    loads = []

    def load_sentiment(directory, device="cpu"):
        loads.append((Path(directory), device))
        if load_error:
            raise load_error
        return sentiment

    monkeypatch.setattr(IntentClassifier, "from_checkpoint", staticmethod(lambda directory, device="auto": category))
    monkeypatch.setattr(SentimentClassifier, "from_checkpoint", staticmethod(load_sentiment))
    return AppTest.from_file(str(APP_PATH), default_timeout=20), category, sentiment, loads


def analyze_sample_categories(app):
    app.run()
    app.button("load_sample").click().run()
    app.button("analyze_categories").click().run()
    assert not app.exception


def analyze_sample_sentiment(app):
    analyze_sample_categories(app)
    app.button("analyze_sentiment").click().run()
    assert not app.exception


def test_explicit_action_runs_only_qualifying_rows_and_keeps_neutral_feature_request(tmp_path, monkeypatch):
    app, category, sentiment, loads = offline_app(tmp_path, monkeypatch)
    app.run()
    assert not app.exception
    assert app.button("analyze_sentiment").disabled
    assert app.selectbox("sentiment_device").value == "CPU"
    assert not loads and not sentiment.messages
    app.button("load_sample").click().run()
    assert app.button("analyze_sentiment").disabled
    assert not loads and not sentiment.messages
    app.button("analyze_categories").click().run()
    assert len(category.messages) == 6
    assert not app.button("analyze_sentiment").disabled
    assert not loads and not sentiment.messages
    app.button("analyze_sentiment").click().run()
    assert not app.exception
    assert len(loads) == 1 and loads[0][1] == "cpu"
    assert len(sentiment.messages) == 5
    assert "Hello, hope you are having a nice day!" not in sentiment.messages
    assert "Hello!" not in sentiment.messages and "" not in sentiment.messages
    result = app.session_state["sentiment_results"]
    assert result.counts == {"partial": 5, "ignored": 3, "error": 4}
    assert result.fully_processed == 0
    assert all(row.prediction is None for row in result.rows if row.category_row.status != "partial")
    feature = next(row for row in result.rows if row.category_row.ticket_id == "T002")
    assert feature.prediction.sentiment == "Neutral"
    assert feature.status == "partial"
    assert "pending" in feature.reason
    records = table(app, "Category + sentiment (partial)")
    assert len(records) == 5
    assert set(records["sentiment"]) <= {"Positive", "Neutral", "Negative"}
    assert "T003" not in records["ticket_id"].tolist()
    assert {m.label: m.value for m in app.metric}["Fully processed"] == "0"
    assert app.button("analyze_tickets").disabled
    assert "Mock sentiment CPU fallback warning" in visible_text(app)


def test_reruns_reuse_results_and_explicit_reanalysis_reuses_cached_model(tmp_path, monkeypatch):
    app, _, sentiment, loads = offline_app(tmp_path, monkeypatch)
    analyze_sample_sentiment(app)
    app.run()
    assert not app.exception
    assert len(loads) == 1 and len(sentiment.messages) == 5
    app.button("analyze_sentiment").click().run()
    assert not app.exception
    assert len(loads) == 1 and len(sentiment.messages) == 10


def test_category_reanalysis_invalidates_sentiment_without_automatically_repeating_it(tmp_path, monkeypatch):
    app, _, sentiment, loads = offline_app(tmp_path, monkeypatch)
    analyze_sample_sentiment(app)
    app.button("analyze_categories").click().run()
    assert not app.exception
    assert "sentiment_results" not in app.session_state
    assert "category_results" in app.session_state
    assert len(loads) == 1 and len(sentiment.messages) == 5
    app.button("analyze_sentiment").click().run()
    assert not app.exception
    assert len(loads) == 1 and len(sentiment.messages) == 10


def test_input_change_clears_sentiment_until_validation_and_both_explicit_actions(tmp_path, monkeypatch):
    selection = {"upload": None, "changed": False}

    def uploader(*args, **kwargs):
        if selection["changed"]:
            kwargs["on_change"]()
            selection["changed"] = False
        return selection["upload"]

    monkeypatch.setattr(st, "file_uploader", uploader)
    app, _, sentiment, loads = offline_app(tmp_path, monkeypatch)
    analyze_sample_sentiment(app)
    selection["upload"] = SimpleNamespace(name="new.csv", getvalue=lambda: b"ticket_id,message\nNEW,App crashes\n")
    selection["changed"] = True
    app.run()
    assert not app.exception
    assert "category_results" not in app.session_state and "sentiment_results" not in app.session_state
    assert app.button("analyze_sentiment").disabled
    assert len(loads) == 1 and len(sentiment.messages) == 5
    app.button("validate_csv").click().run()
    assert app.button("analyze_sentiment").disabled
    app.button("analyze_categories").click().run()
    assert "sentiment_results" not in app.session_state
    assert len(sentiment.messages) == 5
    app.button("analyze_sentiment").click().run()
    assert not app.exception
    assert len(loads) == 1 and len(sentiment.messages) == 6
    assert app.session_state["sentiment_results"].rows[0].category_row.ticket_id == "NEW"


def test_sentiment_device_change_preserves_categories_and_requires_explicit_reload(tmp_path, monkeypatch):
    app, _, sentiment, loads = offline_app(tmp_path, monkeypatch)
    analyze_sample_sentiment(app)
    app.selectbox("sentiment_device").select("Auto").run()
    assert not app.exception
    assert "sentiment_results" not in app.session_state
    assert "category_results" in app.session_state
    assert len(loads) == 1 and len(sentiment.messages) == 5
    app.button("analyze_sentiment").click().run()
    assert not app.exception
    assert len(loads) == 2 and loads[-1][1] == "auto"


def test_sentiment_checkpoint_change_invalidates_results_without_automatic_load(tmp_path, monkeypatch):
    app, _, sentiment, loads = offline_app(tmp_path, monkeypatch)
    analyze_sample_sentiment(app)
    (tmp_path / "models/sentiment/pytorch_model.bin").write_text("new explicit fake weights", encoding="utf-8")
    app.run()
    assert not app.exception
    assert "sentiment_results" not in app.session_state
    assert "category_results" in app.session_state
    assert len(loads) == 1 and len(sentiment.messages) == 5
    app.button("analyze_sentiment").click().run()
    assert not app.exception
    assert len(loads) == 2 and len(sentiment.messages) == 10


def test_missing_sentiment_model_shows_setup_command_without_loading(tmp_path, monkeypatch):
    app, _, sentiment, loads = offline_app(tmp_path, monkeypatch, model_exists=False)
    analyze_sample_categories(app)
    assert app.button("analyze_sentiment").disabled
    assert not loads and not sentiment.messages
    assert "python -m scripts.evaluate_sentiment --download-model" in "\n".join(item.value for item in app.code)
    assert "missing or incomplete" in visible_text(app)
    assert "category_results" in app.session_state


def test_sentiment_setup_failure_retains_categories_and_does_not_retry_on_rerun(tmp_path, monkeypatch):
    app, _, sentiment, loads = offline_app(tmp_path, monkeypatch, load_error=SentimentSetupError("explicit mock invalid weights"))
    analyze_sample_categories(app)
    app.button("analyze_sentiment").click().run()
    assert not app.exception
    assert "sentiment_results" not in app.session_state
    assert "category_results" in app.session_state
    assert "Sentiment could not be loaded or run" in visible_text(app)
    assert "Category results are retained" in visible_text(app)
    assert len(loads) == 1 and not sentiment.messages
    app.run()
    assert not app.exception
    assert len(loads) == 1 and not sentiment.messages


@pytest.mark.parametrize("bad_output", [False, True], ids=["inference-exception", "invalid-label"])
def test_sentiment_row_failure_keeps_category_and_is_error_not_spam(tmp_path, monkeypatch, bad_output):
    app, _, sentiment, loads = offline_app(tmp_path, monkeypatch)
    (sentiment.invalid_messages if bad_output else sentiment.fail_messages).add("App crashes")
    analyze_sample_sentiment(app)
    assert len(loads) == 1 and len(sentiment.messages) == 5
    result = app.session_state["sentiment_results"]
    failed = next(row for row in result.rows if row.category_row.ticket_id == "T007")
    assert failed.status == "error" and failed.prediction is None
    assert failed.category_row.prediction.category == "Bug Report"
    assert result.counts == {"partial": 4, "ignored": 3, "error": 5}
    records = table(app, "Sentiment errors")
    assert records["ticket_id"].tolist() == ["T007"]
    assert records.iloc[0]["category"] == "Bug Report"
    assert records.iloc[0]["sentiment"] is None
    assert {m.label: m.value for m in app.metric}["Sentiment inference errors"] == "1"
    assert {m.label: m.value for m in app.metric}["Fully processed"] == "0"


def test_original_text_and_separate_classifier_sentiment_truncation_are_displayed(tmp_path, monkeypatch):
    original = "  long " + ("message " * 150) + "\n @support https://example.com/help  "
    upload = SimpleNamespace(name="long.csv", getvalue=lambda: ('ticket_id,message\nLONG,"' + original + '"\n').encode())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app, _, sentiment, _ = offline_app(tmp_path, monkeypatch)
    app.run()
    app.button("validate_csv").click().run()
    app.button("analyze_categories").click().run()
    assert not sentiment.messages
    app.button("analyze_sentiment").click().run()
    assert not app.exception
    result = app.session_state["sentiment_results"].rows[0]
    assert result.category_row.original_message == original
    assert result.prediction.truncated
    assert result.prediction.used_tokens == 512
    assert sentiment.messages == [" ".join(original.split())]
    record = table(app, "Category + sentiment (partial)").iloc[0]
    assert record["original_message"] == original
    assert record["classifier_input_truncated"] and record["sentiment_input_truncated"]
    assert record["sentiment_input_tokens"] == 700
    assert record["sentiment_used_tokens"] == 512
    assert record["sentiment_max_length"] == 512
    assert record["sentiment_model_input"].endswith("@user http")
    assert "Only the beginning was analyzed" in visible_text(app)


def test_all_nonqualifying_categories_never_enable_sentiment(tmp_path, monkeypatch):
    app, category, sentiment, loads = offline_app(tmp_path, monkeypatch)
    category.all_ignored = True
    analyze_sample_categories(app)
    assert app.session_state["category_results"].counts["partial"] == 0
    assert app.button("analyze_sentiment").disabled
    assert not loads and not sentiment.messages


def test_separate_sessions_share_sentiment_resource_but_not_results(tmp_path, monkeypatch):
    first, _, sentiment, loads = offline_app(tmp_path, monkeypatch)
    analyze_sample_sentiment(first)
    second = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not second.exception
    assert "sentiment_results" not in second.session_state
    assert len(loads) == 1 and len(sentiment.messages) == 5
    second.button("load_sample").click().run()
    second.button("analyze_categories").click().run()
    assert len(sentiment.messages) == 5
    second.button("analyze_sentiment").click().run()
    assert not second.exception
    assert len(loads) == 1 and len(sentiment.messages) == 10
    assert "sentiment_results" in first.session_state and "sentiment_results" in second.session_state
