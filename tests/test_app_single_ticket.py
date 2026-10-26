"""Single-ticket Streamlit behavior with explicit model/provider doubles only."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from feedback_analyzer import config as config_module, generation as generation_module, readiness
from feedback_analyzer.generation import GeneratedResult, GenerationError, GenerationReceipt
from feedback_analyzer.intent import IntentClassifier, IntentPrediction
from feedback_analyzer.schemas import GeneratedDetails
from feedback_analyzer.sentiment import SentimentClassifier, SentimentPrediction


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def resource_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def setup_app(tmp_path, monkeypatch, *, key=True, refusal=False, category="Bug Report", sentiment_fail=False):
    config = config_module.load_config(tmp_path, environ={})
    for folder in (config.classifier_dir, config.sentiment_dir):
        folder.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "model.safetensors", "tokenizer.json"):
            (folder / name).write_text("test fake files", encoding="utf-8")
    (tmp_path / ".env").write_text("LLM_PROVIDER=openai\nLLM_MODEL=gpt-5.4-mini\nOPENAI_API_KEY=" + ("unit-private-test-key" if key else "") + "\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(readiness, "inspect_device", lambda: readiness.DeviceStatus("CPU", "Mock device"))
    calls = {"category": [], "sentiment": [], "provider": [], "factories": 0}

    class FakeCategory:
        device, device_warning = "cpu", None

        def predict(self, text):
            calls["category"].append(text)
            return IntentPrediction(category=category, confidence=0.3, input_tokens=10, used_tokens=10,
                                    max_length=128, truncated=False, device="cpu")

    class FakeSentiment:
        device, device_warning = "cpu", None

        def predict(self, text):
            calls["sentiment"].append(text)
            if sentiment_fail:
                raise RuntimeError("RAW_LOCAL_EXCEPTION_DO_NOT_SHOW")
            return SentimentPrediction(sentiment="Negative", confidence=0.8, input_tokens=10, used_tokens=10,
                                       max_length=512, truncated=False, device="cpu", model_input=text)

    class FakeProvider:
        def generate(self, context):
            calls["provider"].append(context)
            if refusal:
                raise GenerationError("refusal", "OpenAI refused this request. No reply draft was created.")
            return GeneratedResult(details=GeneratedDetails(priority="High", product_feature=None,
                suggested_reply="Thanks for explaining the issue. Could you share your browser version?"),
                receipt=GenerationReceipt(model="mock-model-not-live", attempts=1, duration_seconds=0.1,
                                          completed_at_utc="test-only"))

    def factory(root):
        calls["factories"] += 1
        return FakeProvider()

    monkeypatch.setattr(IntentClassifier, "from_checkpoint", staticmethod(lambda *args, **kwargs: FakeCategory()))
    monkeypatch.setattr(SentimentClassifier, "from_checkpoint", staticmethod(lambda *args, **kwargs: FakeSentiment()))
    monkeypatch.setattr(generation_module, "create_provider", factory)
    return AppTest.from_file(str(APP), default_timeout=20).run(), calls


def submit(app, message="  Hi,\ncheckout crashes  "):
    app.text_input("single_ticket_id").set_value("A1")
    app.text_area("single_message").set_value(message)
    return app.button("complete_ticket").click().run()


def visible(app):
    return "\n".join(item.value for group in (app.markdown, app.info, app.caption, app.warning, app.error, app.success, app.text) for item in group)


def test_action_makes_one_request_displays_draft_and_reruns_do_no_work(tmp_path, monkeypatch):
    app, calls = setup_app(tmp_path, monkeypatch)
    assert not app.exception and not calls["provider"]
    assert not app.button("complete_ticket").disabled
    submit(app)
    assert not app.exception
    result = app.session_state["single_ticket_result"]
    assert result.status == "processed" and result.reply_is_draft
    assert result.ticket.message == "  Hi,\ncheckout crashes  "
    assert result.category.category == "Bug Report" and result.sentiment.sentiment == "Negative"
    assert len(calls["provider"]) == 1
    assert "Thanks for explaining" in visible(app)
    assert "unsent draft" in visible(app)
    assert "unit-private-test-key" not in visible(app) + str(app.session_state)
    app.run()
    assert len(calls["provider"]) == 1
    submit(app)
    assert len(calls["provider"]) == 1  # duplicate explicit submit still reuses the result
    app.checkbox("single_reprocess").check()
    submit(app)
    assert len(calls["provider"]) == 2


def test_new_input_requires_explicit_action_and_does_not_show_stale_result_after_submit(tmp_path, monkeypatch):
    app, calls = setup_app(tmp_path, monkeypatch)
    submit(app)
    app.text_area("single_message").set_value("The upload freezes")
    assert len(calls["provider"]) == 1
    app.button("complete_ticket").click().run()
    assert len(calls["provider"]) == 2
    assert app.session_state["single_ticket_result"].ticket.message == "The upload freezes"
    submit(app)
    assert len(calls["provider"]) == 2  # returning to earlier input also reuses its paid result


def test_missing_key_is_visible_and_local_tools_remain_available(tmp_path, monkeypatch):
    app, calls = setup_app(tmp_path, monkeypatch, key=False)
    assert app.button("complete_ticket").disabled
    assert "OPENAI_API_KEY" in visible(app)
    assert not calls["provider"]
    app.button("load_sample").click().run()
    assert not app.button("analyze_categories").disabled
    assert not calls["provider"]
    app.button("show_live_check").click().run()
    assert "No successful saved live API check" in visible(app)


@pytest.mark.parametrize("message,category", [("Hello!", "Bug Report"), ("Long greeting", "Non-Qualifying")])
def test_ignored_ticket_never_constructs_provider_or_calls_sentiment(tmp_path, monkeypatch, message, category):
    app, calls = setup_app(tmp_path, monkeypatch, category=category)
    submit(app, message)
    assert app.session_state["single_ticket_result"].status == "ignored"
    assert not calls["sentiment"] and not calls["provider"] and calls["factories"] == 0


@pytest.mark.parametrize("failure", ["refusal", "sentiment"])
def test_failure_preserves_local_fields_as_error_never_spam(tmp_path, monkeypatch, failure):
    app, calls = setup_app(tmp_path, monkeypatch, refusal=failure == "refusal", sentiment_fail=failure == "sentiment")
    submit(app)
    assert not app.exception
    result = app.session_state["single_ticket_result"]
    assert result.status == "error" and result.category.category == "Bug Report"
    assert result.generated is None
    assert "RAW_LOCAL_EXCEPTION" not in visible(app)
    app.run()
    assert len(calls["provider"]) == int(failure == "refusal")


def test_blank_id_is_input_error_before_models_or_provider(tmp_path, monkeypatch):
    app, calls = setup_app(tmp_path, monkeypatch)
    app.text_input("single_ticket_id").set_value(" ")
    app.text_area("single_message").set_value("App crashes")
    app.button("complete_ticket").click().run()
    assert "Input error" in visible(app)
    assert not calls["category"] and not calls["provider"]


def test_invalid_input_does_not_erase_previous_successful_request_cache(tmp_path, monkeypatch):
    app, calls = setup_app(tmp_path, monkeypatch)
    submit(app)
    app.text_input("single_ticket_id").set_value(" ")
    app.button("complete_ticket").click().run()
    assert "Input error" in visible(app)
    submit(app)
    assert app.session_state["single_ticket_result"].status == "processed"
    assert len(calls["provider"]) == 1


def test_key_setup_refresh_never_calls_api(tmp_path, monkeypatch):
    app, calls = setup_app(tmp_path, monkeypatch, key=False)
    path = tmp_path / ".env"
    path.write_text(path.read_text().replace("OPENAI_API_KEY=\n", "OPENAI_API_KEY=unit-new-key\n"), encoding="utf-8")
    app.button("refresh_api_setup").click().run()
    assert not app.button("complete_ticket").disabled
    assert not calls["provider"] and not calls["category"]
