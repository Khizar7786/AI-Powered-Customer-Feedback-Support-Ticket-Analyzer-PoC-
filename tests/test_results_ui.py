"""Full review/download behavior with explicit mocks and real temporary SQLite writes."""

from contextlib import closing
import json
import sqlite3

from streamlit.delta_generator import DeltaGenerator

from feedback_analyzer import results_ui
from feedback_analyzer import config as config_module
from feedback_analyzer.exports import ExportError
from feedback_analyzer.pipeline import analyze_csv
from feedback_analyzer.results_ui import ProgressStore
from feedback_analyzer.storage import OutcomeStore
from test_app_batch import setup, clear_cache


def runs(tmp_path):
    with closing(sqlite3.connect(tmp_path / "outputs/tickets.db")) as db:
        return db.execute("SELECT count(*) FROM batch_runs").fetchone()[0]


def processed_table(app):
    return next(tab for tab in app.tabs if tab.label == "Processed").dataframe[0].value


def test_filters_details_downloads_widgets_and_reruns_do_not_repeat_work(tmp_path, monkeypatch, batch_setup):
    s = batch_setup
    captured = {}
    original = DeltaGenerator.download_button

    def download(self, label, **kwargs):
        captured[kwargs["key"]] = kwargs
        return original(self, label, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", download)
    app, _ = setup(tmp_path, monkeypatch, s)
    app.button("validate_csv").click().run()
    app.button("analyze_tickets").click().run()
    assert not app.exception
    before = {key: list(value) for key, value in s.calls.items()}
    original_bundle = app.session_state["batch_exports"]
    original_hash = original_bundle.files["batch.json"]
    for key, value in (("batch_category_filter", "Bug Report"), ("batch_priority_filter", "Medium"), ("batch_sentiment_filter", "Negative")):
        app.selectbox(key).select(value).run()
        assert not app.exception and len(processed_table(app)) == 1
    app.selectbox("batch_detail_row").select(3).run()  # Missing ID retains source identity.
    assert any("Missing ticket_id" in item.value for item in app.error)
    app.selectbox("batch_detail_row").select(1).run()
    assert any("Thanks for reporting" in item.value for item in app.text)
    app.checkbox("batch_retry_errors").check().run()  # A widget change is not an Analyze action.
    app.run()
    assert s.calls == before and runs(tmp_path) == 1
    assert app.session_state["batch_exports"].files["batch.json"] == original_hash
    assert len(captured) == 4 and all(item["on_click"] == "ignore" for item in captured.values())
    assert captured["download_batch"]["data"] == original_bundle.files["batch.json"]
    assert json.loads(captured["download_batch"]["data"])["total_rows"] == 3
    assert (tmp_path / "outputs/results.csv").read_bytes() == original_bundle.files["results.csv"]


def test_empty_filter_and_new_input_clear_stale_exports_and_filters(tmp_path, monkeypatch, batch_setup):
    s = batch_setup
    app, selected = setup(tmp_path, monkeypatch, s)
    app.button("validate_csv").click().run()
    app.button("analyze_tickets").click().run()
    app.selectbox("batch_priority_filter").select("Medium").run()
    selected["data"] = b"ticket_id,message\nNEW,Hello\n"
    app.button("validate_csv").click().run()
    assert "batch_results" not in app.session_state and "batch_exports" not in app.session_state
    app.button("analyze_tickets").click().run()
    assert not app.exception and not len(processed_table(app))
    assert app.selectbox("batch_priority_filter").value == "All"
    assert json.loads(app.session_state["batch_exports"].files["batch.json"])["rows"][0]["outcome"]["ticket_id"] == "NEW"
    assert len(s.calls["api"]) == 1


def test_export_save_failure_can_retry_without_reanalysis(tmp_path, monkeypatch, batch_setup):
    s = batch_setup
    app, _ = setup(tmp_path, monkeypatch, s)
    real_save = results_ui.save_exports
    monkeypatch.setattr(results_ui, "save_exports", lambda *args: (_ for _ in ()).throw(ExportError("Mock write failed; retry saving exports.")))
    app.button("validate_csv").click().run()
    app.button("analyze_tickets").click().run()
    assert not app.exception and app.session_state["batch_results"].fully_processed == 1
    assert app.session_state["batch_exports"] and not app.session_state["batch_exports_saved"]
    before = {key: list(value) for key, value in s.calls.items()}
    monkeypatch.setattr(results_ui, "save_exports", real_save)
    app.button("retry_save_exports").click().run()
    assert not app.exception and app.session_state["batch_exports_saved"]
    assert s.calls == before and runs(tmp_path) == 1


def test_progress_reports_each_attempt_and_callback_failure_does_not_repeat_work(tmp_path, batch_setup):
    s = batch_setup
    completed = []

    def notify(number):
        completed.append(number)
        if number == 1:
            raise RuntimeError("Mock presentation failure")

    store = ProgressStore(OutcomeStore(tmp_path / "tickets.db"), notify)
    result = analyze_csv(b"ticket_id,message\nA,App crashes\nB,Hello\n,Missing ID\n", s.runtime, store, s.identity)
    assert completed == [1, 2, 3] and result.counts == {"processed": 1, "ignored": 1, "error": 1}
    assert len(s.calls["api"]) == 1


def test_filters_reduce_rows_and_empty_combinations_do_not_change_exports(tmp_path, monkeypatch, batch_setup):
    s = batch_setup
    category_predict, sentiment_predict, generate = s.runtime.category.predict, s.runtime.sentiment.predict, s.provider.generate

    def category(text):
        value = category_predict(text)
        return value.model_copy(update={"category": "Feature Request"}) if text == "Feature idea" else value

    def sentiment(text):
        value = sentiment_predict(text)
        return value.model_copy(update={"sentiment": "Positive"}) if text == "Feature idea" else value

    def provider(context):
        value = generate(context)
        return value.model_copy(update={"details": value.details.model_copy(update={"priority": "High"})}) if context.message == "Bug issue" else value

    s.runtime.category.predict, s.runtime.sentiment.predict, s.provider.generate = category, sentiment, provider
    app, _ = setup(tmp_path, monkeypatch, s, b"ticket_id,message\nA,Bug issue\nB,Feature idea\nC,Hello\n,Missing ID\n")
    app.button("validate_csv").click().run()
    app.button("analyze_tickets").click().run()
    original_json = app.session_state["batch_exports"].files["batch.json"]
    app.selectbox("batch_category_filter").select("Feature Request").run()
    assert processed_table(app)["ticket_id"].tolist() == ["B"]
    app.selectbox("batch_sentiment_filter").select("Positive").run()
    assert processed_table(app)["ticket_id"].tolist() == ["B"]
    app.selectbox("batch_priority_filter").select("High").run()
    assert not len(processed_table(app))
    assert {m.label: m.value for m in app.metric}["Processed"] == "2"
    assert app.session_state["batch_exports"].files["batch.json"] == original_json
    assert len(s.calls["api"]) == 2 and runs(tmp_path) == 1


def test_missing_key_and_missing_model_show_actionable_readiness(tmp_path, monkeypatch, batch_setup):
    real_load_config = config_module.load_config
    s = batch_setup
    app, _ = setup(tmp_path, monkeypatch, s)
    (tmp_path / ".env").write_text("LLM_PROVIDER=openai\nLLM_MODEL=test-model\nOPENAI_API_KEY=\n")
    config = real_load_config(tmp_path, environ={})
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    app.button("validate_csv").click().run()
    assert app.button("analyze_tickets").disabled
    assert any("OPENAI_API_KEY" in message.value for message in app.warning)
    (config.classifier_dir / "model.safetensors").unlink()
    app.run()
    assert any("python -m scripts.train_intent" in block.value for block in app.code)
    assert not any(s.calls.values())
