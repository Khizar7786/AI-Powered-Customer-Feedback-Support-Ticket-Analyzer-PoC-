"""Local previews, explicit single-ticket analysis and persistent batch orchestration."""

from pathlib import Path
import hashlib

import streamlit as st

from feedback_analyzer import ingestion, intent, sentiment
from feedback_analyzer.category_analysis import CategoryBatch, analyze_categories, checkpoint_signature
from feedback_analyzer.config import CLASSIFIER_MODEL_ID, SENTIMENT_MODEL_ID, load_config
from feedback_analyzer.readiness import inspect_api_config, inspect_device, inspect_model_files
from feedback_analyzer.sentiment_analysis import SentimentBatch, analyze_sentiments
from feedback_analyzer.single_ticket_ui import render_single_ticket
from feedback_analyzer.batch_ui import render_batch_action
from feedback_analyzer.results_ui import clear_batch_result


@st.cache_resource
def device_status():
    """Inspect the device once per app process; no model is loaded."""
    return inspect_device()


@st.cache_resource(max_entries=1, show_spinner=False)
def category_classifier(signature: tuple, device: str):
    """Shared locked inference resource; called only after an explicit action."""
    return intent.IntentClassifier.from_checkpoint(Path(signature[0]), device=device)


@st.cache_resource(max_entries=1, show_spinner=False)
def sentiment_classifier(signature: tuple, device: str):
    return sentiment.SentimentClassifier.from_checkpoint(Path(signature[0]), device=device)


def _clear_preview() -> None:
    """Changing the selected file invalidates the previous preview."""
    st.session_state.pop("csv_preview", None)
    st.session_state.pop("csv_preview_source", None)
    st.session_state.pop("csv_source_hash", None)
    clear_batch_result()
    _clear_categories()


def _clear_categories() -> None:
    for key in ("category_results", "category_result_key", "category_error", "category_device_warning", "category_device"):
        st.session_state.pop(key, None)
    _clear_sentiments()


def _clear_sentiments() -> None:
    for key in ("sentiment_results", "sentiment_result_key", "sentiment_error", "sentiment_device_warning", "sentiment_actual_device"):
        st.session_state.pop(key, None)


def render_setup() -> None:
    config = load_config()
    device = device_status()
    st.subheader("Execution device")
    st.write(f"**{device.device}** — {device.detail}")
    if device.warning:
        st.warning(device.warning)
    st.caption("Device information is a startup snapshot. Free VRAM changes as other applications run.")

    st.subheader("Category classifier")
    classifier = inspect_model_files(config.classifier_dir)
    st.write(f"**{classifier.state}** — {classifier.detail}")
    st.caption(f"{CLASSIFIER_MODEL_ID} · Local checkpoint: {config.classifier_dir}")
    st.write("Train the six-category classifier separately using the frozen reviewed splits. See docs/training.md.")
    st.caption("Training command (run in a terminal):")
    st.code("python -m scripts.train_intent", language="powershell")

    st.subheader("Sentiment model")
    sentiment = inspect_model_files(config.sentiment_dir)
    st.write(f"**{sentiment.state}** — {sentiment.detail}")
    st.caption(f"{SENTIMENT_MODEL_ID} · Local model directory: {config.sentiment_dir}")
    st.write("Use the existing Cardiff sentiment weights locally. CPU is the default; no sentiment training is needed.")
    st.code("python -m scripts.evaluate_sentiment --download-model", language="powershell")
    st.caption("Setup downloads the model once. Sentiment inference runs only after its explicit action. See docs/sentiment.md.")

    st.subheader("Generative API configuration")
    api = inspect_api_config(config)
    st.write(f"**{api.state}** — {api.detail}")
    st.write("OpenAI is the selected provider. Set LLM_PROVIDER=openai, LLM_MODEL and OPENAI_API_KEY in ignored local .env. See docs/generation.md.")
    st.caption("Only configuration presence is inspected. Credentials are never displayed; no API call is made.")


def render_preview(preview: ingestion.CsvPreview) -> None:
    st.caption(f"Preview source: {st.session_state['csv_preview_source']}")
    for error in preview.file_errors:
        st.error(error)

    counts = {status: sum(row.status == status for row in preview.rows)
              for status in ("candidate", "ignored", "error")}
    for column, label, value in zip(
        st.columns(4),
        ("Rows read", "Candidates", "Ignored", "Input errors"),
        (len(preview.rows), counts["candidate"], counts["ignored"], counts["error"]),
    ):
        column.metric(label, value)

    st.caption(
        "CSV line numbers include the header; quoted messages can span several lines. "
        "Input errors count invalid records. File-level errors appear above."
    )
    st.info(
        "This preview shows input checks only. Use Analyze categories below to run the local classifier. "
        "A category prediction is partial analysis, not a fully processed ticket."
    )
    tabs = st.tabs(["Candidates", "Ignored", "Input errors"])
    base_columns = ["source_row", "source_line_end", "ticket_id", "original_message"]
    for tab, status in zip(tabs, ("candidate", "ignored", "error")):
        with tab:
            columns = base_columns + (["normalized_message"] if status == "candidate" else ["reason"])
            if status == "error":
                columns += ["raw_record"]
            records = []
            for row in preview.rows:
                if row.status != status:
                    continue
                record = {column: getattr(row, column, None) for column in columns}
                if status != "candidate":
                    record["reason"] = "; ".join(row.issues)
                records.append(record)
            st.dataframe(
                records or {column: [] for column in columns},
                hide_index=True,
                width="stretch",
                column_config={
                    "source_row": st.column_config.NumberColumn("CSV start line", format="%d"),
                    "source_line_end": st.column_config.NumberColumn("CSV end line", format="%d"),
                    "original_message": "Original message",
                    "normalized_message": "Normalized input",
                    "reason": "Reason",
                    "raw_record": "Original CSV record",
                },
            )
            if not records:
                st.caption("No rows in this group.")

    if preview.rows:
        selected = st.selectbox(
            "Inspect a source row",
            preview.rows,
            format_func=lambda row: f"Line {row.source_row} · {row.ticket_id or 'missing ID'} · {row.status}",
        )
        if selected.original_message is not None:
            st.caption("Original message (whitespace preserved)")
            st.code(selected.original_message, language=None, wrap_lines=True)
        elif getattr(selected, "raw_record", None) is not None:
            st.caption("Original CSV record (message could not be parsed)")
            st.code(selected.raw_record, language=None, wrap_lines=True)


def render_categories(batch: CategoryBatch) -> None:
    st.subheader("Category results · partial analysis")
    st.info(
        "This table shows category and qualification. Run sentiment separately below; priority, product/feature "
        "and reply are still pending. Fully processed tickets: 0. Confidence is the model's score, not a calibrated guarantee; "
        "low confidence alone does not make a message spam."
    )
    if st.session_state.get("category_device_warning"):
        st.warning(st.session_state["category_device_warning"])
    st.caption(f"Current classifier device: {st.session_state.get('category_device', 'unknown')}. Errors include input validation and inference failures.")
    for column, label, value in zip(
        st.columns(4), ("Partial analysis", "Ignored after category checks", "Category errors", "Fully processed"),
        (batch.counts["partial"], batch.counts["ignored"], batch.counts["error"], batch.fully_processed),
    ):
        column.metric(label, value)
    tabs = st.tabs(["Partial analysis", "Ignored after category checks", "Category errors"])
    columns = ["source_row", "source_line_end", "ticket_id", "original_message", "normalized_message", "category",
               "confidence", "classifier_input_truncated", "input_tokens", "used_tokens", "max_length", "device", "reason"]
    for tab, status in zip(tabs, ("partial", "ignored", "error")):
        records = []
        for row in batch.rows:
            if row.status != status:
                continue
            record = {name: getattr(row, name) for name in (
                "source_row", "source_line_end", "ticket_id", "original_message", "normalized_message", "reason",
            )}
            prediction = row.prediction
            record.update(category=prediction.category if prediction else None,
                          confidence=prediction.confidence if prediction else None,
                          classifier_input_truncated=prediction.truncated if prediction else None,
                          input_tokens=prediction.input_tokens if prediction else None,
                          used_tokens=prediction.used_tokens if prediction else None,
                          max_length=prediction.max_length if prediction else None,
                          device=prediction.device if prediction else None)
            records.append(record)
        with tab:
            st.dataframe(records or {name: [] for name in columns}, hide_index=True, width="stretch", column_config={
                "confidence": st.column_config.NumberColumn("Category confidence", format="%.3f"),
                "classifier_input_truncated": "Classifier input truncated?",
                "original_message": "Original message",
            })
            if not records:
                st.caption("No rows in this group.")
    truncated = [row for row in batch.rows if row.prediction and row.prediction.truncated]
    if truncated:
        st.warning(
            f"{len(truncated)} message(s) exceeded the classifier token limit. Only the beginning was classified; "
            "the original text is preserved in full."
        )


def category_action(preview: ingestion.CsvPreview | None) -> None:
    config = load_config()
    device_choice = st.selectbox("Classifier device", ["Auto", "CPU"], key="intent_device")
    device = "cpu" if device_choice == "CPU" else "auto"
    signature = checkpoint_signature(config.classifier_dir)
    result_key = (signature, device)
    if st.session_state.get("category_result_key") != result_key:
        _clear_categories()
    files = inspect_model_files(config.classifier_dir)
    available = files.state == "Detected (unverified)"
    candidates = preview is not None and any(row.status == "candidate" for row in preview.rows)
    enabled = available and candidates and not preview.file_errors
    if not available:
        st.info("Category checkpoint is missing or incomplete. Train separately, or set CLASSIFIER_DIR to the saved checkpoint.")
        st.code("python -m scripts.train_intent", language="powershell")
        st.caption("See docs/training.md for setup and checkpoint transfer instructions.")
    already_analyzed = "category_results" in st.session_state
    action = "Reanalyze categories" if already_analyzed else "Analyze categories"
    if st.button(action, key="analyze_categories", disabled=not enabled):
        _clear_categories()
        st.session_state["category_result_key"] = result_key
        try:
            with st.spinner("Running the local category classifier…"):
                classifier = category_classifier(signature, device)
                result = analyze_categories(preview, classifier)
            st.session_state["category_results"] = result
            st.session_state["category_device"] = classifier.device
            st.session_state["category_device_warning"] = classifier.device_warning
        except (intent.IntentSetupError, intent.IntentInferenceError, ValueError, OSError, RuntimeError):
            st.session_state["category_error"] = (
                "The local classifier could not be loaded or run. No category results were created. "
                "Check CLASSIFIER_DIR, verify the saved checkpoint, or select CPU and retry."
            )
    if st.session_state.get("category_error"):
        st.error(st.session_state["category_error"])
        st.code("python -m scripts.train_intent --verify-only", language="powershell")
        st.code("python -m scripts.train_intent", language="powershell")
    if "category_results" in st.session_state:
        render_categories(st.session_state["category_results"])


def render_sentiments(batch: SentimentBatch) -> None:
    st.subheader("Category and sentiment · partial analysis")
    st.info(
        "Sentiment is shown only for qualifying tickets. Priority, product/feature and reply are still pending; "
        "fully processed tickets remain 0. This social-media model may interpret support requests differently "
        "from a human reviewer. Scores are not calibrated guarantees."
    )
    if st.session_state.get("sentiment_device_warning"):
        st.warning(st.session_state["sentiment_device_warning"])
    st.caption(f"Current sentiment device: {st.session_state.get('sentiment_actual_device', 'unknown')}")
    failures = [row for row in batch.rows if row.status == "error" and row.category_row.status == "partial"]
    left, right = st.columns(2)
    left.metric("With sentiment", batch.counts["partial"])
    right.metric("Sentiment inference errors", len(failures))
    records = []
    for row in batch.rows:
        category = row.category_row
        if category.status != "partial":
            continue
        prediction = row.prediction
        records.append({
            "source_row": category.source_row, "source_line_end": category.source_line_end,
            "ticket_id": category.ticket_id, "original_message": category.original_message,
            "normalized_message": category.normalized_message, "category": category.prediction.category,
            "classifier_input_truncated": category.prediction.truncated,
            "sentiment": prediction.sentiment if prediction else None,
            "sentiment_confidence": prediction.confidence if prediction else None,
            "sentiment_input_truncated": prediction.truncated if prediction else None,
            "sentiment_input_tokens": prediction.input_tokens if prediction else None,
            "sentiment_used_tokens": prediction.used_tokens if prediction else None,
            "sentiment_max_length": prediction.max_length if prediction else None,
            "sentiment_model_input": prediction.model_input if prediction else None,
            "sentiment_device": prediction.device if prediction else None,
            "status": row.status, "reason": row.reason,
        })
    columns = ["source_row", "ticket_id", "original_message", "category", "sentiment", "reason"]
    tabs = st.tabs(["Category + sentiment (partial)", "Sentiment errors"])
    for tab, status in zip(tabs, ("partial", "error")):
        with tab:
            filtered = [row for row in records if row["status"] == status]
            st.dataframe(filtered or {key: [] for key in columns}, hide_index=True, width="stretch", column_config={
                "sentiment_confidence": st.column_config.NumberColumn("Sentiment confidence", format="%.3f"),
                "sentiment_input_truncated": "Sentiment input truncated?",
                "original_message": "Original message",
            })
            if not filtered:
                st.caption("No rows in this group.")
    if any(row.prediction and row.prediction.truncated for row in batch.rows):
        st.warning("Some sentiment inputs were truncated. Only the beginning was analyzed; full original messages are preserved.")
    st.caption("Ignored and invalid rows keep their category-check outcomes above; they receive no sentiment label.")


def sentiment_action() -> None:
    config = load_config()
    choice = st.selectbox("Sentiment device", ["CPU", "Auto"], key="sentiment_device")
    device = "cpu" if choice == "CPU" else "auto"
    categories = st.session_state.get("category_results")
    signature = checkpoint_signature(config.sentiment_dir)
    category_fingerprint = hashlib.sha256(categories.model_dump_json().encode()).hexdigest() if categories else None
    result_key = (signature, device, category_fingerprint)
    if st.session_state.get("sentiment_result_key") != result_key:
        _clear_sentiments()
    available = inspect_model_files(config.sentiment_dir).state == "Detected (unverified)"
    qualifying = categories is not None and categories.counts["partial"] > 0
    if not available:
        st.info("Sentiment weights are missing or incomplete. Download the existing model separately; no training is required.")
        st.code("python -m scripts.evaluate_sentiment --download-model", language="powershell")
    label = "Reanalyze sentiment" if "sentiment_results" in st.session_state else "Analyze sentiment"
    if st.button(label, key="analyze_sentiment", disabled=not (available and qualifying)):
        _clear_sentiments()
        st.session_state["sentiment_result_key"] = result_key
        try:
            with st.spinner("Running local sentiment on qualifying tickets…"):
                classifier = sentiment_classifier(signature, device)
                results = analyze_sentiments(categories, classifier)
            st.session_state["sentiment_results"] = results
            st.session_state["sentiment_actual_device"] = classifier.device
            st.session_state["sentiment_device_warning"] = classifier.device_warning
        except (sentiment.SentimentSetupError, sentiment.SentimentInferenceError, ValueError, OSError, RuntimeError):
            st.session_state["sentiment_error"] = (
                "Sentiment could not be loaded or run. Category results are retained. "
                "Check SENTIMENT_DIR, restore the complete model, or select CPU and retry."
            )
    if st.session_state.get("sentiment_error"):
        st.error(st.session_state["sentiment_error"])
        st.code("python -m scripts.evaluate_sentiment --download-model", language="powershell")
    if "sentiment_results" in st.session_state:
        render_sentiments(st.session_state["sentiment_results"])


def main() -> None:
    st.set_page_config(page_title="Feedback Triage AI", page_icon="💬", layout="wide")
    st.title("Feedback Triage AI")
    st.caption("Review customer feedback, prepare reply drafts and export the results.")
    st.subheader("Upload and preview tickets")
    st.write("Upload a UTF-8 CSV with the columns `ticket_id,message`, or try the included sample.")
    st.caption(
        "The labeled training dataset is a separate input. "
        "Validation preserves your original messages and makes no AI or API requests."
    )
    uploaded = st.file_uploader(
        "Upload tickets CSV", type=["csv"], key="ticket_upload", on_change=_clear_preview,
    )
    sample_column, validate_column = st.columns(2)
    load_sample = sample_column.button("Load sample CSV", key="load_sample")
    validate = validate_column.button("Validate CSV", key="validate_csv", disabled=uploaded is None)
    if load_sample:
        _clear_preview()
        try:
            data = (Path(__file__).resolve().parent / "data" / "sample_tickets.csv").read_bytes()
        except OSError:
            st.error("The sample CSV could not be read. Check data/sample_tickets.csv in the project folder.")
        else:
            st.session_state["csv_preview"] = ingestion.load_ticket_csv(data)
            st.session_state["csv_preview_source"] = "sample_tickets.csv"
            st.session_state["csv_source_hash"] = hashlib.sha256(data).hexdigest()
    elif validate and uploaded is not None:
        clear_batch_result()
        data = uploaded.getvalue()
        st.session_state["csv_preview"] = ingestion.load_ticket_csv(data)
        st.session_state["csv_preview_source"] = uploaded.name
        st.session_state["csv_source_hash"] = hashlib.sha256(data).hexdigest()

    if "csv_preview" in st.session_state:
        with st.expander("Review input validation", expanded="batch_results" not in st.session_state):
            render_preview(st.session_state["csv_preview"])
    else:
        st.info("Select a CSV and click Validate CSV, or click Load sample CSV to see a preview.")

    render_batch_action(st.session_state.get("csv_preview"), category_classifier, sentiment_classifier)
    with st.expander("Local model checks and single-ticket analysis"):
        category_action(st.session_state.get("csv_preview"))
        sentiment_action()
        render_single_ticket(category_classifier, sentiment_classifier)
        st.caption("Local category/sentiment buttons make no API calls. Complete batch and single-ticket actions use OpenAI only for qualifying messages.")
    with st.expander("Model and API setup"):
        render_setup()


if __name__ == "__main__":
    main()
