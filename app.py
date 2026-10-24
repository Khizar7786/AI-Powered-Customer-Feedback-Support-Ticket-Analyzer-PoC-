"""Upload and validate tickets locally; AI analysis is a later stage."""

from pathlib import Path

import streamlit as st

from feedback_analyzer import ingestion
from feedback_analyzer.config import CLASSIFIER_MODEL_ID, SENTIMENT_MODEL_ID, load_config
from feedback_analyzer.readiness import inspect_api_config, inspect_device, inspect_model_files


@st.cache_resource
def device_status():
    """Inspect the device once per app process; no model is loaded."""
    return inspect_device()


def _clear_preview() -> None:
    """Changing the selected file invalidates the previous preview."""
    st.session_state.pop("csv_preview", None)
    st.session_state.pop("csv_preview_source", None)


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
    st.write("Stage 4 will fine-tune the six-category classifier after reviewed data is ready.")
    st.caption("Planned training command; available after Stage 4 is implemented:")
    st.code("python -m scripts.train_intent", language="powershell")

    st.subheader("Sentiment model")
    sentiment = inspect_model_files(config.sentiment_dir)
    st.write(f"**{sentiment.state}** — {sentiment.detail}")
    st.caption(f"{SENTIMENT_MODEL_ID} · Local model directory: {config.sentiment_dir}")
    st.write("Stage 6 will download and verify the pretrained sentiment weights for local inference. No sentiment training is planned.")

    st.subheader("Generative API configuration")
    api = inspect_api_config(config)
    st.write(f"**{api.state}** — {api.detail}")
    st.write("In Stage 7, choose one provider and configure its model and key in your local .env file. See docs/setup.md.")
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
        "Candidates have passed input checks only. They have not been classified or processed. "
        "Longer greetings, spam, and other semantic decisions need the later classifier."
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


def main() -> None:
    st.set_page_config(page_title="Feedback Triage AI", page_icon="💬")
    st.title("Feedback Triage AI")
    st.caption("Customer feedback and support-ticket analysis · Stage 2")
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
    elif validate and uploaded is not None:
        st.session_state["csv_preview"] = ingestion.load_ticket_csv(uploaded.getvalue())
        st.session_state["csv_preview_source"] = uploaded.name

    if "csv_preview" in st.session_state:
        render_preview(st.session_state["csv_preview"])
    else:
        st.info("Select a CSV and click Validate CSV, or click Load sample CSV to see a preview.")

    st.button("Analyze tickets", key="analyze_tickets", disabled=True)
    st.caption("AI analysis is unavailable at this stage. No local models, API requests, or database writes run.")
    with st.expander("Model and API setup"):
        render_setup()


if __name__ == "__main__":
    main()
