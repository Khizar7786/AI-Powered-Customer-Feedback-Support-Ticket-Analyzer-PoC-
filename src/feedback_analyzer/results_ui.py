"""Review and download saved session results without model or database operations."""

import streamlit as st

from feedback_analyzer.exports import ExportError, flat_row, save_exports


VIEW_KEYS = ("batch_category_filter", "batch_priority_filter", "batch_sentiment_filter", "batch_detail_row")
ARTIFACT_KEYS = ("batch_exports", "batch_export_error", "batch_exports_saved")


def clear_batch_result():
    for key in ("batch_results", "batch_error", *VIEW_KEYS, *ARTIFACT_KEYS):
        st.session_state.pop(key, None)


class ProgressStore:
    """Observe row write attempts without changing analysis or its configuration identity."""
    def __init__(self, store, notify):
        self.store, self.notify = store, notify

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _write(self, method, run_id, row, *args):
        try:
            return method(run_id, row, *args)
        finally:
            # A presentation failure must never trigger a second paid analysis.
            try:
                self.notify(row.ordinal)
            except Exception:
                pass

    def record_row(self, run_id, row):
        return self._write(self.store.record_row, run_id, row)

    def finish_row(self, run_id, row, owner, metadata):
        return self._write(self.store.finish_row, run_id, row, owner, metadata)


def save_bundle(bundle, directory):
    try:
        save_exports(bundle, directory)
        st.session_state["batch_exports_saved"] = True
        st.session_state.pop("batch_export_error", None)
    except ExportError as error:
        st.session_state["batch_exports_saved"] = False
        st.session_state["batch_export_error"] = str(error)


def _detail(row):
    o, a = row.outcome, row.outcome.analysis
    st.subheader(f"Ticket details: {o.ticket_id or 'missing ID'}")
    st.caption(f"CSV lines {row.source_row}–{row.source_line_end}. Outcome: {o.status}. {'Reused stored result.' if row.reused else 'New outcome.'}")
    if o.original_message is not None:
        st.caption("Original message (whitespace preserved)")
        st.code(o.original_message, language=None, wrap_lines=True)
    if o.raw_record is not None:
        with st.expander("Original CSV record"):
            st.code(o.raw_record, language=None, wrap_lines=True)
    if o.error:
        st.error(f"{o.error.stage}: {o.error.message}")
    elif o.reason:
        st.info(o.reason)
    if a and a.category:
        st.write(f"Category: **{a.category.category}**")
        if a.category.truncated:
            st.warning(f"Classifier input truncated: {a.category.used_tokens}/{a.category.input_tokens} tokens used. The remainder was not classified.")
    if a and a.sentiment:
        st.write(f"Sentiment: **{a.sentiment.sentiment}**")
        if a.sentiment.truncated:
            st.warning(f"Sentiment input truncated: {a.sentiment.used_tokens}/{a.sentiment.input_tokens} tokens used.")
    if a and a.generated:
        details = a.generated.details
        st.write(f"Priority: **{details.priority}**")
        st.write(f"Product/feature: {details.product_feature or 'Not mentioned'}")
        st.caption("Reply draft — review before using. No reply has been sent.")
        st.text(details.suggested_reply)
    elif o.status == "error" and a:
        st.caption("Available fields are partial analysis. This ticket is not fully processed.")


def render_results(result, directory):
    st.subheader("Batch results")
    for column, label, value in zip(st.columns(5), ("Total tickets", "Processed", "Ignored", "Errors", "Reused"),
                                    (len(result.rows), result.counts["processed"], result.counts["ignored"], result.counts["error"], result.reused_count)):
        column.metric(label, value)
    for message in [*result.file_errors, *result.persistence_errors]:
        st.error(message)
    st.caption("Processed means every required field is validated and saved. Errors retain useful partials. Reply drafts are never sent.")
    processed = [r for r in result.rows if r.outcome.status == "processed"]
    records = {r.ordinal: flat_row(r) for r in result.rows}
    selections = {}
    for column, field, label in zip(st.columns(3), ("category", "priority", "sentiment"), ("Category", "Priority", "Sentiment")):
        options = sorted({records[r.ordinal][field] for r in processed if records[r.ordinal][field]})
        selections[field] = column.selectbox(label, ["All", *options], key=f"batch_{field}_filter")
    filtered = [r for r in processed if all(value == "All" or records[r.ordinal][field] == value for field, value in selections.items())]
    st.caption(f"Showing {len(filtered)} of {len(processed)} processed tickets. Filters do not change totals, ignored/errors or downloads.")
    for tab, rows in zip(st.tabs(["Processed", "Ignored", "Errors"]),
                         (filtered, [r for r in result.rows if r.outcome.status == "ignored"], [r for r in result.rows if r.outcome.status == "error"])):
        with tab:
            columns = ["source_row", "source_line_end", "ticket_id", "original_message", "category", "priority", "sentiment", "product_feature",
                       "ignored_reason", "error_stage", "error_message", "reused", "persisted"]
            st.dataframe([{key: records[r.ordinal][key] for key in columns} for r in rows] or {key: [] for key in columns}, hide_index=True, width="stretch")
            if not rows:
                st.info("No tickets in this view.")
    if result.rows:
        # Source ordinal uniquely identifies invalid and duplicate-ID rows too.
        selected = st.selectbox("Inspect a ticket", [r.ordinal for r in result.rows], key="batch_detail_row", format_func=lambda number:
                                f"Line {result.rows[number - 1].source_row} · {result.rows[number - 1].outcome.ticket_id or 'missing ID'} · {result.rows[number - 1].outcome.status}")
        _detail(result.rows[selected - 1])
    bundle = st.session_state.get("batch_exports")
    if bundle:
        st.subheader("Download this batch")
        st.caption("Downloads include the full batch, regardless of filters. CSV protects formula-like cells with an apostrophe; JSON retains exact originals.")
        for column, name, label, key in zip(st.columns(4), ("results.csv", "batch.json", "ignored.csv", "errors.json"),
                ("Processed CSV", "Complete batch JSON", "Ignored CSV", "Error details JSON"),
                ("download_processed", "download_batch", "download_ignored", "download_errors")):
            column.download_button(label, data=bundle.files[name], file_name=name, mime="text/csv; charset=utf-8" if name.endswith(".csv") else "application/json",
                                   key=key, on_click="ignore", width="stretch")
        if bundle.escaped_cells:
            st.caption(f"Spreadsheet protection applied to {bundle.escaped_cells} CSV text cell(s); originals remain in JSON and SQLite.")
    if st.session_state.get("batch_export_error"):
        st.error(st.session_state["batch_export_error"])
        if bundle and st.button("Retry saving exports", key="retry_save_exports"):
            save_bundle(bundle, directory)
            if st.session_state.get("batch_exports_saved"):
                st.success("Exports saved. No analysis was repeated.")
    if st.session_state.get("batch_exports_saved"):
        st.caption("Saved beside tickets.db: results.csv, ignored.csv, errors.json and batch.json. These files represent the latest saved batch.")
