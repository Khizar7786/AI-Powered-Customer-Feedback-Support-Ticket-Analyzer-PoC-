"""Explicit batch analysis with readiness and progress; review uses saved session data."""

import hashlib
import json

import streamlit as st

from feedback_analyzer import config as config_module, generation, pipeline
from feedback_analyzer.category_analysis import checkpoint_signature
from feedback_analyzer.exports import ExportError, build_exports
from feedback_analyzer.pipeline_identity import build_identity
from feedback_analyzer.readiness import inspect_model_files
from feedback_analyzer.results_ui import ProgressStore, clear_batch_result, render_results, save_bundle
from feedback_analyzer.storage import OutcomeStore, StorageError


def render_batch_action(preview, category_loader, sentiment_loader):
    config = config_module.load_config()
    category_signature = checkpoint_signature(config.classifier_dir)
    sentiment_signature = checkpoint_signature(config.sentiment_dir)
    option_error = None
    try:
        options = generation.load_generation_options(config.project_root).model_dump(mode="json")
    except generation.GenerationError as error:
        options, option_error = None, str(error)
    stamp = hashlib.sha256(json.dumps([category_signature, sentiment_signature, options, generation.PROMPT_VERSION], sort_keys=True).encode()).hexdigest()
    if st.session_state.get("batch_setup_stamp") != stamp:
        clear_batch_result()
        st.session_state["batch_setup_stamp"] = stamp
    st.subheader("Analyze and save this batch")
    st.caption("Qualifying messages go to OpenAI. Matching results are reused from SQLite. Local batch models run on CPU; replies remain unsent drafts.")
    reprocess = st.checkbox("Allow analysis of changed messages or configuration", key="batch_reprocess")
    retry_errors = st.checkbox("Retry previous errors explicitly (may make paid requests)", key="batch_retry_errors")
    needs_models = preview is not None and any(row.status == "candidate" for row in preview.rows)
    category_ready = inspect_model_files(config.classifier_dir).state == "Detected (unverified)"
    sentiment_ready = inspect_model_files(config.sentiment_dir).state == "Detected (unverified)"
    if not category_ready:
        st.warning("The category checkpoint is missing or incomplete. Train separately or restore the saved checkpoint.")
        st.code("python -m scripts.train_intent", language="powershell")
    if not sentiment_ready:
        st.warning("The sentiment weights are missing or incomplete. Download the existing model separately.")
        st.code("python -m scripts.evaluate_sentiment --download-model", language="powershell")
    if option_error:
        st.warning(option_error)
    elif not config.api_key_configured:
        st.warning("Set OPENAI_API_KEY in ignored .env, then refresh this page. Never paste the key into a ticket or chat.")
    else:
        st.caption("OpenAI configuration is present. Presence alone does not verify account access. Keys are never displayed.")
    enabled = preview is not None and bool(preview.rows) and not preview.file_errors
    if needs_models:
        enabled = enabled and category_ready and sentiment_ready and options is not None and config.api_key_configured
    progress_slot = st.empty()
    if st.button("Analyze tickets", key="analyze_tickets", disabled=not enabled, type="primary"):
        if st.session_state.get("batch_in_progress"):
            st.warning("This batch is already running.")
            return
        st.session_state["batch_in_progress"] = True
        clear_batch_result()
        total = len(preview.rows)
        progress = progress_slot.progress(0., text="Preparing analysis…")
        try:
            with st.spinner("Analyzing and saving the batch…"):
                identity = build_identity(config, "cpu")
                runtime = pipeline.BatchRuntime.local(config, "cpu",
                    category_loader=lambda: category_loader(category_signature, "cpu"),
                    sentiment_loader=lambda: sentiment_loader(sentiment_signature, "cpu"))
                store = ProgressStore(OutcomeStore(config.project_root / "outputs/tickets.db"),
                                      lambda done: progress.progress(done / total, text=f"Reviewed {done} of {total} tickets"))
                result = pipeline.analyze_batch(preview, runtime, store,
                    identity, source_name=st.session_state.get("csv_preview_source", "input.csv"),
                    source_hash=st.session_state.get("csv_source_hash"), reprocess=reprocess, retry_errors=retry_errors)
                st.session_state["batch_results"] = result
                try:
                    bundle = build_exports(result, identity.metadata)
                    st.session_state["batch_exports"] = bundle
                    save_bundle(bundle, config.project_root / "outputs")
                except ExportError as error:
                    st.session_state["batch_export_error"] = str(error)
        except (OSError, StorageError, ValueError):
            st.session_state.pop("batch_results", None)
            st.session_state["batch_error"] = "Batch setup/storage failed. Check the output directory, local models and configuration. No successful batch is claimed."
        finally:
            st.session_state["batch_in_progress"] = False
    if st.session_state.get("batch_error"):
        st.error(st.session_state["batch_error"])
    result = st.session_state.get("batch_results")
    if result is not None:
        progress_slot.progress(1., text=f"Batch complete: {len(result.rows)} tickets reviewed")
        render_results(result, config.project_root / "outputs")
