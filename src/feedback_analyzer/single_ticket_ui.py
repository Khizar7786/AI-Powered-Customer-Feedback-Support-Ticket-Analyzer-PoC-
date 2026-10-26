"""Explicit single-ticket API action, independent of future batch/storage work."""

import hashlib
import json
from pathlib import Path
from typing import Callable

from pydantic import ValidationError
import streamlit as st

from feedback_analyzer import config as config_module, generation as generation_module
from feedback_analyzer.category_analysis import checkpoint_signature
from feedback_analyzer.readiness import inspect_model_files
from feedback_analyzer.schemas import TicketInput
from feedback_analyzer.ticket_analysis import TicketAnalysis, analyze_ticket


def render_ticket_result(result: TicketAnalysis) -> None:
    st.subheader(f"Single ticket · {result.ticket.ticket_id}")
    st.caption("Original message (whitespace preserved)")
    st.code(result.ticket.message, language=None, wrap_lines=True)
    if result.status == "ignored":
        st.info(f"Ignored: {result.reason}")
        return
    if result.status == "error":
        st.error(f"Error at {result.error.stage}: {result.error.message}")
    if result.category:
        st.write(f"**Category:** {result.category.category}")
        if result.category.truncated:
            st.warning(f"Category input truncated: {result.category.used_tokens}/{result.category.input_tokens} tokens used.")
    if result.sentiment:
        st.write(f"**Sentiment:** {result.sentiment.sentiment}")
        if result.sentiment.truncated:
            st.warning(f"Sentiment input truncated: {result.sentiment.used_tokens}/{result.sentiment.input_tokens} tokens used.")
    if result.status == "processed":
        details = result.generated.details
        st.success("Processed: all required analysis fields validated. This reply is an unsent draft; nothing was saved or sent to a customer.")
        st.write(f"**Priority:** {details.priority}")
        st.write(f"**Product/feature:** {details.product_feature or 'Not mentioned'}")
        st.caption("Suggested reply · DRAFT — review before using")
        st.text(details.suggested_reply)
        st.caption(f"Provider: OpenAI · Model: {result.generated.receipt.model}")


def render_single_ticket(category_loader: Callable, sentiment_loader: Callable) -> None:
    config = config_module.load_config()
    st.subheader("Complete analysis of one ticket")
    st.caption(
        "This separate single-ticket action uses the local models, then sends a qualifying message and its "
        "category/sentiment to OpenAI for one structured request. Replies are drafts. Batch analysis and storage are later stages."
    )
    st.button("Refresh API setup", key="refresh_api_setup")
    settings = None
    try:
        settings = generation_module.load_generation_settings(config.project_root)
    except generation_module.GenerationError as error:
        st.info(str(error))
    models_ready = all(inspect_model_files(path).state == "Detected (unverified)"
                       for path in (config.classifier_dir, config.sentiment_dir))
    if not models_ready:
        st.caption("Complete analysis also needs the saved category checkpoint and downloaded sentiment model; see setup commands below.")
    with st.form("single_ticket_form"):
        ticket_id = st.text_input("Ticket ID", value="DEMO-001", key="single_ticket_id")
        message = st.text_area("Customer message", key="single_message", max_chars=12000)
        reprocess = st.checkbox("Reprocess explicitly (may make another paid API request)", key="single_reprocess")
        submitted = st.form_submit_button("Analyze one ticket", key="complete_ticket", disabled=settings is None or not models_ready)
    safe_config = settings.model_dump(mode="json") if settings else None
    try:
        env_stat = (config.project_root / ".env").stat()
        env_stamp = (env_stat.st_mtime_ns, env_stat.st_size)
    except OSError:
        env_stamp = None
    public_key = json.dumps({
        "api": safe_config, "env_stamp": env_stamp, "prompt_version": generation_module.PROMPT_VERSION,
        "category": checkpoint_signature(config.classifier_dir),
        "sentiment": checkpoint_signature(config.sentiment_dir),
    }, sort_keys=True)
    config_hash = hashlib.sha256(public_key.encode()).hexdigest()
    if st.session_state.get("single_config_hash") != config_hash:
        for name in ("single_ticket_result", "single_request_hash", "single_input_error"):
            st.session_state.pop(name, None)
        st.session_state["single_ticket_cache"] = {}
        st.session_state["single_config_hash"] = config_hash
    cache = st.session_state.setdefault("single_ticket_cache", {})
    if submitted:
        st.session_state.pop("single_input_error", None)
        try:
            ticket = TicketInput(ticket_id=ticket_id, message=message)
        except ValidationError:
            st.session_state.pop("single_ticket_result", None)
            st.session_state["single_input_error"] = "Input error: provide a nonblank ticket ID and a text message."
        else:
            request_hash = hashlib.sha256((config_hash + ticket.model_dump_json()).encode()).hexdigest()
            if request_hash in cache and not reprocess:
                st.session_state["single_ticket_result"] = cache[request_hash]
                st.session_state["single_request_hash"] = request_hash
                st.info("Showing the saved analysis. Enable explicit reprocessing to retry or make another API request.")
            else:
                # Guard against an accidental queued duplicate while a request is running.
                if st.session_state.get("single_in_progress", False):
                    st.warning("A single-ticket analysis is already running.")
                    return
                st.session_state["single_in_progress"] = True
                try:
                    with st.spinner("Analyzing this ticket…"):
                        # Load only on the explicit action. CPU keeps both models conservative.
                        class LazyModel:
                            def __init__(self, loader, signature):
                                self.loader, self.signature = loader, signature

                            def predict(self, text):
                                return self.loader(self.signature, "cpu").predict(text)

                        category = LazyModel(category_loader, checkpoint_signature(config.classifier_dir))
                        sentiment = LazyModel(sentiment_loader, checkpoint_signature(config.sentiment_dir))
                        result = analyze_ticket(ticket, category, sentiment, lambda: generation_module.create_provider(config.project_root))
                        cache[request_hash] = result
                        st.session_state["single_ticket_result"] = result
                        st.session_state["single_request_hash"] = request_hash
                except Exception:
                    st.session_state.pop("single_ticket_result", None)
                    st.session_state["single_input_error"] = "Local model setup failed. Check the saved checkpoints; no complete analysis or API fallback was created."
                finally:
                    st.session_state["single_in_progress"] = False
    if st.session_state.get("single_input_error"):
        st.error(st.session_state["single_input_error"])
    if "single_ticket_result" in st.session_state:
        render_ticket_result(st.session_state["single_ticket_result"])
    if st.button("Show saved live integration check", key="show_live_check"):
        try:
            record = json.loads((config.project_root / "reports/single_ticket/live_check.json").read_text(encoding="utf-8"))
            if record.get("verification_kind") != "real_local_models_and_openai" or record.get("live_api_verified") is not True:
                raise ValueError
            result = TicketAnalysis.model_validate(record["result"])
            if result.status != "processed":
                raise ValueError
        except (OSError, ValueError, KeyError, TypeError):
            st.info("No successful saved live API check is available yet. Mocked unit tests are separate from live verification.")
        else:
            st.caption("Recorded real API output from the local verification report; displaying it makes no API call.")
            render_ticket_result(result)
