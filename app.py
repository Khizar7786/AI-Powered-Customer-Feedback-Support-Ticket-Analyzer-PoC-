"""Stage 1: a local Streamlit setup page, with no analysis side effects."""

import streamlit as st

from feedback_analyzer.config import CLASSIFIER_MODEL_ID, SENTIMENT_MODEL_ID, load_config
from feedback_analyzer.readiness import inspect_api_config, inspect_device, inspect_model_files


@st.cache_resource
def device_status():
    """Inspect the device once per app process; no model is loaded."""
    return inspect_device()


def main() -> None:
    st.set_page_config(page_title="Feedback Triage AI", page_icon="💬")
    st.title("Feedback Triage AI")
    st.caption("Customer feedback and support-ticket analysis · Stage 1")
    st.info("The application shell is running. Ticket analysis becomes available in later stages.")

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

    st.button("Analyze tickets", disabled=True)
    st.caption("Upload and CSV validation are the next stage. No models are loaded on this page.")


if __name__ == "__main__":
    main()
