# Feedback Triage AI

A local customer-feedback and support-ticket analyzer being built as an internship proof of concept.

**Current status: Stage 1 implemented.** The Streamlit app starts and displays the execution device, local model-file status, and API configuration status. Ticket analysis is disabled. CSV ingestion, model training/inference, API integration, SQLite storage, and exports are still pending; no datasets, trained models, prediction outputs, or evaluation metrics are included yet.

## Run locally

Use Python 3.12 and PowerShell from the project root. For a supported NVIDIA GPU:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. No API key or model download is needed for the current setup page. [Setup instructions](docs/setup.md) cover the CPU alternative, the tested dependency export, configuration, and environment checks.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m pytest -q
```

Stage 1 verification recorded 12 passing tests, real tiny GPU/CPU tensor operations, and a browser startup/reload. Tests use isolated configuration, fake model files, mocked devices, and Streamlit's simulated app sessions. These checks establish setup behavior; they do not verify model predictions, API access, training speed, or accuracy. A second fresh-environment installation has not been verified.

## Architecture

Reusable Python code lives in `src/feedback_analyzer/`, executable workflows in `scripts/`, and the local interface in `app.py`. The current implementation covers configuration and setup inspection.

The planned analysis pipeline uses one fine-tuned local `distilbert/distilbert-base-uncased` classifier for Bug Report, Feature Request, Billing Issue, General Inquiry, Customer Feedback, and Non-Qualifying. Local sentiment inference will reuse `cardiffnlp/twitter-roberta-base-sentiment-latest` without initial training. One configurable GPT/OpenAI or Gemini API provider will generate priority, a grounded product/feature name, and a short reply draft. Pydantic validation, SQLite persistence, and CSV/JSON exports will connect to a shared CLI/Streamlit pipeline in later stages.

Source, tests, dependency files, and setup documentation are included. Credentials, local environments, caches, downloaded model weights, and machine-specific diagnostic reports stay outside Git. Actual sample SQLite/CSV outputs will be added after the analysis pipeline works.
