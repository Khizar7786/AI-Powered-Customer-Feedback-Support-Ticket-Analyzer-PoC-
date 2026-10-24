# Feedback Triage AI

A local customer-feedback and support-ticket analyzer being built as an internship proof of concept.

**Current status: Stage 2 implemented.** Upload a tickets CSV or load the included sample, validate it, and inspect candidates, ignored messages, and input errors by source line. Original message text is preserved. Model training/inference, API integration, SQLite storage, and exports are pending; AI analysis stays disabled. The included CSV is an application fixture, not a labeled training dataset or prediction output.

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

Click **Load sample CSV**, or upload a UTF-8 CSV with exactly `ticket_id,message` and click **Validate CSV**. The sample deliberately includes invalid records: expect **6 candidates, 2 ignored, and 4 input errors**. Fix reported rows in the source file and upload again. Blank messages and exact `hi`/`hello`/`hey` greetings are ignored; longer greetings, short complaints, and messages containing URLs remain candidates for later classification. A candidate is not a processed result.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover CSV validation and qualification boundaries plus setup and simulated Streamlit sessions, using isolated configuration and mocked devices. Stage 1 also verified tiny GPU/CPU operations and browser startup/reload. These checks do not verify model predictions, API access, training speed, or accuracy. A second fresh-environment installation has not been verified.

## Architecture

Reusable Python code lives in `src/feedback_analyzer/`, executable workflows in `scripts/`, and the local interface in `app.py`. The current implementation covers configuration, setup inspection, shared CSV ingestion, and conservative filtering. Validated previews live in session state; ordinary reruns do not revalidate the file.

The planned analysis pipeline uses one fine-tuned local `distilbert/distilbert-base-uncased` classifier for Bug Report, Feature Request, Billing Issue, General Inquiry, Customer Feedback, and Non-Qualifying. Local sentiment inference will reuse `cardiffnlp/twitter-roberta-base-sentiment-latest` without initial training. One configurable GPT/OpenAI or Gemini API provider will generate priority, a grounded product/feature name, and a short reply draft. Pydantic validation, SQLite persistence, and CSV/JSON exports will connect to a shared CLI/Streamlit pipeline in later stages.

Source, tests, dependency files, and setup documentation are included. Credentials, local environments, caches, downloaded model weights, and machine-specific diagnostic reports stay outside Git. Actual sample SQLite/CSV outputs will be added after the analysis pipeline works.
