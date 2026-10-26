# Feedback Triage AI

A local customer-feedback and support-ticket analyzer being built as an internship proof of concept.

**Current status: Stage 7 single-ticket analysis passed a real OpenAI integration check.** A separate explicit Streamlit action connects local category/sentiment to validated priority, nullable product/feature and an unsent reply draft. Actual sample T001 returned **Bug Report / Negative / Urgent / Checkout**; its real draft was displayed through Streamlit AppTest without another request. CSV category/sentiment analysis remains partial; complete batches, SQLite and exports are later stages. Category validation scored **0.9305 macro-F1 / 93.33% accuracy** on synthetic, assistant-reviewed data. Independent reviewed sentiment metrics await your examples. Held-out intent-test content stayed unopened during Stages 4–7.

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

Click **Load sample CSV**, or upload a UTF-8 CSV with exactly `ticket_id,message` and click **Validate CSV**. The sample deliberately includes invalid records: expect **6 candidates, 2 ignored, and 4 input errors**. Then click **Analyze categories**: the real saved model produced **5 partial results, 3 ignored, 4 errors and 0 fully processed** on CPU and CUDA. Original text and truncation flags remain visible. Low confidence alone never means spam. The [category inference guide](docs/intent-inference.md) covers CPU fallback, caching, missing-model guidance and actual results.

Then click **Analyze sentiment** (CPU by default). Ignored/invalid rows receive no sentiment. Local models and session results are cached; ordinary reruns do not repeat inference. Bulk **Analyze tickets** remains disabled until batch orchestration is implemented.

For one complete ticket, put your OpenAI key in ignored `.env`, click **Refresh API setup**, then use the separate single-ticket form. The model is configurable (`gpt-5.4-mini` by default). No reply is sent. **Show saved live integration check** displays the recorded real draft without another request. See the [Stage 7 guide](docs/generation.md) for validation, retry limits and the explicit `python -m scripts.check_ticket --live` check. Without `--live`, that command checks configuration without requesting the API.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m pytest -q
```

The full suite passes **318 tests**, including explicitly mocked local/API failures, structured-output/refusal handling and UI cache/review boundaries. Separate real category training/reload, CPU/CUDA category inference, CPU sentiment development inference and actual-model Streamlit AppTest checks passed. Stage 7 also passed a real single-ticket API request and saved-reply Streamlit AppTest, separately from mocked tests. Reviewed sentiment accuracy, held-out intent accuracy, recent browser visual checks and a second fresh-environment installation remain unverified. See the stage guides for measured evidence.

## Prepare category data

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data --audit-only
# After your review/corrections and explicit conflict decisions:
.\.venv\Scripts\python.exe -m scripts.prepare_data
```

Follow the [data review guide](docs/data-review.md). Raw drafts stay unchanged; the command does not approve labels itself. For the current local corpus, the user explicitly delegated review to the assistant: 24 labels were corrected and 67 flagged test candidates were independently reauthored and checked. Seed 42 keeps whole development groups together and all 90 test-pool IDs held out. Actual hashes, IDs, decisions, review origin, and class-count changes are recorded. Data/review artifacts remain local; this preparation command does not train.

## Train the classifier

```powershell
.\.venv\Scripts\python.exe -m scripts.train_intent --smoke
.\.venv\Scripts\python.exe -m scripts.train_intent
.\.venv\Scripts\python.exe -m scripts.train_intent --verify-only
```

The [training guide](docs/training.md) explains prerequisites, pinned reproduction, memory/CPU options, concepts and artifact transfer. Training refuses to overwrite existing run directories. The best model/tokenizer/labels are in `models/ticket_classifier`; actual configuration, duration, history, validation metrics, curves and reload evidence are in `reports/intent_training`. Weights and caches are excluded from Git. Validation scores on this small synthetic corpus do not establish real-customer accuracy.

## Sentiment setup and evaluation

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment --download-model
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment --development --output-dir reports/sentiment_development
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment
# After you supply and check your independently written labels:
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment --reviewed
```

Fill the header-only `data/sentiment_eval.csv` with `message,expected_sentiment`, about 30 independent messages/~10 per sentiment. No human labels are supplied for you. The [sentiment guide](docs/sentiment.md) explains review confirmation, source-row validation and saved metrics/errors. Pending data produces no accuracy claims. Actual assessment checks returned T001 Negative, T002 Positive; ignored T003 was skipped. The Twitter-domain model may behave differently on other support tickets, and no sample-specific label rules are used.

## Architecture

Reusable Python code lives in `src/feedback_analyzer/`, workflows in `scripts/`, and the local interface in `app.py`. Application CSV ingestion is separate from labeled-data preparation. Training uses only message text and category targets. Shared category analysis serves the UI and Python callers. The model is resource-cached, and previews/results live in session state; ordinary reruns do not repeat inference. Labels come from saved configuration, and loading never downloads a fallback model.

The local pipeline uses one fine-tuned `distilbert/distilbert-base-uncased` category classifier and the existing `cardiffnlp/twitter-roberta-base-sentiment-latest` sentiment model. Only messages and targets train the classifier; metadata stays outside features. The selected OpenAI adapter generates priority, a grounded product/feature phrase and a reply draft in one structured operation per qualifying ticket; local category/sentiment remain authoritative. Shared Pydantic schemas validate complete/ignored/error outcomes. SQLite persistence and batch CSV/JSON exports remain later work.

Source, tests, dependency files, and setup documentation are included. Credentials, local environments, caches, downloaded model weights, and machine-specific diagnostic reports stay outside Git. Actual sample SQLite/CSV outputs will be added after the analysis pipeline works.
