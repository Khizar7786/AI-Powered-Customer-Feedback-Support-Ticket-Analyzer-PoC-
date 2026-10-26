# Feedback Triage AI

A local customer-feedback and support-ticket analyzer being built as an internship proof of concept.

**Current status: Stage 9 implements the complete review/export interface.** Upload and validate tickets, explicitly analyze with progress, review Processed/Ignored/Errors, filter category/priority/sentiment and inspect original messages/unsent drafts. Four downloads and shared CLI/UI file exports are available. SQLite reuse and explicit reprocessing/retry remain in place. Category validation scored **0.9305 macro-F1 / 93.33% accuracy** on synthetic, assistant-reviewed data; independently reviewed sentiment metrics await your examples. The automated full-flow check passed; the manual browser check is blocked by Computer Use URL detection. Held-out intent-test content remains unopened.

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

Click **Analyze tickets** for complete batch analysis and SQLite persistence, then review outcome tabs, filters, details and downloads. Ignored/invalid rows receive no downstream sentiment/API call. Filters/downloads/widget changes use saved session results without repeating analysis or inserts. Optional local category/sentiment checks are in the advanced expander. [Interface/export guide](docs/interface.md)

For one complete ticket, put your OpenAI key in ignored `.env`, click **Refresh API setup**, then use the separate single-ticket form. The model is configurable (`gpt-5.4-mini` by default). No reply is sent. **Show saved live integration check** displays the recorded real draft without another request. See the [Stage 7 guide](docs/generation.md) for validation, retry limits and the explicit `python -m scripts.check_ticket --live` check. Without `--live`, that command checks configuration without requesting the API.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m pytest -q
```

The full suite passed **379 tests**. The real sample batch saved **5 processed / 3 ignored / 4 input errors** in [SQLite](outputs/tickets.db). Stage 9 CLI/UI checks reused all 12 actual results, verified all four export payloads/files and made zero additional API calls; filters/reruns made no duplicate inserts. [Verification evidence and limits](docs/interface.md). Reviewed sentiment accuracy, held-out intent accuracy, manual browser downloads and a second fresh-environment installation remain unverified.

## Analyze and save tickets

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze --input data/sample_tickets.csv
# Only when deliberately accepting changed text/configuration for known IDs:
.\.venv\Scripts\python.exe -m scripts.analyze --input data/sample_tickets.csv --reprocess
# Only when deliberately retrying matching previous failures:
.\.venv\Scripts\python.exe -m scripts.analyze --input data/sample_tickets.csv --retry-errors
```

Outcomes go to `outputs/tickets.db`; `--database PATH` overrides it in the CLI. Both interfaces save [results.csv](outputs/results.csv), [ignored.csv](outputs/ignored.csv), [errors.json](outputs/errors.json) and [batch.json](outputs/batch.json) beside the database. CSV protects formula-like text; JSON preserves originals. The sample intentionally has four invalid records, so exit 1 is expected. The [batch guide](docs/batch.md) explains identities, accounting and recovery.

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

The local pipeline uses one fine-tuned `distilbert/distilbert-base-uncased` category classifier and the existing `cardiffnlp/twitter-roberta-base-sentiment-latest` sentiment model. Only messages and targets train the classifier; metadata stays outside features. OpenAI generates priority, a grounded product/feature phrase and a reply draft in one structured operation per qualifying ticket; local category/sentiment remain authoritative. Pydantic validates outcomes. SQLite uses unique identities, short transactions and persistent work claims. CLI and Streamlit share both analysis and exports; downloads read prepared session bytes.

Source, tests, dependency files, setup guides and actual sample SQLite/CSV/JSON output are available. Credentials, local environments, caches, downloaded model weights and machine-specific diagnostic reports stay outside Git.
