# Local setup

Run commands in PowerShell from the project root. Python 3.12 is required by `pyproject.toml`. A virtual environment isolates project dependencies; the editable installation makes source changes available to both the app and scripts.

## Install

Create the environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

For a supported NVIDIA GPU, install the CUDA 12.8 PyTorch build used during development:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU execution, use this command instead of the CUDA command:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the package and test dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
```

The commands use the [official PyTorch wheel indexes](https://pytorch.org/get-started/previous-versions/#v2-11-0). A compatible NVIDIA driver is needed for CUDA execution. Check actual CUDA support with the diagnostic below; driver-supported CUDA and PyTorch's bundled CUDA runtime can have different version numbers.

For the exact dependency versions exported from the tested Windows x64 / Python 3.12 / CUDA 12.8 environment, replace the PyTorch and editable-install commands with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-cu128.txt
.\.venv\Scripts\python.exe -m pip install -c requirements-windows-cu128.txt -e ".[dev]"
```

This export is specific to that CUDA environment; use the CPU path above for a CPU installation. A second clean installation of the export has not been verified.

## Check and start

```powershell
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m scripts.check_environment --device cpu
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. Stop the server with Ctrl+C. If that port is already in use, stop the existing server or add `--server.port 8502` and open the corresponding URL.

The diagnostic checks Python/package versions, CUDA availability, available GPU memory, and tiny tensor calculations. Add `--output reports/environment.json` to save a local JSON report. These reports are machine-specific and excluded from Git. A passing tiny calculation does not prove that model training fits in memory or measure model speed.

The **Model and API setup** expander shows device information and setup status for the category classifier, sentiment model, and generative API. Existing files are labeled **Detected (unverified)** because this lightweight inspection does not load weights; configuration presence is **Configured (unverified)**. Neither status proves working inference or API access. **Analyze categories** uses the trained classifier; **Analyze sentiment** then runs only for qualifying results. Bulk **Analyze tickets** stays disabled pending batch orchestration. The separate Stage 7 form implements one complete ticket when local models and OpenAI configuration are available.

## Upload and preview (Stage 2)

Click **Load sample CSV** to inspect `data/sample_tickets.csv`, or select a file under **Upload tickets CSV** and click **Validate CSV**. Selecting a different file clears the old preview. Validation runs only after a button click, and results remain in the current session across reruns.

Use UTF-8 CSV (a UTF-8 BOM is accepted) with exactly `ticket_id` and `message`. Either column order is accepted; surrounding header and ID whitespace is trimmed. IDs are case-sensitive. The labeled training CSV has a separate schema and is rejected here. Quote any message containing commas, quotes, or newlines, for example:

```csv
ticket_id,message
T1,"Hi, checkout crashes."
T2,App crashes
```

Missing IDs, missing/extra fields, and every occurrence of a duplicated ID appear as **Input errors**. Empty message fields and exact simple greetings (`hi`, `hello`, `hey`, ignoring case and ordinary trailing punctuation) appear as **Ignored**. Other messages appear as **Candidates** without category, sentiment, or processed claims. The longer assessment greeting remains a candidate until semantic classification is implemented.

Tables show the starting and ending physical CSV line, including the header in line numbering. The row inspector displays the original message with whitespace preserved; normalized input is separate. Broken quoting stops parsing because subsequent record boundaries are uncertain. Prior records and the remaining raw text stay inspectable, with a file-level error explaining the stop. Header/encoding errors prevent row parsing. Correct errors in the source file and upload again; the app does not modify files.

The 12-record sample includes all three exact assessment messages and deliberate edge cases. Expected validation counts are **6 candidates, 2 ignored, 4 input errors**. No model loading, API requests, database writes, or training take place during validation. After validation, **Analyze categories** runs the saved model locally and shows partial outcomes; see the [Stage 5 guide](intent-inference.md). Actual sample category results are **5 partial, 3 ignored, 4 errors, 0 fully processed**.

## Configuration

Category-dataset audit/review/splitting uses a separate command and schema; see [data-review.md](data-review.md). `python -m scripts.prepare_data --audit-only` creates the review workflow. The default `python -m scripts.prepare_data` reads `data/reviewed_messages.csv` and refuses to freeze splits while review or conflict resolution is pending. It does not train or alter the Streamlit upload path.

Upload/preview needs no `.env`, credentials or downloaded weights. Category analysis requires the complete trained checkpoint at `CLASSIFIER_DIR`; it needs no API credentials or sentiment model. Missing checkpoints show the training command. To customize local paths or configure later integration, copy the template without overwriting an existing file:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

OpenAI is selected for Stage 7. Set `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-5.4-mini` (configurable), and `OPENAI_API_KEY` locally. The ignored `.env` already exists and contains the locally supplied key in this development workspace; edit it without overwriting other settings. The tracked `.env.example` contains a blank key only. Environment variables override the project-root `.env`; secret values are not displayed by the setup page. Keep `.env` out of Git. Click **Refresh API setup** after changing the key. The [Stage 7 guide](generation.md) documents the single-ticket form, supported structured output, request limits and successful actual development integration check.

`CLASSIFIER_DIR` defaults to `models/ticket_classifier` and `SENTIMENT_DIR` to `models/sentiment`. Relative paths resolve from the project root. Model files are excluded from Git. The [training guide](training.md) covers classifier training and transfer. The [sentiment guide](sentiment.md) covers `python -m scripts.evaluate_sentiment --download-model`, CPU-default inference and your separately reviewed sentiment data. Sentiment weights are reused without training, and no API credentials are needed for either local model.

## Verification scope

Stage 1 was checked with Python 3.12.5, PyTorch 2.11.0+cu128, an NVIDIA RTX 3050 Ti Laptop GPU, real float32/float16 GPU operations, and a separate forced-CPU float32 operation. CPU execution was checked using the CUDA wheel; a separate CPU-wheel installation has not been verified. The Streamlit server and browser startup/reload were checked locally.

Automated tests cover configuration, CSV validation/filtering, data preparation, training, category/sentiment inference, review-gated sentiment evaluation and single-ticket structured generation. The full suite passes 318 tests with explicit mocked UI/device/API failures and cache/reprocessing checks. Separate real training/reload, CPU/CUDA category inference, CPU sentiment development and real-model Streamlit AppTest sessions passed. Stage 7 passed the actual missing-key guard, then a real OpenAI request on T001 and saved-reply Streamlit AppTest with zero additional requests. CUDA failure/OOM paths and API responses are mocked in unit tests; those tests are separate from the live report. Browser visual verification, independent reviewed sentiment metrics, complete batches/storage/exports and held-out intent evaluation remain pending. See the [README](../README.md), [training guide](training.md), [category guide](intent-inference.md), [sentiment guide](sentiment.md) and [Stage 7 guide](generation.md).
