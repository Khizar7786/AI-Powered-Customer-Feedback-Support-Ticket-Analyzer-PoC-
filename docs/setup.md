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

The **Model and API setup** expander shows device information and setup status for the category classifier, sentiment model, and generative API. Missing model files and an unconfigured API are expected. Existing files are labeled **Detected (unverified)**; configuration presence is **Configured (unverified)**. Neither status proves working inference or API access. **Analyze tickets** remains disabled throughout Stage 2.

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

The 12-record sample includes all three exact assessment messages and deliberate edge cases. Expected validation counts are **6 candidates, 2 ignored, 4 input errors**. This is input-validation evidence, not AI prediction output. No model loading, API requests, database writes, or training take place in this flow.

## Configuration

The current app needs no `.env` file, credentials, or downloaded weights. When later integration requires configuration, copy the template without overwriting an existing file:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

Set `LLM_PROVIDER` to the selected provider (`openai` or `gemini`), `LLM_MODEL` to a verified available model, and the matching `OPENAI_API_KEY` or `GEMINI_API_KEY` locally. Provider selection and live API integration are pending. Environment variables override the project-root `.env`; secret values are not displayed by the setup page. Keep `.env` out of Git.

`CLASSIFIER_DIR` defaults to `models/ticket_classifier` and `SENTIMENT_DIR` to `models/sentiment`. Relative paths resolve from the project root. Model files are excluded from Git; training/model acquisition instructions will be added when those workflows are implemented.

## Verification scope

Stage 1 was checked with Python 3.12.5, PyTorch 2.11.0+cu128, an NVIDIA RTX 3050 Ti Laptop GPU, real float32/float16 GPU operations, and a separate forced-CPU float32 operation. CPU execution was checked using the CUDA wheel; a separate CPU-wheel installation has not been verified. The Streamlit server and browser startup/reload were checked locally.

Automated tests cover configuration precedence, placeholder handling, model-file detection, device fallback, CSV schema/encoding/record errors, duplicate IDs, physical line attribution, filtering boundaries, preview persistence, secret exclusion, resource caching, and disabled analysis. UI tests use mocked devices/fake artifacts and simulated Streamlit sessions. No model training, ticket inference, generative API calls, or final evaluation has been performed. See the [README](../README.md) for implementation status and planned architecture.
