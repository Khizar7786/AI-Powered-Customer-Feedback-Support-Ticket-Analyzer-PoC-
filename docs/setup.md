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

The current Stage 1 page shows device information and setup status for the category classifier, sentiment model, and generative API. Missing model files and an unconfigured API are expected. Existing files are labeled **Detected (unverified)**; configuration presence is **Configured (unverified)**. Neither status proves working inference or API access. **Analyze tickets** remains disabled throughout this stage.

## Configuration

The current app needs no `.env` file, credentials, or downloaded weights. When later integration requires configuration, copy the template without overwriting an existing file:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

Set `LLM_PROVIDER` to the selected provider (`openai` or `gemini`), `LLM_MODEL` to a verified available model, and the matching `OPENAI_API_KEY` or `GEMINI_API_KEY` locally. Provider selection and live API integration are pending. Environment variables override the project-root `.env`; secret values are not displayed by the setup page. Keep `.env` out of Git.

`CLASSIFIER_DIR` defaults to `models/ticket_classifier` and `SENTIMENT_DIR` to `models/sentiment`. Relative paths resolve from the project root. Model files are excluded from Git; training/model acquisition instructions will be added when those workflows are implemented.

## Verification scope

Stage 1 was checked with Python 3.12.5, PyTorch 2.11.0+cu128, an NVIDIA RTX 3050 Ti Laptop GPU, real float32/float16 GPU operations, and a separate forced-CPU float32 operation. CPU execution was checked using the CUDA wheel; a separate CPU-wheel installation has not been verified. The Streamlit server and browser startup/reload were checked locally.

The 12 automated setup tests cover configuration precedence, placeholder handling, model-file detection, device fallback, secret exclusion, resource caching, and disabled analysis. They use mocked devices/fake artifacts and simulated Streamlit sessions. No model training, ticket inference, generative API calls, or final evaluation has been performed. See the [README](../README.md) for implementation status and planned architecture.
