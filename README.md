# Feedback Triage AI

Local Streamlit ticket analyzer: read `ticket_id,message` CSV, preserve original text, and save processed, ignored or error outcomes. Replies are drafts.

## Setup (Windows PowerShell)

Use Python 3.12 from the project root. The tested Windows/CUDA versions are pinned in `requirements-windows-cu128.txt`; [CPU setup](docs/setup.md) is also available.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-cu128.txt
.\.venv\Scripts\python.exe -m pip install -c requirements-windows-cu128.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
```

Tested: Python 3.12.5, PyTorch 2.11.0+cu128, Transformers 4.57.6, Streamlit 1.66.1, Pydantic 2.14.1, OpenAI SDK 2.54.0, pytest 9.1.1. Copy `.env.example` to ignored `.env`; set `LLM_PROVIDER=openai`, `LLM_MODEL` (tested: `gpt-5.4-mini`) and your own `OPENAI_API_KEY`.

## Model, dataset, and commands

The public [fine-tuned classifier](https://huggingface.co/khizar7786/feedback-triage-distilbert) has the exact weights and six labels used for the reported category result. The public [reviewed synthetic dataset](https://huggingface.co/datasets/khizar7786/feedback-triage-tickets) contains the same frozen 420/90/90 splits and manifest included here. Download the pinned classifier and the existing Cardiff sentiment model before complete analysis:

```powershell
.\.venv\Scripts\hf.exe download khizar7786/feedback-triage-distilbert --revision f1f2494c16064c3fcd02cca9c1c13a78c1cd8283 --local-dir models/ticket_classifier
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment --download-model
.\.venv\Scripts\python.exe -m scripts.analyze --input data/sample_tickets.csv
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest
```

You can instead [train](docs/training.md) a new classifier from the included splits; its results may differ. Rebuilding splits requires the separately transferred raw/review working copy. The saved one-pass [category evaluation](docs/evaluation.md) belongs to the pinned original checkpoint. Future tuning needs a new independent test set.

Open `http://127.0.0.1:8501`, load/upload a CSV, validate, then click **Analyze tickets**. Preview needs no models or key; complete analysis needs both local models and OpenAI. The 12-row sample has four deliberate input errors, so CLI exit code 1 can accompany saved outcomes. Use `--reprocess` for changed inputs/configuration and `--retry-errors` for explicit failed-row retries.

## Design, evidence, and limits

`src/feedback_analyzer/` validates input, filters blank/exact greetings, uses fine-tuned six-class DistilBERT and pretrained Cardiff sentiment locally, then asks OpenAI once per qualifying ticket for priority, grounded feature and reply. Pydantic validates; SQLite and CSV/JSON save outcomes. `scripts/` and `app.py` share this pipeline. UI caching prevents ordinary reruns from repeating paid calls.

The 600 English messages are synthetic and assistant-reviewed under user delegation, not independently human validated. The original checkpoint scored **89/90** on the held-out category test (98.89% accuracy; macro-F1 0.9889), with **0/75 valid tickets incorrectly ignored**. A prior live run matched all three assessment examples; the latest 12-row sample saved **5 processed / 3 ignored / 4 input errors**, but gave the dark-mode request **Low** rather than expected **Medium** priority. The mixed batch had four priority disagreements and reply cautions. [Requirement check](docs/submission-checklist.md), [evaluation](docs/evaluation.md), [interview notes](docs/interview-notes.md). Independent sentiment accuracy, physical browser downloads, a clean-machine install and LLM-only cost/speed/privacy/accuracy comparisons remain unmeasured.

Actual outputs: [SQLite](outputs/tickets.db), [processed CSV](outputs/results.csv), [ignored CSV](outputs/ignored.csv), [errors](outputs/errors.json), [full batch](outputs/batch.json), plus [three-example acceptance results](reports/acceptance/assessment/results.csv). Keys, environments, caches and weights stay out of Git.
