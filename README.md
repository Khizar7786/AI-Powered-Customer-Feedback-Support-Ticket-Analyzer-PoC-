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

## Models and commands

The reviewed synthetic 420/90/90 splits are included; weights are excluded. Transfer the original `models/ticket_classifier/` directory separately, or train a new checkpoint from the included splits with the pinned revision below. New scores may differ. Download Cardiff sentiment weights with `python -m scripts.evaluate_sentiment --download-model`. [Training](docs/training.md), [sentiment setup](docs/sentiment.md). Data preparation needs the separately transferred raw/review working copy. Future tuning needs a new independent test set.

Use `.\.venv\Scripts\python.exe` in place of `python` below:

```powershell
python -m scripts.prepare_data --audit-only  # requires the separately transferred raw draft
python -m scripts.train_intent --smoke --revision 12040accade4e8a0f71eabdb258fecc2e7e948be
python -m scripts.train_intent --revision 12040accade4e8a0f71eabdb258fecc2e7e948be --report-dir reports/intent_training_reproduction
python -m scripts.evaluate_intent  # requires the exact original checkpoint for saved one-pass result
python -m scripts.evaluate_sentiment
python -m scripts.analyze --input data/sample_tickets.csv
python -m streamlit run app.py
python -m pytest
```

Open `http://127.0.0.1:8501`, load/upload a CSV, validate, then click **Analyze tickets**. Preview needs no models or key; complete analysis needs both local models and OpenAI. The 12-row sample has four deliberate input errors, so CLI exit code 1 can accompany saved outcomes. Use `--reprocess` for changed inputs/configuration and `--retry-errors` for explicit failed-row retries.

## Design, evidence, and limits

`src/feedback_analyzer/` validates input, filters blank/exact greetings, uses fine-tuned six-class DistilBERT and pretrained Cardiff sentiment locally, then asks OpenAI once per qualifying ticket for priority, grounded feature and reply. Pydantic validates; SQLite and CSV/JSON save outcomes. `scripts/` and `app.py` share this pipeline. UI caching prevents ordinary reruns from repeating paid calls.

The 600 English messages are synthetic and assistant-reviewed under user delegation, not independently human validated. The original checkpoint scored **89/90** on the held-out category test (98.89% accuracy; macro-F1 0.9889), with **0/75 valid tickets incorrectly ignored**. A prior live run matched all three assessment examples; the latest 12-row sample saved **5 processed / 3 ignored / 4 input errors**, but gave the dark-mode request **Low** rather than expected **Medium** priority. The mixed batch had four priority disagreements and reply cautions. [Requirement check](docs/submission-checklist.md), [evaluation](docs/evaluation.md), [interview notes](docs/interview-notes.md). Independent sentiment accuracy, physical browser downloads, a clean-machine install and LLM-only cost/speed/privacy/accuracy comparisons remain unmeasured.

Actual outputs: [SQLite](outputs/tickets.db), [processed CSV](outputs/results.csv), [ignored CSV](outputs/ignored.csv), [errors](outputs/errors.json), [full batch](outputs/batch.json), plus [three-example acceptance results](reports/acceptance/assessment/results.csv). Keys, environments, caches and weights stay out of Git.
