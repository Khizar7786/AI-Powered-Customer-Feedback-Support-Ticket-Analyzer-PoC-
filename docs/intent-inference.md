# Local category analysis (Stage 5)

Start Streamlit from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Click **Load sample CSV**, or upload `ticket_id,message` and click **Validate CSV**. Then click **Analyze categories**. Input errors and conservative blank/exact-greeting ignores bypass the model. File-level errors block analysis until corrected. Missing sentiment weights or API credentials do not prevent category-only analysis.

The result tabs show **Partial analysis**, **Ignored after category checks**, and **Category errors**. Only a predicted `Non-Qualifying` category adds a semantic ignore reason. A low score alone never means spam. Error counts include input-validation and inference failures; each result retains its source lines and original text. Category-only results always have **Fully processed = 0**. Stage 6 can add [sentiment separately](sentiment.md); these local actions still lack priority, product/feature and reply. Use [Stage 8 Analyze tickets](batch.md) for complete persisted batches or the separate [single-ticket form](generation.md) when API configuration is ready.

Original messages stay unchanged. The classifier receives the whitespace-normalized input displayed separately. The result table includes confidence, full/used token counts, the trained input limit, device and **Classifier input truncated?**. Above the saved limit (128 tokens here, including special tokens), only the beginning is classified; a warning says that the remainder was not analyzed. Confidence is a softmax score, not a calibrated probability of correctness.

## Model loading and CPU fallback

`feedback_analyzer.intent.IntentClassifier` loads the local tokenizer, configuration and safetensors weights from `CLASSIFIER_DIR` (default `models/ticket_classifier`). It requires a completed full-training record, a valid saved input limit and a complete matching six-label model. Smoke/base models, generic label mappings, incomplete weights and missing files are rejected. Numeric label order comes from the saved configuration, never a hardcoded index order. Loading is local-only; the app cannot download or train a substitute model.

The **Classifier device** selector provides **Auto** and **CPU**. Auto uses CUDA when available. If CUDA initialization or inference fails, the same classifier moves to CPU and displays a warning. CPU uses the same saved weights. Invalid outputs or a failed CPU retry become row errors, with no invented category. Actual CPU inference passed; CUDA initialization/OOM failures are tested with explicit mocks, not by intentionally exhausting this laptop's VRAM.

The model runs in evaluation mode (training dropout disabled) inside `torch.inference_mode()` (no gradients). A lock protects the cached model across Streamlit sessions. `st.cache_resource` retains one model resource keyed by checkpoint file stamps and device choice; changing those invalidates previous results. Results live in session state, so inspecting rows and ordinary reruns do not invoke inference again. **Reanalyze categories** explicitly repeats inference using the cached model. Changing the input clears old results and requires another deliberate action.

If the checkpoint is missing, the UI shows:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_intent
```

See the [training guide](training.md) for reviewed-data prerequisites and model transfer. To check an existing checkpoint in a separate process:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_intent --verify-only
```

The training command protects nonempty run directories. Restore the complete saved model or choose new training output/report paths instead of overwriting an existing experiment. A custom `CLASSIFIER_DIR` is set through your local environment or ignored `.env`; pass the same directory to training verification using `--output-dir`.

## Reuse outside Streamlit

```python
from pathlib import Path
from feedback_analyzer.intent import IntentClassifier
from feedback_analyzer.ingestion import load_ticket_csv
from feedback_analyzer.category_analysis import analyze_categories

classifier = IntentClassifier.from_checkpoint(Path("models/ticket_classifier"), device="cpu")
preview = load_ticket_csv(Path("data/sample_tickets.csv").read_bytes())
batch = analyze_categories(preview, classifier)
print(batch.counts, batch.fully_processed)
```

`analyze_categories` preserves every validated input record, sends only candidates to `.predict()`, validates predictions and continues after an individual inference failure. It makes no API requests, database inserts, or training calls. The complete CLI/storage/export pipeline remains later-stage work.

## Actual checks — 2026-10-26

Separate real inference on the sample CSV passed with both forced CPU and Auto/CUDA. Both produced **5 partial / 3 ignored / 4 errors / 0 fully processed**. The six candidates were classified as follows (CPU scores shown):

| Ticket | Actual category | Score | Outcome |
|---|---|---:|---|
| T001 | Bug Report | 0.5859 | Partial |
| T002 | Feature Request | 0.5251 | Partial |
| T003 | Non-Qualifying | 0.3293 | Ignored |
| T006 | Bug Report | 0.4294 | Partial |
| T007 | Bug Report | 0.4897 | Partial |
| T008 | Billing Issue | 0.2843 | Partial |

The three published assessment category expectations matched without hardcoding. Priority/sentiment/replies were not inferred. T008's modest score remains a partial prediction, not a spam rejection. A separate long fixture had 457 tokens; exactly 128 were used and truncation was flagged. Full real results, device choices, measured load/inference intervals and model SHA-256 are in [sample_inference.json](../reports/intent_inference/sample_inference.json). These few visible fixtures are integration checks, not a new accuracy estimate.

An actual Streamlit AppTest session used the real CPU checkpoint and the sample: no model load or prediction on startup/validation; one load and six predictions after the explicit action; no additional work on ordinary rerun; twelve predictions total after explicit reanalysis while model loads stayed at one. See [streamlit_real_check.json](../reports/intent_inference/streamlit_real_check.json). This was a simulated Streamlit session; visual browser verification was unavailable because computer-use inventory returned no browser/native connection.

The Stage 5 suite passed **175 tests**, including 27 explicit fake-model inference tests, 9 shared category-workflow tests and 12 mocked UI tests. Mocked tests cover saved label permutation, strict local loading, missing/uninitialized weights, inference/evaluation mode, malformed scores, CPU fallback, truncation, source-text preservation, cache/reprocessing boundaries and per-row failures. No held-out test content was opened/evaluated, no retraining occurred, and the synthetic/assistant-reviewed training-data limitations remain. See the README for later-stage verification.

References: [Transformers local model loading](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model), [PyTorch inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html), and [Streamlit resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource). Actual installed versions are recorded with training; Streamlit for these checks was 1.66.1.
