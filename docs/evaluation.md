# Final intent evaluation and acceptance (Stage 10)

The selected epoch-3 DistilBERT checkpoint was evaluated **once on 90 frozen test rows** after model/prompt choices were locked. It correctly classified **89/90: accuracy 98.89%, macro-F1 0.9889**. All rows produced valid predictions. This is category performance on a small synthetic, assistant-reviewed English corpus; it is not measured real-customer or complete-pipeline accuracy.

## Fixed choices and one-pass protection

```powershell
# Record choices without decoding held-out messages or predicting:
.\.venv\Scripts\python.exe -m scripts.evaluate_intent --freeze-only
# Perform the first held-out pass, or return its saved result thereafter:
.\.venv\Scripts\python.exe -m scripts.evaluate_intent
```

The command verifies frozen train/validation/test byte hashes, disjoint ID/scenario memberships, recorded review statuses, class counts and checkpoint/training-data identity. Only message text reaches the classifier; IDs/labels/metadata never enter its tokenizer input. Test messages are not sent to any generative API. CPU is the default; `--device auto` is available.

An exclusive registry reservation keyed by test-file SHA-256 is created before decoding/prediction. Predictions are journaled as they complete. A repeated command returns the recorded report with **zero new predictions/model loads**; choosing another output folder cannot bypass that reservation. A changed model/prompt choice is refused against a consumed set. An interrupted run retains its journal and refuses an automatic second pass. Inspect the partial record instead of deleting the reservation to chase a score.

The exact model/tokenizer/label/config fingerprints, public provider settings, prompt/schema/source fingerprints and selected validation checkpoint are in [frozen_choices.json](../reports/intent_evaluation/frozen_choices.json). Model weights SHA-256 is `993d236aa4473be875c004a9a1c8256271286349b7c929eb67bbb7aa5c087b06`. The reviewed dataset fingerprint is `c8f76e80e95337bdfd874780bcad16483a2db02a9539f05ab44794b8a2ae15bf`; test CSV SHA-256 is `c82f49f18e7e05f3324ab400b5b45b8230289def09d42c46560d4a428b27ad06`.

The frozen manifest's historical `evaluation_run=false` and training record's test-unopened flags are deliberately preserved rather than changing their hashes. The external registry and final evaluation record now document that the set has been evaluated. No model/prompt tuning, training, threshold adjustment, relabeling or split change followed the result. If future tuning uses these results, call this set development data and create a new independent held-out set.

## Actual category results

All six classes have 15 test rows. Five classes had perfect recall; Feature Request recall was 14/15. The single mismatch, **TEST0042**, was Feature Request → General Inquiry. It was retained as a qualifying ticket, not ignored. No test example was rewritten or used to select another model.

- **Valid tickets incorrectly ignored:** 0/75.
- **Non-Qualifying incorrectly accepted:** 0/15.
- **Inference failures:** 0/90; prediction coverage 100%.
- **Measured inference/metric phase:** 12.58 seconds; excludes model loading and report/plot writing. This one run is not a speed guarantee.

[Main metrics/identity](../reports/intent_metrics.json), [per-class report](../reports/intent_evaluation/report.md), [confusion matrix](../reports/intent_evaluation/confusion_matrix.png), [predictions](../reports/intent_evaluation/predictions.json), [errors](../reports/intent_evaluation/errors.json), [qualification errors](../reports/intent_evaluation/qualification_errors.json), [repeat-command guard check](../reports/intent_evaluation/reuse_check.json).

Accuracy is the proportion of correct labels. Precision asks how many predictions of a category were correct; recall asks how many examples carrying that label were found. Macro-F1 averages each category's F1 equally. The matrix uses true categories as rows and predicted categories as columns. Failed predictions, if any, count as wrong/false negatives and get an explicit error column rather than disappearing from metrics. [Classification report reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html), [matrix convention](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).

The initial evaluation command encountered a stray Unicode character in `intent.py` during import. It made **no test predictions or reservation**. Removing exactly that character restored bytes matching the already frozen source SHA-256; the subsequent pass produced the only 90 predictions. [Recovery record](../reports/intent_evaluation/startup_recovery.json).

## Real complete-pipeline acceptance

```powershell
.\.venv\Scripts\python.exe -m scripts.check_acceptance
# Only for an explicit retry of previous failures; successful results stay reused:
.\.venv\Scripts\python.exe -m scripts.check_acceptance --retry-errors
```

The [visible fixtures](../data/acceptance/README.md) are separate from the held-out split. They use real saved CPU category/sentiment models and the configured OpenAI adapter. Each batch saves actual SQLite, processed/ignored CSV, error/full JSON, checksum manifest and per-run checks under `reports/acceptance/`. Initial failures and later retries are archived separately, never replaced by expected values.

After one explicit retry of the two failed tickets:

| Batch | Actual outcomes | Field/check result |
|---|---|---|
| Assessment, 3 records | 2 processed / 1 ignored / 0 errors | T001 Bug Report / Urgent / Negative; T002 Feature Request / Medium / Positive; T003 ignored |
| Mixed, 14 records | 7 processed / 2 ignored / 5 deliberate errors | Categories/sentiment matched the prewritten references; four priority disagreements remain |

The mixed errors are missing ID, both duplicate-ID occurrences, malformed fields and the overlong API-input case. The latter retains both local predictions, with independent 128/512-token truncation flags and the complete original message. The promotion/exact greeting are ignored. Greeting-prefixed issues, the short complaint and URL/untrusted-instruction issue remain valid. The final untrusted-instruction reply did not claim an account fix.

The first live run also produced a real schema-validation error on T001 and a real timeout on the URL case. Both stayed errors with useful local partials and later rows continued. The explicit two-ticket retry recovered them. Total observed adapter operations/known HTTP attempts: **11**, of which **9 returned validated results**. Usage/billing of failed or timed-out requests is unknown and is not fabricated. No LLM-only comparison, extra LLM judge or additional model training was performed.

The fixed mixed-priority references disagreed on **ACC-M001 (Urgent vs High), ACC-M002 (Medium vs Low), ACC-M003 (Medium vs High), and ACC-M008 (Medium vs High)**. These were not overridden. The duplicate-charge priority is particularly judgement-dependent; all mixed expectations are assistant-defined acceptance references, not independently established truth.

All final processed features were grounded in their messages and all drafts passed structural/policy checks, but that does not prove semantic quality. Assistant reading found redundant questions on T001/ACC-M001, an unspecified secure-channel concern for ACC-M003's request for partial payment details, and an unverified help/settings location in ACC-M004. The drafts require human review. See [actual checks](../reports/acceptance/summary.json), [qualitative review](../reports/acceptance/review.md) and [review data/history](../reports/acceptance/qualitative_review.json).

## General fixes and limits

Before held-out messages were opened, a pre-existing development reply promising to “pass it along” exposed an incomplete guard. General first-person future commitments are now rejected, the policy is versioned **ticket-details-v2**, and the prompt reinforces the existing no-action-promise rule. Batch UI caches now invalidate on prompt-version changes. Regression tests passed **404 cases in 106.74 seconds** before the real runs. The fix was not derived from held-out predictions and has no ticket-specific branches.

Older saved application inputs under v1 require explicit changed-configuration reprocessing to use v2; the app does not silently pay to regenerate them. Stage 10 reports use isolated databases and do not overwrite the prior sample database.

Final verification rechecked all four archived acceptance exports against their checksum manifests and SQLite run counts; both databases had integrity `ok`, no foreign-key failures, duplicate outcome identities or unresolved claims. Frozen choices still match, with one completed test reservation and exactly 90 journal entries. Dependency and whitespace checks passed, local documentation links resolve, and a scan of 211 deliverable text files found no configured keys or key-like secrets. [Verification record](../reports/intent_evaluation/verification.json).

A fresh Streamlit AppTest loaded the actual sample and reran successfully with guards proving zero model loads, API calls or batch inserts; existing output bytes stayed unchanged. The local server was restarted with the project virtual environment and returned HTTP 200 from its health endpoint. These checks supplement the mocked full UI integration tests; physical browser upload/download remains unverified. [Startup/preview check](../reports/intent_evaluation/ui_startup_check.json).

The 600-message corpus is synthetic, reviewed by the assistant under the user's explicit delegation, not independently human-validated. Test scenarios were held out from training/checkpoint selection after freezing, but common authorship/template/domain patterns can inflate scores. The test is now consumed for final reporting, not an untouched future tuning resource. Independent sentiment accuracy still needs the user's separately reviewed labels; acceptance references do not replace that dataset. Physical browser upload/download verification remains blocked by Computer Use URL detection, and no second clean installation is claimed. Complete-pipeline accuracy, cost savings or speed advantages over an LLM-only pipeline were not measured.
