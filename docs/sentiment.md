# Local sentiment and separate evaluation (Stage 6)

The app uses the existing `cardiffnlp/twitter-roberta-base-sentiment-latest` weights. No sentiment training takes place. Download them once, separately from ordinary UI reruns:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment --download-model
# Clearly identified development inference and separate assessment checks:
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment --development --output-dir reports/sentiment_development
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The pinned model revision is `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`. `SENTIMENT_DIR` defaults to `models/sentiment`; use local configuration for another directory and pass the same `--model-dir` to the CLI. Setup records the source, revision, weight SHA-256 and actual acquisition duration in `sentiment_source.json`. The official snapshot supplies `pytorch_model.bin`; loading uses `weights_only=True`, local files and the existing trained classification head. Unknown labels or incomplete model weights cause an error, with setup guidance.

## Use it in Streamlit

1. Load/upload and validate `ticket_id,message`.
2. Click **Analyze categories**. Blank/exact-greeting ignores, invalid rows and predicted Non-Qualifying messages receive no sentiment.
3. Click **Analyze sentiment**. Only qualifying category results reach the sentiment model. **CPU** is the default; **Auto** can use CUDA, with visible CPU fallback if CUDA fails.

The category-plus-sentiment batch table remains **partial analysis**. Priority, product/feature and reply are absent from this batch action, so **Fully processed stays 0** and bulk **Analyze tickets** stays disabled. The separate [Stage 7 single-ticket form](generation.md) adds validated API fields when configuration is ready. A sentiment failure is an error and retains the category and original message; it never becomes spam or an invented Neutral label.

Models are cached separately with `st.cache_resource`, batch size one and a lock for concurrent sessions. Inference uses evaluation mode and no gradients. Results remain in session state: ordinary reruns do not repeat predictions, while **Reanalyze sentiment** explicitly repeats them using the cached model. Changing input, category results, model files or device invalidates stale sentiment results. Startup/validation do not load sentiment weights, download a model or analyze messages.

The saved mapping is checked rather than guessing label indices. The output names are **Positive**, **Neutral** and **Negative**. As recommended by the model card, usernames and URL tokens become `@user` and `http` in a separate model-input string; original text stays unchanged. The table shows that input, full/used token counts and a separate sentiment truncation flag. This model uses up to **512 tokens**, including special tokens; longer input is cut at the end. Category input has its own 128-token limit and flag, so the two may differ.

## Supply your reviewed evaluation data

[`data/sentiment_eval.csv`](../data/sentiment_eval.csv) is deliberately a header-only template:

```csv
message,expected_sentiment
```

Write about **30 independent English customer messages**, around **10 per sentiment**, and label/check each yourself. The target counts are reported rather than fabricated or forced. Do not copy the assessment fixtures, assistant development examples, or category-training messages into this evaluation. The application CSV and category CSV are separate contracts.

- **Positive:** clear praise, satisfaction or favorable feeling.
- **Neutral:** a factual question or statement without clear favorable/unfavorable feeling.
- **Negative:** dissatisfaction, frustration or a negative opinion.

Judge tone and meaning, not category alone: a feature request can be neutral, and a billing question need not be negative. Include realistic wording and record a consistent decision for mixed feelings. Use the exact label spelling/capitalization above. Save UTF-8 CSV and quote messages containing commas, quotes or newlines. Preserve your original messages. No labels have been generated or inserted in this template.

Audit first; this command does not load the model while labels/review are pending:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment
```

It reports missing/blank fields, labels, malformed records and normalized duplicate/conflicting messages by physical source line. A valid populated file without review confirmation reports `pending_review_confirmation`. Fix issues and check every label yourself, then attest that review explicitly:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_sentiment --reviewed
```

`--reviewed` records the caller's confirmation. File existence and assistant development checks do not imply human authorship or review. All three sentiments must be represented for evaluation; smaller/imbalanced datasets are reported with their actual counts and limitations.

Reports go to `reports/sentiment_evaluation/`. `status.json` and `report.md` describe the current evaluation state. Reviewed and development runs have distinct timestamped directories under `runs/`, preserving earlier evidence. A successful reviewed run saves dataset hash, model revision, versions, duration, predictions, errors, accuracy, macro-F1, per-class precision/recall/F1 and confusion matrix. If inference fails on some rows, all-row accuracy counts those failures as wrong; coverage and successful-subset metrics are explicitly separated and status is incomplete. Pending/invalid data or missing weights produces no invented metrics.

`--development` uses identified assistant-authored messages **without expected labels or accuracy metrics**, including a long message and URL/username preprocessing. The published assessment fixtures are checked in a separate report: T001 expects Negative, T002 expects Positive, and ignored T003 is not passed to sentiment. A mismatch stays visible; there are no exceptions to force the feature request to Positive.

## Limits and attribution

The model was trained on English tweets and sentiment data from TweetEval. Customer-support language, politeness, sarcasm, mixed feelings and product-specific complaints can differ from that domain. Its sentiment expresses model interpretation, not ticket urgency or category. Scores are not calibrated correctness guarantees; the small assessment sample cannot establish support-ticket accuracy. Your independent reviewed examples are still needed before reporting that accuracy. [Cardiff model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

The original Cardiff NLP/TimeLMs model is licensed CC BY 4.0; retain the downloaded model card and attribution when transferring its weights. Model files/caches stay outside Git. No external artifact upload, API request, sentiment training or held-out intent-test evaluation belongs to this stage.

To reuse the model elsewhere, transfer the entire `models/sentiment/` directory, including `sentiment_source.json`, the tokenizer, configuration, weights and model card. Run the development command on the receiving machine to check loading. No external model upload has been performed.

## Actual checks — 2026-10-26

The exact official **501,045,531-byte** checkpoint was acquired and its SHA-256 matched upstream LFS: `4d24a3e32a88ed1c4e5b789fc6644e2e767500554e954b27dccf52a8e762cbae`. The original large HTTP download stalled; a checked parallel byte-range recovery completed in **355.25 seconds**, including assembly, checksum verification and tokenizer preparation. This acquisition method is recorded honestly in the local source metadata. No weights were trained or changed.

The current Transformers classification class deliberately creates RoBERTa without a pooler. The original Cardiff file also contains two unused `roberta.pooler.dense` tensors. Inspection of the installed constructor and saved three-class head confirmed this compatibility detail. The loader explicitly permits and records only those unused tensors and known legacy buffers; any missing encoder/classifier tensor or unknown extra trainable tensor still fails. The library may print its unused-pooler warning on loading; this does not indicate an untrained classification head.

Real CPU inference passed on five identified assistant development messages, with **no expected labels or accuracy calculation**: the actual predictions were Positive, Neutral, Negative, Neutral and Negative. The long fifth message had **1,082 tokens / 512 used**, and its truncation flag was true. The URL/username example used `@user` and `http` while preserving the original text in its report. Source, revision, versions, discarded-key list and actual predictions are retained under [development reports](../reports/sentiment_development/status.json).

The published assessment checks were separate:

| Ticket | Published expectation | Actual result |
|---|---|---|
| T001 checkout crash | Negative | Negative (0.8986) |
| T002 dark-mode request | Positive | Positive (0.9744) |
| T003 social greeting | Ignored | Sentiment not run |

These matches are visible fixture checks, not reviewed support-ticket accuracy. The exact results are in the timestamped run's `assessment_examples.json`. Future feature requests can receive Neutral or Negative; tests ensure outputs are never forced to Positive.

A separate real-model Streamlit AppTest session loaded both models on CPU. Sentiment had zero loads/calls on startup, validation, category analysis and ordinary rerun; its explicit action loaded once and predicted **only five qualifying rows**. Explicit reanalysis made ten predictions total while loads stayed at one. Outcomes remained **5 partial / 3 ignored / 4 errors / 0 fully processed**. See [streamlit_real_check.json](../reports/sentiment_development/streamlit_real_check.json). This is a simulated Streamlit session, not a browser visual check; CUDA failure routes are tested with mocks.

The default evaluation command reports [pending_reviewed_data](../reports/sentiment_evaluation/status.json): the template contains zero labeled rows and no reviewed metrics exist. The evaluation implementation is ready to save real metrics/errors after you supply and confirm your labels. No held-out intent-test file was opened.

The final full suite passed **258 tests in 65.47 seconds**. New Stage 6 cases include 34 mocked model tests, 21 evaluation/review tests, 13 mocked UI tests, 8 shared workflow tests and 7 setup guards. Those deterministic tests are distinct from the actual local model/UI checks above.
