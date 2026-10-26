# Review and prepare the category dataset (Stage 3)

Run commands in PowerShell from the project root using the existing Python 3.12 environment. This workflow performs data checks only. It does not train a model or approve generated labels.

The frozen synthetic train/validation/test CSVs and manifest are included so the classifier can be retrained. Raw drafts, original audit notes, review working copies/provenance and detailed data-preparation reports remain local. To redo review or splitting, transfer those original inputs under `data/raw/` or pass `--input` and `--notes`. Tests generate their own fixtures.

## Current dataset review origin

The frozen local version has now received its one final Stage 10 category evaluation. Its split bytes/manifest remain unchanged; the external evaluation registry records test consumption. See [evaluation.md](evaluation.md). Do not use its result to tune/relabel while continuing to describe this test set as unseen.

On 2026-10-25 the user explicitly delegated the review and splitting work to the assistant, superseding the earlier requirement to wait for manual review for this corpus. All 600 records were read and assessed by the assistant/team. The `reviewed` flags for this version mean **user-authorized assistant review**. No human label validation is claimed. The included manifest records this origin; the local `reports/data_preparation/assistant_review.json` retains per-row decisions. Synthetic origin remains recorded. The review corrected 24 labels and conservatively replaced 67 flagged test candidates in a separate authoring context before overlap checking.

The manual procedure below remains available for subsequent corrections. Preserve the review origin when reporting results or preparing a new dataset version.

## 1. Audit the draft and create your working copy

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data --audit-only
```

This reads `data/raw/tickets_600_draft.csv` and the supplied `data/raw/dataset_audit.md`. It writes:

- `data/reviewed_messages.csv`: a separate **review working copy**, initially still unreviewed.
- `data/reviewed_messages.csv.provenance.json`: the source fingerprint and original ID/pool membership. Keep this sidecar with the copy.
- `reports/data_preparation/dataset_audit.md` and `.json`: actual counts, issue IDs, and reasons.
- `reports/data_preparation/dataset_review_decisions_template.csv`: pending adjudications, never automatic approval.

An audit-only command returning 0 means the audit ran, not that the dataset is approved. Rerunning it preserves an existing review copy and sidecar. Reports/templates are refreshed; keep your own decisions in the separate file described below. Malformed source structure prevents creation of an incomplete review copy. Preserve raw files; correct such errors in a new source version and use explicit `--input`/`--review-copy` paths.

The supplied draft has `ticket_id` as its ID header. This is reported explicitly; only the new copy uses the canonical training header:

```text
example_id,message,label,source,is_synthetic,scenario_group,review_status,split_pool
```

The application uploader accepts only `ticket_id,message`. Do not upload this training dataset there. The dataset contains 510 development and 90 test candidates; all are synthetic until provenance says otherwise, and synthetic origin remains recorded after review.

## 2. Review the rows yourself

Open `data/reviewed_messages.csv` in a CSV editor, keeping UTF-8 encoding and correct CSV quoting. Work in small batches. Read the original message and check one primary category:

| Label | Rule |
|---|---|
| Bug Report | Existing behavior is broken, crashing, or incorrect. A payment-screen crash is a bug. |
| Feature Request | New functionality or an enhancement is requested. |
| Billing Issue | A charge, invoice, subscription, refund, or payment-account problem. |
| General Inquiry | A genuine information/how-to question. Pure questions about payment policy need consistent review; a question mark does not turn a disputed charge into an inquiry. |
| Customer Feedback | An opinion or praise/criticism without a more specific issue or request. |
| Non-Qualifying | Greeting-only content, spam, unsolicited promotion, or unrelated text. A greeting followed by an issue and a short complaint remain qualifying. |

Correct `label`, unclear message text, or scenario grouping when justified. Preserve IDs and `split_pool`. Record true provenance if you author a replacement; do not mark manually authored replacements as generated. Keep `is_synthetic` as lowercase `true` or `false`. Mark **only rows you actually reviewed** as `reviewed`. Leave others `unreviewed`, or use `needs_correction`. Review test labels without model predictions. The program cannot certify that your review happened.

Related paraphrases belong in the same `scenario_group`; changing a group name does not make messages independent. Exact duplicates require correction even within a pool. If a test candidate overlaps development, replace it with a fresh independently authored test scenario under its existing ID, update its provenance/group, and review it again. Never move test rows or their content into development. The sidecar rejects changed IDs/pool membership and changed source bytes. Manifest change tracking records corrected fields.

## 3. Resolve duplicate and prior-audit findings

The original audit includes semantic overlap and label-policy questions that lexical checks can miss. They remain separate pending findings. Different IDs and a passing lexical check do not establish independence.

After correcting/reviewing rows, audit the **working copy** to get current issue IDs and its current file fingerprint:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data --audit-only --input data/reviewed_messages.csv
Copy-Item reports/data_preparation/dataset_review_decisions_template.csv data/review_decisions.csv
```

Copy the template only when creating your decision file; do not overwrite existing decisions you want to retain. Its columns are `issue_id,decision,notes,dataset_sha256`. Replace `pending` only after inspecting the implicated rows:

- Near-duplicate or prior pair: `distinct` with a reason the scenarios are independent; `related` if they share a scenario family. Related development groups are kept together. Related pairs across pools block freezing until the test scenario is replaced; re-audit and assess the replacement.
- Prior label-policy issue: `confirmed`, with the rationale for the final label after any correction.
- Prior audit notes as a whole: `acknowledged`, with a note confirming you read and resolved their findings, including anything not individually imported.

Each accepted decision needs a nonempty explanation and the **current input file's SHA-256** from the report/template. Editing the dataset makes older decisions stale. Re-audit and reassess them; never update hashes merely to bypass review. Hard errors and unreviewed rows cannot be waived by a decision. Lexical scores are diagnostics, not semantic probabilities.

## 4. Freeze the reviewed splits

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data
```

The default reads `data/reviewed_messages.csv`, its provenance sidecar, and `data/review_decisions.csv` when present. Pending review/errors return exit code 2 and create no final splits. Correct the report and rerun. The program requires every row to be reviewed rather than silently discarding unreviewed examples.

When checks pass, `data/splits/` receives `train.csv`, `validation.csv`, `test.csv`, `manifest.json`, and `split_report.md`. Seed 42 balances whole development groups toward 420 train / 90 validation; the original 90 test-pool rows remain test-only. Actual class/count deviations are reported when grouping needs them. Every class must appear in each split; infeasible grouping requires more independent development scenarios, not splitting paraphrases across train/validation.

CSV splits retain metadata for traceability. `feedback_analyzer.dataset_split.model_records()` projects **only message text into `text` and category into `label`** for future model training. IDs, pool, group, provenance, and review metadata are never model features.

The manifest records dataset/source/split hashes, IDs, groups, seed, applied decisions, and corrected-field names. An identical repeat verifies frozen files without rewriting them. Changed data/configuration or altered split files are refused; use an explicit new `--output-dir` for a new dataset version. Once model choices are fixed, the held-out test is evaluated in Stage 10; do not use it for training or tuning.

All paths can be set explicitly; see `python -m scripts.prepare_data --help`. Output paths cannot overwrite inputs, audit notes, human decisions, or raw folders. Keep raw drafts, the review file/sidecar, and your decision file as provenance. No human review, final split readiness, accuracy, or model results are implied by an audit report.
