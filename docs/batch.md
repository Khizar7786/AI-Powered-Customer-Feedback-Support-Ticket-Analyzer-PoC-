# Analyze and save a batch (Stage 8)

The CLI and Streamlit call the same ingestion, conservative filtering, local category/sentiment and structured OpenAI workflow. SQLite stores validated outcomes and lets either interface reuse work completed by the other. Replies remain unsent drafts. [Stage 9 review/export controls](interface.md) now add filters, details, downloads and shared CSV/JSON file exports.

## Run

Use the trained category checkpoint, downloaded sentiment model and ignored OpenAI configuration described in [setup.md](setup.md) and [generation.md](generation.md):

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze --input data/sample_tickets.csv
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The default database is `outputs/tickets.db`. CLI `--database PATH` selects another file; Streamlit uses the project default. CPU is the batch default, keeping local GPU use conservative. CLI `--device auto` allows the existing local models' CUDA/CPU fallback. No training/download occurs during analysis.

In Streamlit, load/upload and validate the CSV, then explicitly click **Analyze tickets**. Category/sentiment-only buttons remain in the advanced expander. Outcome tabs, filters and details show complete fields or useful partials, ignored/error reasons, original text, truncation and reuse/persistence status. Ordinary reruns only display session results; they do not query/insert into SQLite or call models/API again. Each deliberate batch submission has an audit run. Both interfaces also save the four export files beside the database.

The sample deliberately contains input errors. CLI exit **0** means a batch without errors, **1** means recorded row/file/persistence errors, and **2** means input/setup/database initialization failed. Exit 1 on this sample is expected; inspect the actual counts and errors rather than treating all rows as successful.

## Identity and deliberate retries

An analysis identity combines **ticket ID + exact original-message hash + configuration hash**. ID whitespace is trimmed and case is preserved. Even a message whitespace change changes input identity. Configuration includes content fingerprints of model weights/tokenizers/labels, public provider/model/settings, prompt/schema versions, relevant policy/preprocessing code and installed library versions. Model-directory locations and API keys are excluded; rotating a key does not invalidate successful results.

Matching `processed` and `ignored` outcomes are reused without local inference or a paid request, including after restarting Python or switching between CLI and Streamlit. Matching errors are reused by default too. These flags have different purposes:

| Intent | CLI | Streamlit |
|---|---|---|
| Allow new text/configuration for a previously analyzed ID | `--reprocess` | Allow analysis of changed messages or configuration |
| Retry a previous error under its matching identity | `--retry-errors` | Retry previous errors explicitly |

`--reprocess` does not force a matching successful input to call the API again. Previous input/configuration versions remain stored. A changed input/configuration without authorization produces a source-row `reprocess_required` error in the run audit, without model/API work. A corrected invalid CSV row can be submitted normally: invalid records do not reserve the valid ticket ID.

## Outcome accounting

- **processed:** all local predictions and generated fields pass the shared Pydantic validation and are saved. Product/feature may be `null` when absent.
- **ignored:** conservative blank/exact-greeting rule or semantic Non-Qualifying category, with a reason. No downstream paid request.
- **error:** input, model, API, reprocessing, coordination or persistence failure. Available validated local fields remain visible, but do not count as complete.

Each parsed CSV record has exactly one final outcome; processed + ignored + error equals rows read. Physical source lines include the header and quoted multiline records can span lines. Duplicate IDs are errors for every occurrence. Malformed records retain available original fields and raw CSV text.

File-level errors block model/API work. The parser cannot invent row boundaries after broken quoting: one error record covers the uncertain remaining source lines. Wrong headers/encoding can produce zero recoverable rows; the file error is saved separately in the batch run, not disguised as a ticket. Counts refer to recoverable records rather than guessed physical-line tickets.

## SQLite contents and safety

| Table | Purpose |
|---|---|
| `outcomes` | One canonical record per identity: originals/normalized text, fields/partials, status/reason/error, input/config hashes, model/prompt metadata, receipt and UTC created/updated times |
| `batch_runs` | Each explicit submission's source hash/name, configuration, UTC times, file/storage errors and reconciled counts |
| `batch_rows` | Every source record's run position/lines and immutable outcome snapshot, including reused results and blocked decisions |
| `claims` | Unfinished work reservations, separate from the three final outcome statuses |

Additional run-audit appearances are intentional history, not duplicate canonical analyses. Customer values use SQL parameters. Each operation opens/closes its own connection; transactions commit a final outcome, source-row audit and claim release together. Model/API work happens outside the short database transaction. WAL and a bounded busy timeout support independent workers; no SQLite connection is resource-cached or shared across Streamlit threads. See [Python sqlite3 transaction/connection guidance](https://docs.python.org/3.12/library/sqlite3.html) and [SQLite transactions](https://www.sqlite.org/lang_transaction.html).

The claim is committed **before** model/API work. Concurrent requests for the same identity do not both run it. An interrupted worker or failed final commit leaves the claim in place rather than silently repeating a possibly paid request. Only after confirming the prior worker stopped, recover explicitly:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze --input data/sample_tickets.csv --recover-incomplete
# Add --retry-errors if replacing a matching saved error, and --reprocess for changed input/configuration.
```

Recovery may repeat an uncertain paid operation; there is no exactly-once billing guarantee after a crash/network failure. The previous owner cannot commit after ownership changes. If SQLite becomes unwritable, affected rows return errors with `persisted=false` and available analysis retained; their unfinished claims prevent silent paid repeats. Later rows are still attempted. A failed run/audit write is explicitly reported; an unwritable database cannot be claimed to contain those failed writes. Stored results are revalidated on read, and corrupt/mismatched records fail closed without an automatic API retry.

## Verification

The full suite passed **347 tests in 109.83 seconds**. Stage 8 adds 29 tests (18 pipeline/storage, four identity, three CLI and four UI) using explicit model/API doubles and temporary databases. They cover mixed-row accounting, partial failures, source text, duplicate/missing IDs, explicit reprocessing/retry, cross-run reuse, concurrent claims, commit rollback, interruption recovery, SQL values/closed connections, API configuration drift and Streamlit reruns. These deterministic tests are separate from the actual execution below.

On **2026-10-26**, the real CLI used the saved CPU models and OpenAI on `data/sample_tickets.csv`, saving [outputs/tickets.db](../outputs/tickets.db). Actual outcome counts were **5 processed / 3 ignored / 4 input errors = 12 records**, with five successful API attempts, no persistence errors and no reply sending. First-process wall time was **35.75 seconds**, including model setup and requests; this is one measured run, not a speed guarantee. The deliberate invalid rows produced exit 1. [Live batch report](../reports/batch/live_batch.json)

| Published fixture | Actual saved result |
|---|---|
| T001 | Bug Report / Negative / Urgent / Checkout |
| T002 | Feature Request / Positive / Medium / mobile app |
| T003 | Ignored: local category Non-Qualifying; no sentiment/API request |

A fresh CLI process reused **all 12** results in **1.00 second**, keeping 12 canonical records. Guarded model-loading/API methods were never called; those guards supplied no replacement predictions. [Replay evidence](../reports/batch/replay_check.json)

An actual Streamlit AppTest then submitted the same CSV through **Analyze tickets**, reused all 12 real SQLite results and displayed actual reply drafts. It made zero model loads/API calls; an ordinary rerun added no database rows/run. This checks Streamlit behavior, not visual browser appearance. [Streamlit evidence](../reports/batch/streamlit_check.json)

Fresh-process database inspection returned `integrity_check=ok`, zero foreign-key violations, **12 validated canonical outcomes**, zero duplicate identities and zero unfinished claims. Three intentional submissions have 36 source-row audit appearances; each run independently reconciles to 5/3/4. Saved model/prompt versions, hashes, original text and errors were inspected; the database contains no API key. [Inspection report](../reports/batch/database_inspection.json)

These visible fixtures verify integration, not held-out model accuracy. No held-out intent-test content was opened, no sentiment labels were generated and no training occurred. Independent reviewed sentiment metrics remain pending. Stage 9's [interface guide](interface.md) records later export/rerun checks separately from this historical Stage 8 evidence.
