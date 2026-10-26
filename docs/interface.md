# Review and export tickets (Stage 9)

The Streamlit workflow now supports upload, validation, explicit analysis with progress, summary counts, outcome tabs, filters, details and four downloads. CLI and UI share the same export functions. Model inference/training behavior is unchanged; no training button or startup training exists.

## Use the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`:

1. Upload UTF-8 `ticket_id,message`, then click **Validate CSV**, or use **Load sample CSV**. Review source-line validation in **Review input validation**. Uploading/previewing makes no model/API request.
2. Check setup guidance beside **Analyze tickets**. Missing category/sentiment artifacts show their terminal setup commands; missing provider/model/key settings show the corresponding local `.env` correction. File presence/configuration presence are not live-access claims. Never paste a key into a ticket. Train separately using `python -m scripts.train_intent`; download sentiment separately with `python -m scripts.evaluate_sentiment --download-model`.
3. Click **Analyze tickets**. The progress bar advances as rows are reviewed. All counts reconcile to the parsed input records. Matching stored outcomes are reused; the explicit changed-input/error-retry controls retain the [Stage 8 semantics](batch.md).
4. Review **Processed**, **Ignored** and **Errors**. Processed tickets require complete validated fields and persistence. Errors keep useful partials without counting as complete. Category/priority/sentiment filters affect only processed rows; ignored/errors and total counts stay visible.
5. Use **Inspect a ticket** to select any source record, including missing/duplicate IDs. Details show the original text, available fields, ignored/error explanation, truncation warnings and the unsent reply draft. A source ordinal distinguishes repeated IDs.
6. Download **Processed CSV**, **Complete batch JSON**, **Ignored CSV**, or **Error details JSON**. Downloads always contain the complete batch snapshot, regardless of filters. Reply drafts are never sent.

The wide layout gives tables room; detailed local checks and the session-only single-ticket action are under **Local model checks and single-ticket analysis**. The palette uses a blue action accent and pale neutral surfaces, with native sans-serif controls for consistent legibility. Layout follows the actual work sequence rather than adding decorative cards.

## Saved files and fidelity

Both an explicit Streamlit analysis and the CLI command save these beside the selected database:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze --input data/sample_tickets.csv
```

| File | Contents |
|---|---|
| [results.csv](../outputs/results.csv) | Validated processed tickets, analysis fields, original/normalized text, source identity, hashes, draft and truncation flags |
| [ignored.csv](../outputs/ignored.csv) | Ignored tickets and reasons |
| [errors.json](../outputs/errors.json) | Source-row errors with raw records/available partials, plus file/persistence errors |
| [batch.json](../outputs/batch.json) | Every batch record, full validated outcomes, counts, public model/prompt metadata and original text |
| [exports_manifest.json](../outputs/exports_manifest.json) | Run identity and SHA-256 hashes for the four export files |

The latest explicitly saved batch replaces these filenames. Session downloads use their own prepared bytes, so another session's disk files cannot change the current user's download. Export bytes are built once after analysis; filters and download widgets never reenter the pipeline, create a database connection or rewrite files. **Retry saving exports** only retries file I/O when a save fails; it does not repeat analysis.

CSV uses the standard Python writer, quotes every cell and doubles embedded quotes, preserving commas and multiline values correctly. A UTF-8 BOM supports common Windows spreadsheet imports. Text beginning with formula markers (including leading whitespace/common invisible prefixes and full-width variants) or leading tab/newline controls gets an apostrophe prefix in the CSV viewing copy. `spreadsheet_escaped_fields` identifies modified cells. The source object, SQLite and JSON retain exact original strings; nullable JSON features remain `null`.

CSV quoting alone does not prevent spreadsheet formula interpretation. The prefix is a viewing precaution, not a universal guarantee across spreadsheet applications or repeated save/reopen cycles. Use JSON for lossless programmatic exchange; import IDs as text when a spreadsheet might coerce leading zeros or dates. See [OWASP CSV Injection guidance](https://owasp.org/www-community/attacks/CSV_Injection).

Files are staged before replacement. Each file replacement is atomic; the whole set is not one filesystem transaction. The checksum manifest is replaced last so a consumer can detect a partial/mismatched set. Check permissions/disk space and retry saving after failure; the in-memory downloads and already saved SQLite analyses remain available.

## Rerun and spending boundaries

The existing local models use `st.cache_resource`; result/export objects live in session state. Filters, details, checkbox changes and ordinary reruns only read that snapshot. Download buttons use prepared byte payloads and `on_click="ignore"`, which avoids a download-triggered app rerun; even a pending widget change causing a rerun does not repeat analysis. [Streamlit download behavior](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button)

Progress is observed around row persistence without modifying the core analysis/prompt identity. Consequently Stage 9's UI/export work can reuse Stage 8's real saved results instead of paying to regenerate them. Explicit Analyze submissions still create intentional batch-run audit records; unchanged canonical analysis identities remain unique.

## Recorded verification — 2026-10-26

The full suite passed **379 tests in 147.62 seconds**. Stage 9 adds 32 export/UI checks covering quoted multiline/Unicode text, formula/control-leading cells, nullable values, original-text fidelity, file/error exports, manifest hashes, save failures, progress, genuine filtering/empty combinations, readiness and rerun boundaries. These unit/UI tests use explicit doubles and temporary databases.

Separately, the actual CLI reused **12 real saved outcomes** (5 processed / 3 ignored / 4 deliberate input errors), wrote the four files and verified manifest hashes. It made zero additional API calls/model loads and retained 12 canonical records. Exit 1 is expected because the sample contains invalid inputs. [CLI export evidence](../reports/exports/cli_export_check.json)

An actual Streamlit AppTest used a simulated uploader carrying the real sample CSV, submitted the shared batch, reused all 12 real SQLite results, selected Feature Request / Medium / Positive to show T002, and displayed an actual original/reply draft. All four download payloads matched the saved files; they retained full-batch data despite filtering. Filters, details and reruns made no additional model loads/API calls or database inserts. This verifies the app's upload-to-download logic, not a physical browser download. [Full-flow evidence](../reports/exports/streamlit_flow_check.json)

**Manual browser verification remains pending.** Computer Use discovered Chrome but stopped when it could not determine the current browser URL confidently enough to enforce policy. No workaround or browser-download success is claimed. Once browser access is available, repeat steps 1–6 and verify the four downloaded files and unchanged database counts after filtering/downloading. The implementation and independent automated/file checks are complete.

No held-out intent-test content was opened, no further API spending or model training occurred in this stage, and no sentiment review labels were generated. Final intent evaluation remains Stage 10.
