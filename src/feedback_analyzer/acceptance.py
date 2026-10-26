"""Real pipeline acceptance checks on visible fixtures, separate from held-out accuracy."""

from contextlib import closing
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import sqlite3
import time

from feedback_analyzer.exports import build_exports, flat_row, save_exports
from feedback_analyzer.generation import GenerationError, validate_generated_details
from feedback_analyzer.ingestion import load_ticket_csv
from feedback_analyzer.pipeline import BatchRuntime, analyze_csv
from feedback_analyzer.pipeline_identity import build_identity
from feedback_analyzer.storage import OutcomeStore


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def compare_expectations(result, preview, expectations):
    """References check outcomes; they are never provided to any classifier/provider."""
    if len(result.rows) != len(expectations["rows"]) or len(result.rows) != len(preview.rows):
        raise ValueError("Acceptance fixture/reference record counts differ.")
    checks = []
    for row, source, expected in zip(result.rows, preview.rows, expectations["rows"]):
        if expected["ordinal"] != row.ordinal:
            raise ValueError("Acceptance reference order does not match source records.")
        actual = flat_row(row)
        actual["error_stage"] = row.outcome.error.stage if row.outcome.error else None
        matches = {field: actual[field] == value for field, value in expected.items() if field != "ordinal"}
        original_preserved = (row.outcome.original_message == source.original_message
                              and row.outcome.raw_record == source.raw_record)
        a = row.outcome.analysis
        quality = {"original_text_preserved": original_preserved, "persisted": row.persisted}
        if a and a.generated:
            try:
                validate_generated_details(a.generated.details, row.outcome.original_message)
                quality.update(grounded_feature_or_null=True, reply_schema_policy=True, unsent_draft=a.reply_is_draft)
            except ValueError:
                quality.update(grounded_feature_or_null=False, reply_schema_policy=False, unsent_draft=a.reply_is_draft)
        elif expected["status"] == "processed":
            quality.update(grounded_feature_or_null=False, reply_schema_policy=False, unsent_draft=True)
        checks.append({"ordinal": row.ordinal, "source_row": row.source_row, "ticket_id": row.outcome.ticket_id,
                       "expected": expected, "actual": actual, "field_matches": matches,
                       "engineering_checks": quality, "all_automated_checks_pass": all(matches.values()) and all(quality.values()),
                       "reply_review": "Automated schema/policy checks only; assistant qualitative review is recorded separately."})
    return {"expected_counts": expectations["expected_counts"], "actual_counts": result.counts,
            "counts_match": expectations["expected_counts"] == result.counts,
            "row_accounting_reconciles": sum(result.counts.values()) == len(preview.rows), "rows": checks,
            "mismatches": [{"ordinal": r["ordinal"], "ticket_id": r["ticket_id"],
                            "fields": [key for key, value in r["field_matches"].items() if not value],
                            "engineering_checks": [key for key, value in r["engineering_checks"].items() if not value]}
                           for r in checks if not r["all_automated_checks_pass"]]}


def verify_exports(result, bundle, directory):
    payload = json.loads(bundle.files["batch.json"])
    errors = json.loads(bundle.files["errors.json"])
    processed = list(csv.DictReader(StringIO(bundle.files["results.csv"].decode("utf-8-sig"), newline="")))
    ignored = list(csv.DictReader(StringIO(bundle.files["ignored.csv"].decode("utf-8-sig"), newline="")))
    manifest = json.loads((directory / "exports_manifest.json").read_bytes())
    exact = all((directory / name).read_bytes() == value for name, value in bundle.files.items())
    checksums = all(_hash((directory / name).read_bytes()) == sha for name, sha in manifest["sha256"].items())
    originals = [r["outcome"]["original_message"] for r in payload["rows"]] == [r.outcome.original_message for r in result.rows]
    reconciles = (len(processed) == result.counts["processed"] and len(ignored) == result.counts["ignored"]
                  and len(errors["rows"]) == result.counts["error"] and len(payload["rows"]) == len(result.rows))
    return {"saved_bytes_equal_shared_exports": exact, "manifest_checksums_match": checksums,
            "json_originals_preserved": originals, "export_row_accounting": reconciles,
            "processed_csv_rows": len(processed), "ignored_csv_rows": len(ignored), "error_json_rows": len(errors["rows"])}


class ObservedProvider:
    """Record actual adapter operations/attempts; never substitutes a response."""
    def __init__(self, provider, events):
        self.provider, self.events = provider, events

    @property
    def public_options(self):
        return getattr(self.provider, "public_options", None)

    def generate(self, context):
        event = {"message_sha256": _hash(context.message.encode()), "status": "started"}
        self.events.append(event)
        try:
            result = self.provider.generate(context)
            event.update(status="validated", attempts=result.receipt.attempts, model=result.receipt.model,
                         response_id=result.receipt.response_id, duration_seconds=result.receipt.duration_seconds)
            return result
        except GenerationError as error:
            event.update(status="error", code=error.code, attempts=error.attempts)
            raise


def run_acceptance(config, fixture_dir: Path, output_dir: Path, *, runtime=None, identity=None,
                   reprocess=False, retry_errors=False):
    fixture_dir, output_dir = Path(fixture_dir), Path(output_dir)
    references_bytes = (fixture_dir / "expectations.json").read_bytes()
    references = json.loads(references_bytes)
    inputs = {}
    for name in ("assessment", "mixed"):
        data = (fixture_dir / f"{name}.csv").read_bytes()
        preview = load_ticket_csv(data)
        if (preview.file_errors or len(preview.rows) != len(references[name]["rows"])
                or [row["ordinal"] for row in references[name]["rows"]] != list(range(1, len(preview.rows) + 1))
                or sum(references[name]["expected_counts"].values()) != len(preview.rows)):
            raise ValueError("Acceptance input/reference validation failed before model/API work.")
        inputs[name] = (data, preview)
    identity = identity or build_identity(config, "cpu")
    runtime = runtime or BatchRuntime.local(config, "cpu")
    original_factory = runtime.provider_factory
    events = []
    runtime = BatchRuntime(runtime.category, runtime.sentiment, lambda: ObservedProvider(original_factory(), events))
    records = []
    for name in ("assessment", "mixed"):
        data, preview = inputs[name]
        destination = output_dir / name
        store = OutcomeStore(destination / "tickets.db")
        event_start, started = len(events), time.perf_counter()
        result = analyze_csv(data, runtime, store, identity, source_name=f"{name}.csv",
                             reprocess=reprocess, retry_errors=retry_errors)
        bundle = build_exports(result, identity.metadata)
        save_exports(bundle, destination)
        archive = destination / "runs" / result.run_id
        save_exports(bundle, archive)
        comparison = compare_expectations(result, preview, references[name])
        exports = verify_exports(result, bundle, destination)
        with closing(sqlite3.connect(store.path)) as db:
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            saved_count = db.execute("SELECT count(*) FROM batch_rows WHERE run_id=?", (result.run_id,)).fetchone()[0]
        record = {"batch": name, "run_id": result.run_id, "input_sha256": _hash(data),
                  "expectations_sha256": _hash(references_bytes), "config_hash": identity.config_hash,
                  "duration_seconds": time.perf_counter() - started, "reused_rows": result.reused_count,
                  "comparison": comparison, "export_checks": exports,
                  "database_integrity": integrity, "saved_source_rows": saved_count,
                  "new_provider_events": events[event_start:], "reply_sent": False,
                  "held_out_data_used": False, "reference_authorship": references["authorship"],
                  "live_verification_scope": "Default runtime uses real saved CPU category/sentiment models and configured OpenAI. No LLM-only comparison or LLM judge.",
                  "status": "checks_passed" if not comparison["mismatches"] and comparison["counts_match"]
                            and all(exports[key] for key in ("saved_bytes_equal_shared_exports", "manifest_checksums_match", "json_originals_preserved", "export_row_accounting"))
                            and integrity == "ok" and saved_count == len(result.rows) else "mismatches_or_errors"}
        _json(archive / "checks.json", record)
        _json(destination / "checks.json", record)
        records.append(record)
    summary = {"batches": [{"batch": r["batch"], "status": r["status"], "actual_counts": r["comparison"]["actual_counts"],
                            "mismatches": r["comparison"]["mismatches"], "run_id": r["run_id"]} for r in records],
               "provider_operations": len(events), "successful_provider_operations": sum(e["status"] == "validated" for e in events),
               "known_http_attempts": sum(e.get("attempts", 0) for e in events),
               "config_hash": identity.config_hash, "held_out_data_used": False, "reply_sent": False,
               "limitations": "Visible assistant-authored acceptance references are engineering checks, not independently reviewed accuracy metrics. "
               "Automatic reply checks cannot certify complete semantic quality. Sentiment still lacks the user's independent reviewed evaluation set."}
    _json(output_dir / "summary.json", summary)
    return summary
