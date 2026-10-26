"""Shared deterministic exports from validated results; no inference or database access."""

import csv
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import tempfile

from pydantic import ValidationError

from feedback_analyzer.batch_contracts import BatchResult


EXPORT_VERSION = "batch-export-v1"
CSV_POLICY = "Quoted UTF-8 BOM CSV; apostrophe-prefix formula/control-leading text. JSON preserves originals."
COLUMNS = ("source_row", "source_line_end", "ticket_id", "original_message", "normalized_message", "status",
           "category", "priority", "sentiment", "product_feature", "suggested_reply_draft", "ignored_reason",
           "error_stage", "error_code", "error_message", "classifier_input_truncated", "sentiment_input_truncated",
           "raw_record", "input_hash", "config_hash", "pipeline_version", "prompt_version", "api_model",
           "reused", "persisted", "recorded_at", "spreadsheet_escaped_fields")


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExportBundle:
    run_id: str
    files: dict[str, bytes]
    escaped_cells: int


def spreadsheet_text(value: str) -> tuple[str, bool]:
    """Protect the viewing copy, without changing the underlying ticket/result."""
    probe = re.sub(r"^[\s\ufeff\u200b]*", "", value)
    risky = (bool(probe) and probe[0] in "=+-@＝＋－＠") or value.startswith(("\t", "\r", "\n"))
    return ("'" + value, True) if risky else (value, False)


def flat_row(row, metadata=None) -> dict:
    metadata = metadata or {}
    o, a = row.outcome, row.outcome.analysis
    generated = a.generated if a else None
    details = generated.details if generated else None
    error = o.error
    return {
        "source_row": row.source_row, "source_line_end": row.source_line_end, "ticket_id": o.ticket_id,
        "original_message": o.original_message, "normalized_message": o.normalized_message, "status": o.status,
        "category": a.category.category if a and a.category else None,
        "priority": details.priority if details else None, "sentiment": a.sentiment.sentiment if a and a.sentiment else None,
        "product_feature": details.product_feature if details else None,
        "suggested_reply_draft": details.suggested_reply if details else None, "ignored_reason": o.reason,
        "error_stage": error.stage if error else None, "error_code": error.code if error else None,
        "error_message": error.message if error else None,
        "classifier_input_truncated": a.category.truncated if a and a.category else None,
        "sentiment_input_truncated": a.sentiment.truncated if a and a.sentiment else None,
        "raw_record": o.raw_record, "input_hash": row.input_hash, "config_hash": row.config_hash,
        "pipeline_version": metadata.get("pipeline_version"),
        "prompt_version": generated.receipt.prompt_version if generated else metadata.get("prompt_version"),
        "api_model": generated.receipt.model if generated else None,
        "reused": row.reused, "persisted": row.persisted, "recorded_at": row.recorded_at,
    }


def _csv(rows, metadata):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writeheader()
    count = 0
    for row in rows:
        record = flat_row(row, metadata)
        escaped = []
        for key, value in record.items():
            if isinstance(value, str):
                record[key], changed = spreadsheet_text(value)
                if changed:
                    escaped.append(key)
        record["spreadsheet_escaped_fields"] = ";".join(escaped)
        count += len(escaped)
        writer.writerow(record)
    return stream.getvalue().encode("utf-8-sig"), count


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _check_public_metadata(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(token in str(key).casefold() for token in ("api_key", "secret", "authorization", "access_token")):
                raise ExportError("Private configuration cannot be included in exports.")
            _check_public_metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_public_metadata(item)


def build_exports(result: BatchResult, metadata: dict | None = None) -> ExportBundle:
    """Build full-batch bytes once; UI filters never affect these downloads."""
    try:
        result = BatchResult.model_validate(result.model_dump())
        if any(row.outcome.status == "processed" and not row.persisted for row in result.rows):
            raise ExportError("A processed export requires validated, saved results.")
        metadata = metadata or {}
        _check_public_metadata(metadata)
        processed, processed_escaped = _csv([r for r in result.rows if r.outcome.status == "processed"], metadata)
        ignored, ignored_escaped = _csv([r for r in result.rows if r.outcome.status == "ignored"], metadata)
        payload = result.model_dump(mode="json")
        payload.update(export_version=EXPORT_VERSION, counts=result.counts, total_rows=len(result.rows),
                       fully_processed=result.fully_processed, model_metadata=metadata, csv_text_policy=CSV_POLICY)
        errors = {"export_version": EXPORT_VERSION, "run_id": result.run_id,
                  "file_errors": result.file_errors, "persistence_errors": result.persistence_errors,
                  "rows": [r.model_dump(mode="json") for r in result.rows if r.outcome.status == "error"]}
        return ExportBundle(result.run_id, {"results.csv": processed, "ignored.csv": ignored,
                                            "errors.json": _json(errors), "batch.json": _json(payload)},
                            processed_escaped + ignored_escaped)
    except (ValidationError, TypeError, ValueError):
        raise ExportError("Export validation failed. No replacement fields or output were invented.") from None


def save_exports(bundle: ExportBundle, directory: Path) -> None:
    """Stage bytes first, replace individual files atomically, write checksum manifest last."""
    directory = Path(directory)
    pending = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if set(bundle.files) != {"results.csv", "ignored.csv", "errors.json", "batch.json"}:
            raise ExportError("Unexpected export file names; no output was written.")
        manifest = _json({"run_id": bundle.run_id, "export_version": EXPORT_VERSION,
                          "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in bundle.files.items()}})
        for name, data in [*bundle.files.items(), ("exports_manifest.json", manifest)]:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".export-", suffix=".tmp", delete=False) as stream:
                pending.append((Path(stream.name), directory / name))
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        for source, target in pending:
            os.replace(source, target)
    except OSError:
        raise ExportError("Exports could not be fully saved. Check output permissions/disk space and retry saving; analysis does not need to run again.") from None
    finally:
        for source, _ in pending:
            try:
                source.unlink(missing_ok=True)
            except OSError:
                pass
