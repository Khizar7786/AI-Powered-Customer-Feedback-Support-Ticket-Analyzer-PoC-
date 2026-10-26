"""Analyze application tickets using local models, OpenAI and persistent SQLite reuse."""

import argparse
import json
from pathlib import Path

from feedback_analyzer.config import PROJECT_ROOT, load_config
from feedback_analyzer.pipeline import BatchRuntime, analyze_csv
from feedback_analyzer.pipeline_identity import build_identity
from feedback_analyzer.storage import OutcomeStore, StorageError
from feedback_analyzer.exports import ExportError, build_exports, save_exports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "outputs/tickets.db")
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--reprocess", action="store_true", help="Allow changed input/configuration for known ticket IDs; matching successes remain reused.")
    parser.add_argument("--retry-errors", action="store_true", help="Explicitly retry matching previous failures; may make another paid request.")
    parser.add_argument("--recover-incomplete", action="store_true", help="Recover unfinished reservations ONLY after the previous worker has stopped; billing may be uncertain.")
    args = parser.parse_args(argv)
    try:
        data = args.input.read_bytes()
        config = load_config()
        identity = build_identity(config, args.device)
        result = analyze_csv(data, BatchRuntime.local(config, args.device), OutcomeStore(args.database),
                             identity, source_name=args.input.name,
                             reprocess=args.reprocess, retry_errors=args.retry_errors, recover_incomplete=args.recover_incomplete)
    except (OSError, StorageError):
        print("Input/configuration or SQLite setup failed. Check paths, permissions and local model files; no successful batch is claimed.")
        return 2
    export_error = None
    try:
        save_exports(build_exports(result, identity.metadata), args.database.parent)
    except ExportError as error:
        export_error = str(error)
    print(json.dumps({"run_id": result.run_id, "database": str(args.database), "rows": len(result.rows),
                      "counts": result.counts, "fully_processed": result.fully_processed, "reused": result.reused_count,
                      "file_errors": result.file_errors, "persistence_errors": result.persistence_errors,
                      "export_error": export_error,
                      "errors": [{"source_row": row.source_row, "ticket_id": row.outcome.ticket_id,
                                  "stage": row.outcome.error.stage, "code": row.outcome.error.code,
                                  "message": row.outcome.error.message} for row in result.rows if row.outcome.error]}, indent=2))
    return 1 if result.counts["error"] or result.file_errors or result.persistence_errors or export_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
