"""Run the assessment fixtures and a separate mixed batch through the complete real pipeline."""

import argparse
import json
from pathlib import Path

from feedback_analyzer.acceptance import run_acceptance
from feedback_analyzer.config import PROJECT_ROOT, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=PROJECT_ROOT / "data/acceptance")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/acceptance")
    parser.add_argument("--reprocess", action="store_true", help="Explicitly accept changed input/configuration; matching successes stay reused.")
    parser.add_argument("--retry-errors", action="store_true", help="Explicit retry may make another paid request. No automatic acceptance-output repair.")
    args = parser.parse_args(argv)
    try:
        record = run_acceptance(load_config(), args.fixtures, args.output_dir, reprocess=args.reprocess, retry_errors=args.retry_errors)
    except (OSError, ValueError, RuntimeError):
        print("Acceptance setup/run failed. Check local fixtures, models, output access and ignored API configuration. No live success is claimed.")
        return 2
    print(json.dumps(record, indent=2))
    return 0 if all(row["status"] == "checks_passed" for row in record["batches"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
