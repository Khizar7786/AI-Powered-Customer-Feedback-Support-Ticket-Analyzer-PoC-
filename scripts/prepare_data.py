"""Audit raw drafts, create a human-review copy, or freeze reviewed splits."""

import argparse
import json
from pathlib import Path

from feedback_analyzer.dataset_audit import load_dataset
from feedback_analyzer.data_preparation import (
    add_prior_notes, create_review_copy, freeze_splits, resolve_reviews,
    verify_provenance, write_reports,
)


ROOT = Path(__file__).resolve().parents[1]


def _validate_destinations(source: Path, notes: Path | None, args, original_inputs=()) -> None:
    raw_roots = {(ROOT / "data/raw").resolve()}
    for path in (source, notes, *original_inputs):
        if path:
            raw_roots.update(parent for parent in path.resolve().parents if parent.name.casefold() == "raw")
    protected = {source, args.decisions.resolve()}
    protected.update(path.resolve() for path in original_inputs)
    if notes:
        protected.add(notes.resolve())
    destinations = [args.output_dir, args.report_dir / "dataset_audit.md",
                    args.report_dir / "dataset_audit.json", args.report_dir / "dataset_review_decisions_template.csv"]
    if args.audit_only:
        destinations += [args.review_copy, Path(str(args.review_copy) + ".provenance.json")]
    resolved_paths = [path.resolve() for path in destinations]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("Output paths collide; keep the review copy, audit reports, and splits separate.")
    for destination, resolved in zip(destinations, resolved_paths):
        if any(resolved.is_relative_to(raw) for raw in raw_roots):
            raise ValueError("Unsafe output path: raw files and folders must never be overwritten.")
        # Auditing the review file itself is read-only when that copy exists.
        preserved_copy = args.audit_only and destination == args.review_copy and destination.is_file()
        if resolved in protected and not (preserved_copy and resolved == source):
            raise ValueError("Unsafe output path: inputs, audit notes, and human decisions must never be overwritten.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="Audit a draft and create a review copy only; never freeze splits.")
    parser.add_argument("--input", type=Path, help="Default: raw draft in audit mode; reviewed_messages.csv otherwise.")
    parser.add_argument("--notes", type=Path, help="Prior audit Markdown; defaults to supplied notes for the project dataset.")
    parser.add_argument("--review-copy", type=Path, default=ROOT / "data/reviewed_messages.csv")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/review_decisions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/data_preparation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    source = (args.input or ROOT / ("data/raw/tickets_600_draft.csv" if args.audit_only else "data/reviewed_messages.csv")).resolve()
    raw_root = (ROOT / "data/raw").resolve()
    notes = args.notes
    if notes is None:
        candidate = raw_root / "dataset_audit.md" if source.is_relative_to(ROOT / "data") else source.parent / "dataset_audit.md"
        notes = candidate if candidate.is_file() else None
    try:
        _validate_destinations(source, notes, args)
        audit = load_dataset(source)
        if not args.audit_only and any(issue.code == "legacy_id_column" for issue in audit.issues):
            raise ValueError("Final reviewed input must use example_id; create a separate canonical copy with --audit-only first.")
        provenance = None
        original_inputs = []
        if not args.audit_only and source.exists():
            sidecar = Path(str(source) + ".provenance.json")
            if sidecar.is_file():
                recorded_inputs = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(recorded_inputs, dict) or not isinstance(recorded_inputs.get("prior_notes", {}), dict):
                    raise ValueError("Review provenance must contain a source/provenance object.")
                original_path = recorded_inputs.get("source_path")
                original_notes = recorded_inputs.get("prior_notes", {}).get("path")
                for value in (original_path, original_notes):
                    if value:
                        if not isinstance(value, str):
                            raise ValueError("Review provenance input paths must be strings.")
                        original_inputs.append(Path(value))
            try:
                provenance = verify_provenance(audit, source)
            except (ValueError, OSError, KeyError) as error:
                # Still emit the audit and human-review guidance before refusing.
                provenance_error = str(error)
            else:
                provenance_error = ""
                recorded = provenance.get("prior_notes", {})
                if recorded:
                    if notes is not None and notes.resolve() != Path(recorded["path"]).resolve():
                        raise ValueError("Prior audit notes differ from recorded review provenance.")
                    notes = Path(recorded["path"])
        else:
            provenance_error = ""
        _validate_destinations(source, notes, args, original_inputs)
        prior = add_prior_notes(audit, notes)
        problems, pairs, applied = resolve_reviews(audit, args.decisions)
        if provenance_error:
            problems.append(provenance_error)
        write_reports(audit, args.report_dir, prior, problems, applied)
        print(f"Rows: {audit.counts.get('total', 0)}; issues: {len(audit.issues)}; blocking findings: {len(problems)}.")
        print(f"Audit report: {args.report_dir / 'dataset_audit.md'}")
        if args.audit_only:
            print(create_review_copy(audit, args.review_copy, prior))
            print("Audit complete. This is not human approval; no final splits were created.")
            return 0
        if problems:
            print("Splits not created. Correct/review the dataset and current issue report first.")
            if provenance_error:
                print(provenance_error)
            return 2
        try:
            manifest = freeze_splits(audit, args.output_dir, args.seed, pairs, applied, provenance)
        except (ValueError, OSError, KeyError) as error:
            problems.append("Split generation stopped: " + str(error))
            write_reports(audit, args.report_dir, prior, problems, applied)
            raise
        print("Reviewed splits frozen/verified: " + ", ".join(f"{name}={info['count']}" for name, info in manifest["splits"].items()))
        print("Dataset SHA-256: " + manifest["dataset_sha256"])
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"Preparation stopped: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
