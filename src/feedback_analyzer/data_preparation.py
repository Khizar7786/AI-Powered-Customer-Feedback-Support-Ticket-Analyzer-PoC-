"""Human-reviewed dataset preparation; never train or approve a label."""

from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from feedback_analyzer.dataset_audit import COLUMNS, AuditResult, load_dataset, make_issue
from feedback_analyzer.dataset_split import dataset_fingerprint, prepare_grouped_splits


DECISION_COLUMNS = ("issue_id", "decision", "notes", "dataset_sha256")
PAIR_CODES = {"near_duplicate", "prior_audit_pair"}
STRUCTURAL_CODES = {"file_read", "file_encoding", "file_schema", "csv_parse", "csv_width"}


@contextmanager
def _staging_directory(parent: Path):
    """Inherit normal workspace ACLs; Windows private temp ACLs survive renames."""
    temporary = parent / (".prepare-splits-" + uuid4().hex)
    temporary.mkdir()
    try:
        yield temporary
    finally:
        resolved = temporary.resolve()
        if resolved.parent != parent.resolve() or not resolved.name.startswith(".prepare-splits-"):
            raise ValueError("Unexpected staging directory; refusing cleanup.")
        if temporary.exists():
            shutil.rmtree(temporary)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_text(rows: list[dict], columns=COLUMNS) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_prior_notes(audit: AuditResult, notes_path: Path | None) -> dict:
    """Carry earlier semantic findings forward; a lexical pass cannot clear them.

    Recognize numbered N/F findings and the label-policy table in the supplied
    Markdown format. Other notes require a separate explicit acknowledgement.
    Notes are evidence, never executable instructions or permission to relabel.
    """
    if notes_path is None:
        return {}
    if not notes_path.is_file():
        audit.issues.append(make_issue("prior_notes_missing", "error", [], f"Audit notes not found: {notes_path.name}"))
        return {}
    content = notes_path.read_text(encoding="utf-8-sig")
    notes_hash = file_hash(notes_path)
    historical = re.search(r"\|\s*" + re.escape(Path(audit.source_path).name) + r"\s*\|\s*\d+\s*\|\s*([0-9a-f]{64})\s*\|", content)
    if historical and historical.group(1) != audit.source_sha256:
        audit.issues.append(make_issue(
            "prior_notes_source_mismatch", "warning", [],
            "The prior notes record different source bytes. Current counts and checks were recomputed; "
            "the historical report is not proof of this file's present integrity.", salt=notes_hash,
        ))
    row_ids = {row["example_id"] for row in audit.rows}
    known_pairs = {frozenset(issue.example_ids) for issue in audit.issues if issue.code in PAIR_CODES | {"exact_duplicate"}}
    imported = 0
    in_label_table = False
    for line in content.splitlines():
        if line.startswith("## "):
            in_label_table = "label-policy" in line.casefold()
        if not line.startswith("|"):
            continue
        identifiers = sorted(set(re.findall(r"\b(?:DEV|TEST)\d{4}\b", line)) & row_ids)
        numbered_pair = re.match(r"\|\s*[NF]\d+\s*\|", line)
        if numbered_pair and len(identifiers) >= 2:
            key = frozenset(identifiers)
            if key not in known_pairs:
                audit.issues.append(make_issue(
                    "prior_audit_pair", "review", identifiers,
                    "Prior audit flags possible shared scenario/template. Inspect these IDs in the review copy; "
                    "record distinct or related with a reason. Lexical screening cannot clear this finding.",
                    salt=notes_hash + line,
                ))
                known_pairs.add(key)
                imported += 1
        elif in_label_table and len(identifiers) == 1:
            audit.issues.append(make_issue(
                "prior_audit_label", "review", identifiers,
                "Prior audit flags a billing/inquiry label-policy question. Check the category guide, "
                "correct if needed, and explicitly confirm the final label with a reason.", salt=notes_hash + line,
            ))
            imported += 1
    # Even when a note format is not recognized, its findings must not vanish.
    if content.strip():
        audit.issues.append(make_issue(
            "prior_audit_notes", "review", [],
            "Read the supplied audit notes, including limitations and any findings not imported as individual issues. "
            "Record acknowledged with a substantive note after resolving their findings.", salt=notes_hash,
        ))
    return {"path": str(notes_path.resolve()), "sha256": notes_hash, "imported_findings": imported}


def resolve_reviews(audit: AuditResult, decisions_path: Path | None) -> tuple[list[str], list[tuple[str, str]], list[dict]]:
    """Apply explicit, dataset-bound human adjudications, without changing rows."""
    decisions = {}
    problems = []
    if decisions_path is not None and decisions_path.exists():
        try:
            with decisions_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                if reader.fieldnames != list(DECISION_COLUMNS):
                    raise ValueError("Expected decision columns: " + ",".join(DECISION_COLUMNS))
                for record in reader:
                    if None in record or any(value is None for value in record.values()):
                        raise ValueError("Malformed review-decision record.")
                    identifier = record["issue_id"]
                    if identifier in decisions:
                        raise ValueError("Duplicate issue_id in review decisions: " + identifier)
                    decisions[identifier] = record
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            problems.append(f"Review decisions could not be read: {error}")
    related_pairs = []
    applied = []
    rows = {row["example_id"]: row for row in audit.rows}
    for issue in audit.issues:
        if issue.severity == "error":
            problems.append(f"{issue.issue_id}: {issue.message}")
        elif issue.severity == "review":
            record = decisions.get(issue.issue_id)
            valid = bool(record and record["dataset_sha256"] == audit.source_sha256 and record["notes"].strip())
            decision = record["decision"].strip() if valid else "pending"
            if issue.code in PAIR_CODES and decision in {"distinct", "related"}:
                if decision == "related":
                    pools = {rows[identifier]["split_pool"] for identifier in issue.example_ids}
                    if len(pools) != 1:
                        problems.append(f"{issue.issue_id}: related scenarios cross pools; replace test candidates with fresh scenarios and re-audit.")
                        continue
                    related_pairs.extend((issue.example_ids[0], identifier) for identifier in issue.example_ids[1:])
            elif issue.code == "prior_audit_label" and decision == "confirmed":
                pass
            elif issue.code == "prior_audit_notes" and decision == "acknowledged":
                pass
            else:
                problems.append(f"{issue.issue_id}: {issue.message} (human review pending or decision stale)")
                continue
            applied.append(record)
    return problems, related_pairs, applied


def write_reports(audit: AuditResult, directory: Path, prior_notes: dict, problems: list[str], applied: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = audit.model_dump()
    # The review CSV holds messages; reports focus on counts, IDs and reasons.
    payload.pop("rows")
    payload.update(prior_notes=prior_notes, blocking_issues=problems, applied_decisions=applied,
                   ready_for_split=not problems, method={"near_cosine": 0.80, "near_sequence": 0.88,
                   "limits": "Lexical candidates only; no proof of semantic independence or human review."})
    _json_write(directory / "dataset_audit.json", payload)
    lines = ["# Dataset preparation audit", "", f"Source: `{Path(audit.source_path).name}`",
             f"Source SHA-256: `{audit.source_sha256}`", "",
             "Status: " + ("ready for split checks" if not problems else "human correction/review required"), "",
             "## Actual counts", "", "```json", json.dumps(audit.counts, indent=2), "```", "",
             "## Issues", "", "| Issue ID | Severity | Example IDs | Finding |", "|---|---|---|---|"]
    for issue in audit.issues:
        message = issue.message.replace("|", "\\|").replace("\n", " ")
        ids = ", ".join(issue.example_ids[:12]) or "dataset"
        if len(issue.example_ids) > 12:
            ids += f" (+{len(issue.example_ids) - 12} more; see JSON/review copy)"
        lines.append(f"| {issue.issue_id} | {issue.severity} | {ids} | {message} |")
    lines += ["", "## Interpretation", "",
              "Warnings about target counts do not certify accuracy or block a smaller reviewed development dataset. "
              "Errors and unresolved review findings prevent freezing. Different scenario IDs do not prove independence. "
              "Near-duplicate scores are lexical diagnostics, not model predictions.", "",
              "Only message text becomes model input; the label is the target. IDs, provenance, review status, "
              "groups and pools are excluded by the separate model_records projection."]
    if prior_notes:
        lines += ["", f"Prior audit: `{Path(prior_notes['path']).name}`, SHA-256 `{prior_notes['sha256']}`. "
                  f"Imported {prior_notes['imported_findings']} unresolved findings; original notes remain unchanged."]
    (directory / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    template = [{"issue_id": issue.issue_id, "decision": "pending", "notes": "",
                 "dataset_sha256": audit.source_sha256}
                for issue in audit.issues if issue.severity == "review" and issue.code != "human_review_pending"]
    (directory / "dataset_review_decisions_template.csv").write_text(csv_text(template, DECISION_COLUMNS), encoding="utf-8")


def create_review_copy(audit: AuditResult, path: Path, prior_notes: dict) -> str:
    if path.exists():
        return "Existing review copy preserved; no review edits or statuses were overwritten."
    if any(issue.code in STRUCTURAL_CODES for issue in audit.issues):
        return "No review copy created: correct structural CSV errors in a separate source version first."
    if not audit.rows:
        return "No review copy created: there are no readable records."
    provenance = Path(str(path) + ".provenance.json")
    if provenance.exists():
        raise ValueError("Review provenance already exists; choose a new review-copy path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a concurrent human edit.
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(csv_text(audit.rows))
    _json_write(provenance, {
        "source_path": str(Path(audit.source_path).resolve()), "source_sha256": audit.source_sha256,
        "original_pool_by_id": {row["example_id"]: row["split_pool"] for row in audit.rows},
        "prior_notes": prior_notes,
        "policy": "Rows/statuses copied unchanged; only a reported legacy ID header is canonicalized. No human approval implied.",
    })
    return "Created a separate review copy and provenance record; no labels or review statuses were approved."


def verify_provenance(audit: AuditResult, path: Path) -> dict:
    provenance_path = Path(str(path) + ".provenance.json")
    if not provenance_path.is_file():
        raise ValueError("Missing review provenance. First run --audit-only to create a separate review copy and its sidecar.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source = Path(provenance["source_path"])
    if not source.is_file() or file_hash(source) != provenance["source_sha256"]:
        raise ValueError("Original source is missing or changed since review-copy creation; audit a separate source version.")
    pools = {row["example_id"]: row["split_pool"] for row in audit.rows}
    if pools != provenance["original_pool_by_id"]:
        raise ValueError("Example IDs or split_pool changed from the source. Never move test examples into development or silently drop rows.")
    notes = provenance.get("prior_notes", {})
    if notes and (not Path(notes["path"]).is_file() or file_hash(Path(notes["path"])) != notes["sha256"]):
        raise ValueError("Prior audit notes changed or are missing; preserve their original provenance and re-audit a new version.")
    return provenance


def freeze_splits(audit: AuditResult, output: Path, seed: int, related_pairs: list[tuple[str, str]],
                  decisions: list[dict], provenance: dict) -> dict:
    splits = prepare_grouped_splits(audit.rows, seed=seed, related_pairs=related_pairs)
    fingerprint = dataset_fingerprint(audit.rows)
    decision_hash = hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()
    if output.exists():
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Output directory already exists without a manifest; choose a new output directory.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not (output / "split_report.md").is_file():
            raise ValueError("Existing frozen manifest/report is incomplete; refusing to overwrite it.")
        if (manifest.get("dataset_sha256"), manifest.get("source_sha256"), manifest.get("seed"), manifest.get("decisions_sha256")) != (fingerprint, audit.source_sha256, seed, decision_hash):
            raise ValueError("A different dataset/configuration is already frozen here; choose a new output directory.")
        if manifest.get("source_provenance") != provenance:
            raise ValueError("Review/source provenance changed after freezing; choose a new output directory.")
        if set(manifest.get("splits", {})) != {"train", "validation", "test"}:
            raise ValueError("Existing manifest does not list exactly train/validation/test.")
        for name, info in manifest["splits"].items():
            path = output / (name + ".csv")
            expected_hash = hashlib.sha256(csv_text(splits[name]).encode("utf-8")).hexdigest()
            if (not path.is_file() or file_hash(path) != info["sha256"] or info["sha256"] != expected_hash
                    or info.get("count") != len(splits[name])
                    or info.get("example_ids") != [row["example_id"] for row in splits[name]]):
                raise ValueError("Existing frozen split files changed; refusing to overwrite them.")
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with _staging_directory(output.parent) as temporary:
        stage = Path(temporary) / "frozen"
        stage.mkdir()
        metadata = {}
        for name, rows in splits.items():
            path = stage / (name + ".csv")
            path.write_bytes(csv_text(rows).encode("utf-8"))
            metadata[name] = {"count": len(rows), "labels": dict(Counter(row["label"] for row in rows)),
                              "sha256": file_hash(path), "example_ids": [row["example_id"] for row in rows],
                              "scenario_groups": sorted({row["scenario_group"] for row in rows})}
        source_audit = load_dataset(Path(provenance["source_path"]))
        original = {row["example_id"]: row for row in source_audit.rows}
        changes = {row["example_id"]: [key for key in COLUMNS if row[key] != original[row["example_id"]][key]]
                   for row in audit.rows if row != original[row["example_id"]]}
        manifest = {"version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                    "dataset_sha256": fingerprint, "source_sha256": audit.source_sha256,
                    "source_provenance": provenance, "reviewed_field_changes": changes,
                    "decisions_sha256": decision_hash, "applied_decisions": decisions,
                    "splits": metadata, "targets": {"train": 420, "validation": 90, "test": 90},
                    "algorithm": "seeded whole-group class/count balancing; related groups remain together",
                    "model_input": ["message"], "target": "label", "evaluation_run": False}
        review_origin = provenance.get("review_authorization", {})
        if review_origin:
            manifest["review_origin"] = review_origin
        _json_write(stage / "manifest.json", manifest)
        lines = ["# Frozen reviewed dataset splits", "", f"Dataset SHA-256: `{fingerprint}`", f"Seed: {seed}", "",
                 "| Split | Actual rows | Target rows | Class counts |", "|---|---|---|---|"]
        for name, info in metadata.items():
            lines.append(f"| {name} | {info['count']} | {manifest['targets'][name]} | {info['labels']} |")
        review_statement = (
            "Review was performed by the assistant on the user's explicit delegation; no human label review is claimed."
            if review_origin.get("method") == "assistant_user_authorized"
            else "Human review status is supplied by the reviewer, not certified by this script."
        )
        lines += ["", "Count deviations are reported, not hidden. Related scenario groups stay together. "
                  "All original test-pool IDs remain test-only. Source/review hashes and IDs are retained in the manifest.", "",
                  review_statement + " "
                  "Synthetic data remains synthetic after review. No model was trained or evaluated."]
        (stage / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.rename(stage, output)
    return manifest
