"""CLI checks with synthetic fixtures; no user examples are approved or split."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = (
    "example_id", "message", "label", "source", "is_synthetic",
    "scenario_group", "review_status", "split_pool",
)
LABELS = (
    "Bug Report", "Feature Request", "Billing Issue", "General Inquiry",
    "Customer Feedback", "Non-Qualifying",
)


def test_delegated_review_origin_is_preserved_without_claiming_human_review(tmp_path):
    source = tmp_path / "data" / "raw" / "draft.csv"
    write_rows(source, fixture_rows())
    result = invoke(tmp_path, source, "--audit-only")
    assert result.returncode == 0, result.stdout + result.stderr
    reviewed = tmp_path / "reviewed_messages.csv"
    sidecar = Path(str(reviewed) + ".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    origin = {"method": "assistant_user_authorized", "human_review_performed": False}
    provenance["review_authorization"] = origin
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    result = invoke(tmp_path, reviewed)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "splits" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_origin"] == origin
    assert "no human label review is claimed" in (tmp_path / "splits" / "split_report.md").read_text(encoding="utf-8")
    provenance["review_authorization"]["human_review_performed"] = True
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    result = invoke(tmp_path, reviewed)
    assert result.returncode == 2
    assert "provenance changed" in result.stdout


def fixture_rows(status="reviewed"):
    """Distinct deterministic text avoids making lexical test fixtures related."""
    rows = []
    for label_index, label in enumerate(LABELS):
        for position in range(4):
            identifier = f"F{label_index}_{position}"
            rows.append({
                "example_id": identifier,
                "message": hashlib.sha256(identifier.encode()).hexdigest(),
                "label": label,
                "source": "pytest_fixture",
                "is_synthetic": "true",
                "scenario_group": f"independent_{identifier}",
                "review_status": status,
                "split_pool": "test" if position == 3 else "development",
            })
    return rows


def write_rows(path, rows, *, legacy_header=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("ticket_id",) + COLUMNS[1:] if legacy_header else COLUMNS
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if legacy_header:
                row = {"ticket_id": row["example_id"], **{key: row[key] for key in COLUMNS[1:]}}
            writer.writerow(row)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(COLUMNS)
        return list(reader)


def invoke(tmp_path, source, *extra):
    notes = tmp_path / "notes.md"
    if not notes.exists():
        notes.write_text("", encoding="utf-8")
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.prepare_data",
            "--input", str(source),
            "--notes", str(notes),
            "--review-copy", str(tmp_path / "reviewed_messages.csv"),
            "--decisions", str(tmp_path / "review_decisions.csv"),
            "--output-dir", str(tmp_path / "splits"),
            "--report-dir", str(tmp_path / "reports"),
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=60,
    )


def test_audit_creates_canonical_unreviewed_copy_without_touching_raw_or_existing_edits(tmp_path):
    rows = fixture_rows("unreviewed")
    source = tmp_path / "data" / "raw" / "draft.csv"
    write_rows(source, rows, legacy_header=True)
    original = source.read_bytes()

    result = invoke(tmp_path, source, "--audit-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert source.read_bytes() == original
    copy = tmp_path / "reviewed_messages.csv"
    assert read_rows(copy) == rows
    assert {row["review_status"] for row in read_rows(copy)} == {"unreviewed"}
    provenance = Path(str(copy) + ".provenance.json")
    assert provenance.is_file()
    provenance_bytes = provenance.read_bytes()
    assert not (tmp_path / "splits").exists()
    assert (tmp_path / "reports" / "dataset_audit.md").is_file()
    report = json.loads((tmp_path / "reports" / "dataset_audit.json").read_text(encoding="utf-8"))
    assert report["counts"]["total"] == 24
    assert "legacy_id_column" in json.dumps(report)

    edited = list(rows)
    edited[0] = {**edited[0], "label": "Customer Feedback", "review_status": "reviewed"}
    write_rows(copy, edited)
    existing = copy.read_bytes()
    repeated = invoke(tmp_path, source, "--audit-only")
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert copy.read_bytes() == existing
    assert provenance.read_bytes() == provenance_bytes
    assert source.read_bytes() == original


def test_unreviewed_input_produces_report_but_never_final_splits(tmp_path):
    raw = tmp_path / "data" / "raw" / "draft.csv"
    write_rows(raw, fixture_rows("unreviewed"))
    audited = invoke(tmp_path, raw, "--audit-only")
    assert audited.returncode == 0, audited.stdout + audited.stderr
    source = tmp_path / "reviewed_messages.csv"
    original = source.read_bytes()

    result = invoke(tmp_path, source)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "review" in (result.stdout + result.stderr).lower()
    assert not (tmp_path / "splits").exists()
    assert (tmp_path / "reports" / "dataset_audit.md").is_file()
    assert source.read_bytes() == original


def test_reviewed_fixture_freezes_traceable_disjoint_splits_and_preserves_existing_freeze(tmp_path):
    rows = fixture_rows()
    raw = tmp_path / "data" / "raw" / "draft.csv"
    write_rows(raw, rows)
    audited = invoke(tmp_path, raw, "--audit-only")
    assert audited.returncode == 0, audited.stdout + audited.stderr
    source = tmp_path / "reviewed_messages.csv"
    original = source.read_bytes()

    result = invoke(tmp_path, source)
    assert result.returncode == 0, result.stdout + result.stderr
    directory = tmp_path / "splits"
    split_rows = {name: read_rows(directory / f"{name}.csv") for name in ("train", "validation", "test")}
    assert {row["example_id"] for row in split_rows["test"]} == {
        row["example_id"] for row in rows if row["split_pool"] == "test"
    }
    for name in ("train", "validation"):
        assert {row["split_pool"] for row in split_rows[name]} == {"development"}
    assert all({row["label"] for row in subset} == set(LABELS) for subset in split_rows.values())
    combined = [row for subset in split_rows.values() for row in subset]
    assert sorted(combined, key=lambda row: row["example_id"]) == sorted(rows, key=lambda row: row["example_id"])
    group_sets = [{row["scenario_group"] for row in subset} for subset in split_rows.values()]
    assert not any(group_sets[left] & group_sets[right] for left in range(3) for right in range(left + 1, 3))

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest_text = json.dumps(manifest)
    assert manifest["seed"] == 42
    assert len(manifest["dataset_sha256"]) == 64
    assert hashlib.sha256(original).hexdigest() in manifest_text
    assert all(row["example_id"] in manifest_text for row in rows)
    for name in split_rows:
        assert hashlib.sha256((directory / f"{name}.csv").read_bytes()).hexdigest() in manifest_text
    assert (directory / "split_report.md").is_file()
    assert source.read_bytes() == original

    frozen = {path.name: path.read_bytes() for path in directory.iterdir()}
    repeated = invoke(tmp_path, source)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == frozen

    changed = [{**row} for row in rows]
    changed[0]["message"] = "Replacement artificial fixture statement."
    write_rows(source, changed)
    changed_run = invoke(tmp_path, source)
    assert changed_run.returncode == 2, changed_run.stdout + changed_run.stderr
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == frozen


@pytest.mark.parametrize("destination", ["--review-copy", "--report-dir", "--output-dir"])
def test_outputs_cannot_alias_input_or_write_inside_raw_directory(tmp_path, destination):
    source = tmp_path / "data" / "raw" / "draft.csv"
    write_rows(source, fixture_rows("unreviewed"))
    original = source.read_bytes()
    target = source if destination == "--review-copy" else source.parent / "generated"

    result = invoke(tmp_path, source, "--audit-only", destination, str(target))
    assert result.returncode == 2, result.stdout + result.stderr
    assert source.read_bytes() == original
    assert not (source.parent / "generated").exists()
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "reviewed_messages.csv").exists()


def test_report_destination_cannot_overwrite_original_source_recorded_in_provenance(tmp_path):
    source = tmp_path / "source_version" / "dataset_audit.json"
    write_rows(source, fixture_rows())
    original = source.read_bytes()
    audited = invoke(tmp_path, source, "--audit-only")
    assert audited.returncode == 0, audited.stdout + audited.stderr
    reviewed = tmp_path / "reviewed_messages.csv"

    result = invoke(tmp_path, reviewed, "--report-dir", str(source.parent))
    assert result.returncode == 2, result.stdout + result.stderr
    assert source.read_bytes() == original
    assert not (source.parent / "dataset_audit.md").exists()
    assert not (tmp_path / "splits").exists()


@pytest.mark.parametrize("change", ["source", "pool"])
def test_freezing_rejects_changed_raw_source_or_test_pool_reassignment(tmp_path, change):
    rows = fixture_rows()
    raw = tmp_path / "data" / "raw" / "draft.csv"
    write_rows(raw, rows)
    audited = invoke(tmp_path, raw, "--audit-only")
    assert audited.returncode == 0, audited.stdout + audited.stderr
    reviewed = tmp_path / "reviewed_messages.csv"

    altered = [{**row} for row in rows]
    if change == "source":
        altered[0]["message"] = "Changed source fixture after creating provenance."
        write_rows(raw, altered)
    else:
        altered[3]["split_pool"] = "development"
        write_rows(reviewed, altered)

    result = invoke(tmp_path, reviewed)
    assert result.returncode == 2, result.stdout + result.stderr
    assert change in (result.stdout + result.stderr).lower()
    assert not (tmp_path / "splits").exists()


def test_missing_reviewed_input_explains_dependency_without_creating_splits(tmp_path):
    result = invoke(tmp_path, tmp_path / "reviewed_messages.csv")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "review" in (result.stdout + result.stderr).lower()
    assert not (tmp_path / "splits").exists()


@pytest.mark.parametrize("decision_case", ["distinct", "stale_hash", "related_cross_pool"])
def test_prior_semantic_findings_require_current_explicit_adjudication(tmp_path, decision_case):
    from feedback_analyzer.data_preparation import add_prior_notes, resolve_reviews
    from feedback_analyzer.dataset_audit import AuditResult

    rows = [dict(fixture_rows()[0]), dict(fixture_rows()[3])]
    rows[0]["example_id"] = "DEV0001"
    rows[1]["example_id"] = "TEST0001"
    before = [dict(row) for row in rows]
    audit = AuditResult(rows=rows, source_sha256="a" * 64, source_path="artificial.csv")
    notes = tmp_path / "prior.md"
    notes.write_text(
        "## Likely related scenarios\n"
        "| N01 | DEV0001 | TEST0001 | Possible shared scenario to adjudicate. |\n",
        encoding="utf-8",
    )
    prior = add_prior_notes(audit, notes)
    assert prior["imported_findings"] == 1
    pair = next(issue for issue in audit.issues if issue.code == "prior_audit_pair")
    assert resolve_reviews(audit, None)[0]

    decisions = tmp_path / "decisions.csv"
    with decisions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("issue_id", "decision", "notes", "dataset_sha256"))
        writer.writeheader()
        for issue in audit.issues:
            is_pair = issue.issue_id == pair.issue_id
            writer.writerow({
                "issue_id": issue.issue_id,
                "decision": ("related" if decision_case == "related_cross_pool" else "distinct") if is_pair else "acknowledged",
                "notes": "Artificial human adjudication fixture with a recorded reason.",
                "dataset_sha256": "b" * 64 if is_pair and decision_case == "stale_hash" else "a" * 64,
            })

    problems, related_pairs, applied = resolve_reviews(audit, decisions)
    assert not related_pairs
    if decision_case == "distinct":
        assert not problems
        assert len(applied) == 2
    else:
        assert problems
        assert pair.issue_id not in {record["issue_id"] for record in applied}
    assert audit.rows == before
