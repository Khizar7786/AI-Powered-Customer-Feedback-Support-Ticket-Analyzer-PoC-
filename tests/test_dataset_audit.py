"""Dataset audit checks use small authored fixtures, not the held-out corpus."""

import csv
import hashlib
from pathlib import Path

import pytest

from feedback_analyzer.dataset_audit import COLUMNS, LABELS, audit_rows, load_dataset, normalized_text


MESSAGES = (
    "Opening checkout immediately crashes the application.",
    "Please add a dark appearance to the calendar.",
    "My card was debited twice for the same order.",
    "Where can I find the keyboard shortcut settings?",
    "The redesigned folders are pleasant to browse.",
    "Earn lottery cash today by visiting our prize website!",
    "Uploading a document produces a server error.",
    "A configurable weekly digest would be useful.",
    "The cancellation refund still has not reached my account.",
    "Can collaborators view my private address book?",
    "Your support agents explained everything clearly.",
    "A wandering blue giraffe ate the imaginary moon.",
)


def rows_fixture():
    return [
        dict(zip(COLUMNS, (
            f"EX{index:02d}", message, LABELS[index % 6], "authored_test_fixture", "true",
            f"scenario_{index}", "reviewed", "development" if index < 6 else "test",
        )))
        for index, message in enumerate(MESSAGES)
    ]


def write_csv(path: Path, rows, columns=COLUMNS):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get("example_id" if column == "ticket_id" else column, "") for column in columns])


def codes(result):
    return {issue.code for issue in result.issues}


def test_reviewed_small_fixture_has_coverage_but_honest_target_warnings():
    rows = rows_fixture()
    original = [dict(row) for row in rows]
    result = audit_rows(rows)
    assert not [issue for issue in result.issues if issue.severity in {"error", "review"}]
    assert "class_count_target" in codes(result)
    assert "dataset_count_target" in codes(result)
    assert result.counts["total"] == 12
    assert result.counts["pools"] == {"development": 6, "test": 6}
    assert result.counts["review_status"] == {"reviewed": 12}
    assert rows == original


def test_legacy_id_alias_is_visible_and_only_canonicalizes_separate_in_memory_rows(tmp_path):
    path = tmp_path / "draft.csv"
    columns = tuple("ticket_id" if column == "example_id" else column for column in COLUMNS)
    rows = rows_fixture()
    rows[0]["message"] = "  Quoted, text\r\nwith whitespace  "
    rows[0]["review_status"] = "unreviewed"
    write_csv(path, rows, columns)
    raw = path.read_bytes()
    result = load_dataset(path)
    assert result.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.original_columns[0] == "ticket_id"
    assert set(result.rows[0]) == set(COLUMNS)
    assert result.rows[0]["example_id"] == "EX00"
    assert result.rows[0]["message"] == "  Quoted, text\r\nwith whitespace  "
    assert result.rows[0]["review_status"] == "unreviewed"
    assert "legacy_id_column" in codes(result)
    assert "human_review_pending" in codes(result)
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("label", "Other", "invalid_label"),
        ("review_status", "approved_by_ai", "invalid_review_status"),
        ("split_pool", "training", "invalid_split_pool"),
        ("is_synthetic", "maybe", "invalid_is_synthetic"),
        ("source", "", "missing_value"),
        ("scenario_group", " ", "missing_value"),
        ("message", "\x00corrupt", "invalid_character"),
        ("scenario_group", " hidden_group ", "identity_whitespace"),
    ],
)
def test_required_provenance_and_enums_are_enforced(field, value, expected):
    rows = rows_fixture()
    rows[0][field] = value
    result = audit_rows(rows)
    assert expected in codes(result)
    assert any(issue.code == expected and issue.example_ids == ["EX00"] for issue in result.issues)


def test_duplicate_ids_and_cross_pool_normalized_messages_are_errors():
    rows = rows_fixture()
    rows[0]["message"] = "  Ｈｅｙ there.  "
    rows[6]["message"] = "HEY\tthere!"
    rows[1]["example_id"] = rows[0]["example_id"]
    result = audit_rows(rows)
    assert normalized_text(rows[0]["message"]) == normalized_text(rows[6]["message"])
    duplicate = next(issue for issue in result.issues if issue.code == "exact_duplicate")
    assert duplicate.severity == "error"
    assert duplicate.example_ids == ["EX00", "EX06"]
    assert "cross development/test" in duplicate.message
    assert "duplicate_id" in codes(result)


def test_groups_crossing_pools_are_errors_but_contrasting_development_intents_are_valid():
    rows = rows_fixture()
    rows[1]["scenario_group"] = rows[0]["scenario_group"]
    within = audit_rows(rows)
    issue = next(issue for issue in within.issues if issue.code == "shared_scenario_group")
    assert issue.severity == "warning"
    assert "Mixed labels" in issue.message
    assert "cross_pool_group" not in codes(within)
    rows[6]["scenario_group"] = rows[0]["scenario_group"]
    crossing = audit_rows(rows)
    issue = next(issue for issue in crossing.issues if issue.code == "cross_pool_group")
    assert issue.severity == "error"
    assert issue.example_ids == ["EX00", "EX01", "EX06"]


def test_near_duplicate_issue_is_stable_and_needs_fresh_review_after_text_changes():
    rows = rows_fixture()
    rows[0]["message"] = "The mobile calendar crashes whenever I select the annual overview."
    rows[6]["message"] = "The mobile calendar crashes whenever I select the monthly overview."
    first = audit_rows(rows)
    near = next(issue for issue in first.issues if issue.code == "near_duplicate" and set(issue.example_ids) == {"EX00", "EX06"})
    assert near.severity == "review"
    assert "across pools" in near.message
    assert near.similarity["char_tfidf_cosine"] > 0
    assert near.issue_id in {issue.issue_id for issue in audit_rows(rows).issues}
    rows[6]["message"] = "The mobile calendar crashes whenever I select the weekly overview."
    changed = next(issue for issue in audit_rows(rows).issues if issue.code == "near_duplicate" and set(issue.example_ids) == {"EX00", "EX06"})
    assert changed.issue_id != near.issue_id


@pytest.mark.parametrize("data", [b"", b"\xff", b"ticket_id,message\nT1,hello\n", b"example_id,example_id\nA,B\n"])
def test_invalid_dataset_files_return_readable_errors(tmp_path, data):
    path = tmp_path / "invalid.csv"
    path.write_bytes(data)
    result = load_dataset(path)
    assert not result.rows
    assert any(issue.severity == "error" for issue in result.issues)
    assert path.read_bytes() == data


def test_csv_width_and_parse_errors_are_not_silently_accepted(tmp_path):
    path = tmp_path / "damaged.csv"
    header = ",".join(COLUMNS) + "\n"
    path.write_text(header + "A,too,few\nB,too,many,fields,for,this,row,here,EXTRA\nC,\"unterminated\n", encoding="utf-8")
    result = load_dataset(path)
    width_errors = [issue for issue in result.issues if issue.code == "csv_width"]
    assert [issue.source_rows for issue in width_errors] == [[2], [3]]
    assert any(issue.code == "csv_parse" and issue.source_rows == [4] for issue in result.issues)
    assert result.rows[0]["example_id"] == "A"
    assert result.rows[0]["split_pool"] == ""
    assert "B" not in {row["example_id"] for row in result.rows}


def test_missing_file_and_missing_pool_coverage_cannot_be_frozen(tmp_path):
    missing = load_dataset(tmp_path / "absent.csv")
    assert "file_read" in codes(missing)
    result = audit_rows(rows_fixture()[:6])
    assert any(issue.code == "missing_class_coverage" and "test" in issue.message for issue in result.issues)
