"""Shared CSV ingestion behavior, without model loading or external services."""

from pathlib import Path

import pytest

from feedback_analyzer.ingestion import load_ticket_csv


@pytest.mark.parametrize(
    "data",
    [
        b"", b"\xef\xbb\xbf", b"ticket_id,message\n", b"ticket_id\nT1\n",
        b"ticket_id,ticket_id\nT1,hello\n", b"ticket_id,\nT1,hello\n",
        b"ticket_id,message,label\nT1,App crashes,Bug Report\n",
        b"example_id,message,label,split_pool\nDEV1,App crashes,Bug Report,development\n",
        b'ticket_id,"message\nT1,hello', b"ticket_id,message\nT1,\xff",
    ],
)
def test_file_contract_failures_return_guidance_without_rows(data):
    preview = load_ticket_csv(data)
    assert preview.file_errors
    assert not preview.rows


def test_bom_reordered_columns_and_multiline_messages_preserve_text_and_source_lines():
    data = '\ufeff message , ticket_id\r\n"  First line,\r\nsecond \'quoted\' line  ", T1 \r\nApp crashes,T2\r\n'.encode("utf-8")
    preview = load_ticket_csv(data)
    assert not preview.file_errors
    first, second = preview.rows
    assert first.ticket_id == "T1"
    assert first.original_message == "  First line,\r\nsecond 'quoted' line  "
    assert first.normalized_message == "First line, second 'quoted' line"
    assert (first.source_row, first.source_line_end) == (2, 3)
    assert (second.source_row, second.source_line_end) == (4, 4)
    assert [row.status for row in preview.rows] == ["candidate", "candidate"]


def test_missing_field_is_an_error_but_explicit_empty_message_is_ignored():
    preview = load_ticket_csv(b"ticket_id,message\nT1\nT2,\n,hi\nT3,App crashes,extra\n\nT4,App crashes\n")
    assert not preview.file_errors
    assert [row.status for row in preview.rows] == ["error", "ignored", "error", "error", "error", "candidate"]
    assert [row.source_row for row in preview.rows] == [2, 3, 4, 5, 6, 7]
    assert preview.rows[0].original_message is None
    assert preview.rows[1].original_message == ""
    assert "Missing ticket_id" in preview.rows[2].issues[0]
    assert preview.rows[3].original_message == "App crashes"
    assert preview.rows[3].raw_record == "T3,App crashes,extra\n"


def test_duplicate_ids_mark_all_occurrences_including_ignored_or_malformed_rows():
    preview = load_ticket_csv(b"ticket_id,message\nT1,App crashes\n T1 ,hi\nT1,other,extra\nt1,App crashes\n")
    assert [row.status for row in preview.rows] == ["error", "error", "error", "candidate"]
    for row in preview.rows[:3]:
        assert any("source rows 2, 3, 4" in issue for issue in row.issues)
    assert len(preview.rows[2].issues) == 2
    assert preview.rows[1].issues == ["Duplicate ticket_id in source rows 2, 3, 4; all occurrences require correction."]


@pytest.mark.parametrize(
    "bad_record",
    ['T2,"unclosed\nT3,still uncertain\n', 'T2,"closed"unexpected\nT3,App crashes\n'],
)
def test_malformed_quoting_preserves_prior_rows_and_stops_at_uncertain_record(bad_record):
    preview = load_ticket_csv(("ticket_id,message\nT1,App crashes\n" + bad_record).encode())
    assert preview.file_errors
    assert len(preview.rows) == 2
    good, bad = preview.rows
    assert good.status == "candidate"
    assert bad.status == "error"
    assert bad.source_row == 3
    assert bad.original_message is None
    assert bad.raw_record.startswith("T2,")
    assert "parsing stopped" in bad.issues[0]


def test_unicode_separators_do_not_shift_physical_rows_or_lose_unparsed_remainder():
    message = "Unicode line separator\u2028and next line\u0085remain in the field"
    remainder = 'T2,"closed"unexpected\nT3,App crashes\n'
    preview = load_ticket_csv((f"ticket_id,message\nT1,{message}\n" + remainder).encode())
    good, bad = preview.rows
    assert good.original_message == message
    assert (good.source_row, good.source_line_end) == (2, 2)
    assert (bad.source_row, bad.source_line_end) == (3, 4)
    assert bad.raw_record == remainder


def test_long_message_exceeding_default_csv_field_limit_is_preserved():
    message = "x" * 131_073
    preview = load_ticket_csv(f"ticket_id,message\nT1,{message}\nT2,App crashes\n".encode())
    assert not preview.file_errors
    first, second = preview.rows
    assert first.original_message == message
    assert first.normalized_message == message
    assert first.status == second.status == "candidate"
    assert (first.source_row, second.source_row) == (2, 3)
    assert second.original_message == "App crashes"


def test_sample_retains_exact_assessment_examples_without_hardcoded_predictions():
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_tickets.csv"
    preview = load_ticket_csv(sample.read_bytes())
    assert not preview.file_errors
    assert [row.original_message for row in preview.rows[:3]] == [
        "The payment page crashes every time I click 'Checkout' on Chrome. Please fix this ASAP!",
        "Hi team, it would be awesome if you could add a dark mode to the mobile app.",
        "Hello, hope you are having a nice day!",
    ]
    assert [row.status for row in preview.rows[:3]] == ["candidate"] * 3
    assert len(preview.rows) == 12
    assert {status: sum(row.status == status for row in preview.rows) for status in ("candidate", "ignored", "error")} == {
        "candidate": 6, "ignored": 2, "error": 4,
    }
    assert all("category" not in row.model_dump() and row.status != "processed" for row in preview.rows)
