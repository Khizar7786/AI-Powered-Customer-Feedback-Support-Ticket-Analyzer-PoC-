"""Read application ticket CSVs without models, API calls, or persistence."""

from collections import defaultdict
import csv
from io import StringIO
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from feedback_analyzer.filtering import ignore_reason, normalize_message


# Python's default 128 KiB field limit rejects otherwise valid long messages.
# Set this once, not around each parse, because CSV sessions may run concurrently.
# This value also fits Windows' C long and exceeds the app's upload size limit.
csv.field_size_limit(2**31 - 1)


class PreviewRow(BaseModel):
    """One CSV record; source positions are physical file lines, starting at 1."""

    model_config = ConfigDict(extra="forbid")

    source_row: int = Field(ge=1)
    source_line_end: int = Field(ge=1)
    ticket_id: str | None = None
    original_message: str | None = None
    normalized_message: str | None = None
    status: Literal["candidate", "ignored", "error"]
    issues: list[str] = Field(default_factory=list)
    raw_record: str | None = None


class CsvPreview(BaseModel):
    """Validation results only; candidates have not been analyzed or processed."""

    model_config = ConfigDict(extra="forbid")

    rows: list[PreviewRow] = Field(default_factory=list)
    file_errors: list[str] = Field(default_factory=list)


def load_ticket_csv(data: bytes) -> CsvPreview:
    """Validate a UTF-8 (optionally BOM-prefixed) application CSV.

    Exactly ``ticket_id`` and ``message`` are required, in either order.
    Surrounding header/ID whitespace is stripped, IDs remain case-sensitive,
    and message fields are preserved exactly as decoded by the CSV parser.
    Labeled training data belongs to the separate data-preparation workflow.

    Wrong field counts are recoverable row errors. Invalid quoting stops the
    parser because later physical lines may belong to the damaged record.
    Prior parsed rows remain visible; all remaining raw text is one error.
    """
    preview = CsvPreview()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        preview.file_errors.append("The file is not valid UTF-8. Save it as CSV UTF-8 and upload again.")
        return preview

    reader = csv.reader(StringIO(text, newline=""), strict=True)
    source_lines = StringIO(text, newline="").readlines()
    try:
        header = next(reader)
    except StopIteration:
        preview.file_errors.append("The CSV file is empty. Include the header ticket_id,message and ticket rows.")
        return preview
    except csv.Error as error:
        preview.file_errors.append(f"Source row 1: invalid CSV header ({error}).")
        return preview

    header = [column.strip() for column in header]
    if len(header) != 2 or set(header) != {"ticket_id", "message"}:
        preview.file_errors.append(
            "Source row 1: expected exactly two unique columns: ticket_id,message (either order). "
            "Missing, duplicate, blank, or extra columns are invalid. Labeled training CSVs use a separate workflow."
        )
        return preview

    id_index = header.index("ticket_id")
    message_index = header.index("message")
    while True:
        start_line = reader.line_num + 1
        try:
            fields = next(reader)
        except StopIteration:
            break
        except csv.Error as error:
            end_line = max(start_line, len(source_lines))
            issue = (
                f"CSV parsing failed ({error}); parsing stopped. "
                "The remaining raw text is included because later row boundaries are uncertain."
            )
            preview.rows.append(
                PreviewRow(
                    source_row=start_line,
                    source_line_end=end_line,
                    status="error",
                    issues=[issue],
                    raw_record="".join(source_lines[start_line - 1 :]),
                )
            )
            preview.file_errors.append(f"Source row {start_line}: {issue}")
            break

        ticket_id = fields[id_index].strip() if id_index < len(fields) else None
        message = fields[message_index] if message_index < len(fields) else None
        issues = []
        if len(fields) != 2:
            issues.append(f"Expected 2 CSV fields, found {len(fields)}. Quote messages containing commas or newlines.")
        if not ticket_id:
            issues.append("Missing ticket_id.")
        preview.rows.append(
            PreviewRow(
                source_row=start_line,
                source_line_end=reader.line_num,
                ticket_id=ticket_id or None,
                original_message=message,
                normalized_message=normalize_message(message) if message is not None else None,
                status="error" if issues else "candidate",
                issues=issues,
                raw_record="".join(source_lines[start_line - 1 : reader.line_num]) if issues else None,
            )
        )

    if not preview.rows:
        preview.file_errors.append("The CSV has a header but no ticket rows.")
        return preview

    by_id: dict[str, list[PreviewRow]] = defaultdict(list)
    for row in preview.rows:
        if row.ticket_id is not None:
            by_id[row.ticket_id].append(row)
    for rows in by_id.values():
        if len(rows) > 1:
            source_rows = ", ".join(str(row.source_row) for row in rows)
            for row in rows:
                row.status = "error"
                row.issues.append(f"Duplicate ticket_id in source rows {source_rows}; all occurrences require correction.")

    for row in preview.rows:
        if row.status == "candidate":
            reason = ignore_reason(row.original_message)
            if reason is not None:
                row.status = "ignored"
                row.issues.append(reason)
    return preview
