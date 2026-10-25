"""Audit labeled data without changing labels, review decisions, or source files.

Lexical similarity finds candidates for human inspection. It cannot establish
semantic independence, and passing this audit does not approve generated labels.
"""

from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, Field


COLUMNS = (
    "example_id", "message", "label", "source", "is_synthetic",
    "scenario_group", "review_status", "split_pool",
)
LABELS = (
    "Bug Report", "Feature Request", "Billing Issue", "General Inquiry",
    "Customer Feedback", "Non-Qualifying",
)
REVIEW_STATUSES = ("unreviewed", "reviewed", "needs_correction")
POOLS = ("development", "test")
NEAR_COSINE_THRESHOLD = 0.80
NEAR_SEQUENCE_THRESHOLD = 0.88

# Avoid rejecting valid long fields at Python's legacy 128 KiB CSV default.
csv.field_size_limit(2**31 - 1)


class AuditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    code: str
    severity: Literal["error", "warning", "review"]
    example_ids: list[str] = Field(default_factory=list)
    message: str
    source_rows: list[int] = Field(default_factory=list)
    similarity: dict[str, float] = Field(default_factory=dict)


class AuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, str]] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    source_sha256: str = ""
    source_path: str = ""
    original_columns: list[str] = Field(default_factory=list)
    counts: dict = Field(default_factory=dict)


def normalized_text(message: str) -> str:
    """Comparison only: NFKC/casefold and punctuation-tolerant whitespace."""
    text = unicodedata.normalize("NFKC", message).casefold()
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    normalized = " ".join(text.split())
    # Avoid equating every punctuation-only message with every other one.
    return normalized or " ".join(unicodedata.normalize("NFKC", message).casefold().split())


normalize_duplicate_text = normalized_text


def make_issue(
    code: str,
    severity: Literal["error", "warning", "review"],
    example_ids: list[str],
    message: str,
    *,
    source_rows: list[int] | None = None,
    salt: str = "",
    similarity: dict[str, float] | None = None,
) -> AuditIssue:
    """Use content-aware stable IDs so corrected pairs need a fresh decision."""
    identity = json.dumps([code, sorted(example_ids), salt], ensure_ascii=False)
    identifier = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return AuditIssue(
        issue_id=f"{code}-{identifier}", code=code, severity=severity,
        example_ids=example_ids, message=message, source_rows=source_rows or [],
        similarity=similarity or {},
    )


def _counts(rows: list[dict[str, str]]) -> dict:
    return {
        "total": len(rows),
        "labels": dict(Counter(row.get("label", "") for row in rows)),
        "pools": dict(Counter(row.get("split_pool", "") for row in rows)),
        "by_pool": {
            pool: {
                "total": sum(row.get("split_pool") == pool for row in rows),
                "labels": dict(Counter(row.get("label", "") for row in rows if row.get("split_pool") == pool)),
            }
            for pool in POOLS
        },
        "review_status": dict(Counter(row.get("review_status", "") for row in rows)),
        "synthetic": dict(Counter(row.get("is_synthetic", "") for row in rows)),
        "sources": dict(Counter(row.get("source", "") for row in rows)),
    }


def audit_rows(
    rows: list[dict[str, str]],
    source_sha256: str = "",
    source_path: str = "",
    original_columns: list[str] | None = None,
) -> AuditResult:
    """Validate canonical rows and report unresolved issues without mutating input."""
    result = AuditResult(
        rows=[dict(row) for row in rows], source_sha256=source_sha256,
        source_path=source_path, original_columns=original_columns or list(COLUMNS),
        counts=_counts(rows),
    )
    by_id: dict[str, list[int]] = defaultdict(list)
    by_text: dict[str, list[int]] = defaultdict(list)
    by_group: dict[str, list[int]] = defaultdict(list)
    unreviewed = []
    for index, row in enumerate(result.rows):
        example_id = row.get("example_id", "")
        ids = [example_id] if example_id else []
        fields = set(row)
        if fields != set(COLUMNS):
            result.issues.append(make_issue(
                "row_schema", "error", ids,
                f"Record {index + 1} does not contain exactly the eight training-data columns.", salt=str(index),
            ))
        for field in COLUMNS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                result.issues.append(make_issue(
                    "missing_value", "error", ids,
                    f"Record {index + 1} requires a nonblank string in {field}.", salt=f"{index}:{field}",
                ))
            elif "\x00" in value or "\ufffd" in value:
                result.issues.append(make_issue(
                    "invalid_character", "error", ids,
                    f"{field} contains a NUL or Unicode replacement character.", salt=f"{index}:{field}",
                ))
            elif field in {"example_id", "scenario_group"} and value != value.strip():
                result.issues.append(make_issue(
                    "identity_whitespace", "error", ids,
                    f"{field} has surrounding whitespace; correct it explicitly to avoid hidden identity/group collisions.",
                    salt=f"{index}:{field}",
                ))
        for field, allowed in (
            ("label", LABELS), ("review_status", REVIEW_STATUSES),
            ("split_pool", POOLS), ("is_synthetic", ("true", "false")),
        ):
            if row.get(field) not in allowed:
                result.issues.append(make_issue(
                    "invalid_" + field, "error", ids,
                    f"Record {index + 1} has unsupported {field}; allowed values: {', '.join(allowed)}.",
                    salt=f"{index}:{field}",
                ))
        if row.get("review_status") != "reviewed":
            unreviewed.extend(ids)
        if example_id:
            by_id[example_id].append(index)
        message = row.get("message", "")
        if isinstance(message, str) and message.strip():
            comparison = normalized_text(message)
            if comparison:
                by_text[comparison].append(index)
        group = row.get("scenario_group", "")
        if isinstance(group, str) and group.strip():
            by_group[group].append(index)

    if unreviewed:
        result.issues.append(make_issue(
            "human_review_pending", "review", unreviewed,
            f"{len(unreviewed)} records lack review_status=reviewed. Automated checks do not approve labels.",
        ))
    for name, mapping in (("duplicate_id", by_id), ("exact_duplicate", by_text)):
        for key, indexes in mapping.items():
            if len(indexes) > 1:
                ids = [result.rows[index]["example_id"] for index in indexes]
                pools = {result.rows[index].get("split_pool") for index in indexes}
                result.issues.append(make_issue(
                    name, "error", ids,
                    ("Repeated example_id" if name == "duplicate_id" else "Normalized exact duplicate messages")
                    + (" cross development/test pools." if len(pools) > 1 else " occur within a pool.")
                    + " Keep the raw draft and correct the review copy; do not move test content into development.",
                    salt=key,
                ))
    for group, indexes in by_group.items():
        if len(indexes) < 2:
            continue
        ids = [result.rows[index]["example_id"] for index in indexes]
        pools = {result.rows[index].get("split_pool") for index in indexes}
        labels = {result.rows[index].get("label") for index in indexes}
        if len(pools) > 1:
            result.issues.append(make_issue(
                "cross_pool_group", "error", ids,
                f"Scenario group {group!r} spans development and test; resolve independence before freezing.", salt=group,
            ))
        else:
            result.issues.append(make_issue(
                "shared_scenario_group", "warning", ids,
                f"Scenario group {group!r} must stay together in any train/validation subdivision."
                + (" Mixed labels can represent different intents; check the family without automatically relabeling it." if len(labels) > 1 else ""),
                salt=group,
            ))

    _audit_counts(result)
    _near_duplicates(result)
    return result


def _audit_counts(result: AuditResult) -> None:
    if not result.rows:
        result.issues.append(make_issue("empty_dataset", "error", [], "No usable dataset records were found."))
    for pool, expected in (("development", 85), ("test", 15)):
        counts = result.counts["by_pool"][pool]["labels"]
        missing = [label for label in LABELS if not counts.get(label)]
        if missing:
            result.issues.append(make_issue(
                "missing_class_coverage", "error", [],
                f"{pool} pool has no examples for: {', '.join(missing)}.", salt=pool,
            ))
        if any(counts.get(label, 0) != expected for label in LABELS):
            result.issues.append(make_issue(
                "class_count_target", "warning", [],
                f"{pool} target is {expected} examples per class; actual counts are in the report. "
                "Targets are learning goals, not permission to add, drop, or silently move examples.", salt=pool,
            ))
    if result.counts["total"] != 600 or any(result.counts["labels"].get(label, 0) != 100 for label in LABELS):
        result.issues.append(make_issue(
            "dataset_count_target", "warning", [],
            "Dataset target is 600 rows and 100 per class; report the actual counts and development-data limitation.",
        ))


def _near_duplicates(result: AuditResult) -> None:
    """Deterministic lexical screening; no neural model or external API."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    indexes = [index for index, row in enumerate(result.rows) if isinstance(row.get("message"), str) and normalized_text(row["message"])]
    if len(indexes) < 2:
        return
    texts = [normalized_text(result.rows[index]["message"]) for index in indexes]
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False).fit_transform(texts)
    similarities = cosine_similarity(vectors)
    for left in range(len(indexes)):
        for right in range(left + 1, len(indexes)):
            if texts[left] == texts[right]:
                continue  # Already a hard exact-duplicate issue.
            cosine = float(similarities[left, right])
            # Sequence matching is useful for small substitutions. Skip it when
            # lengths already make reaching the threshold mathematically impossible.
            maximum_ratio = 2 * min(len(texts[left]), len(texts[right])) / (len(texts[left]) + len(texts[right]))
            sequence = 0.0
            if maximum_ratio >= NEAR_SEQUENCE_THRESHOLD:
                matcher = SequenceMatcher(None, texts[left], texts[right], autojunk=False)
                if matcher.quick_ratio() >= NEAR_SEQUENCE_THRESHOLD:
                    sequence = matcher.ratio()
            if cosine < NEAR_COSINE_THRESHOLD and sequence < NEAR_SEQUENCE_THRESHOLD:
                continue
            row_left, row_right = (result.rows[indexes[value]] for value in (left, right))
            ids = [row_left.get("example_id", ""), row_right.get("example_id", "")]
            cross_pool = row_left.get("split_pool") != row_right.get("split_pool")
            result.issues.append(make_issue(
                "near_duplicate", "review", ids,
                "Possible lexical near-duplicate " + ("across pools" if cross_pool else "within one pool")
                + f" (character TF-IDF cosine={cosine:.3f}, sequence ratio={sequence:.3f}). "
                "A human must confirm distinct scenarios or correct the overlap/grouping; scores do not prove semantic equivalence.",
                salt=json.dumps(sorted((texts[left], texts[right])), ensure_ascii=False),
                similarity={"char_tfidf_cosine": round(cosine, 6), "sequence_ratio": round(sequence, 6)},
            ))


def load_dataset(path: Path) -> AuditResult:
    """Read a draft/review CSV, allowing only the explicitly reported ID alias."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as error:
        return AuditResult(source_path=str(path), issues=[make_issue(
            "file_read", "error", [], f"Dataset could not be read ({type(error).__name__}). Check the input path.",
        )], counts=_counts([]))
    fingerprint = hashlib.sha256(data).hexdigest()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return AuditResult(source_path=str(path), source_sha256=fingerprint, issues=[make_issue(
            "file_encoding", "error", [], "Dataset is not valid UTF-8. Save a separate UTF-8 review copy; preserve the raw file.",
        )], counts=_counts([]))
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        original = next(reader)
    except (StopIteration, csv.Error):
        return AuditResult(source_path=str(path), source_sha256=fingerprint, issues=[make_issue(
            "file_schema", "error", [], "Dataset is empty or its CSV header cannot be parsed.", source_rows=[1],
        )], counts=_counts([]))
    alias_columns = ["ticket_id" if column == "example_id" else column for column in COLUMNS]
    legacy_alias = len(original) == len(COLUMNS) and set(original) == set(alias_columns)
    if len(original) != len(COLUMNS) or len(set(original)) != len(COLUMNS) or (set(original) != set(COLUMNS) and not legacy_alias):
        return AuditResult(source_path=str(path), source_sha256=fingerprint, original_columns=original, issues=[make_issue(
            "file_schema", "error", [], "Expected exactly these training-data columns: " + ",".join(COLUMNS)
            + ". The application ticket_id,message CSV is a separate contract.", source_rows=[1],
        )], counts=_counts([]))
    canonical = ["example_id" if legacy_alias and column == "ticket_id" else column for column in original]
    rows = []
    parse_issues = []
    source_rows = []
    while True:
        start = reader.line_num + 1
        try:
            fields = next(reader)
        except StopIteration:
            break
        except csv.Error as error:
            parse_issues.append(make_issue(
                "csv_parse", "error", [], f"CSV parsing failed at source line {start} ({error}); later record boundaries are uncertain.",
                source_rows=[start], salt=str(start),
            ))
            break
        if len(fields) != len(COLUMNS):
            ids = [fields[canonical.index("example_id")]] if len(fields) > canonical.index("example_id") else []
            parse_issues.append(make_issue(
                "csv_width", "error", ids,
                f"Source line {start} has {len(fields)} fields; expected 8. It cannot be silently omitted from a frozen dataset.",
                source_rows=[start], salt=str(start),
            ))
            # Pad missing fields, retaining the known values in the review report.
            # Extra fields cannot fit the canonical CSV and are not silently truncated.
            if len(fields) > len(COLUMNS):
                continue
            fields += [""] * (len(COLUMNS) - len(fields))
        rows.append(dict(zip(canonical, fields)))
        source_rows.append(start)
    result = audit_rows(rows, fingerprint, str(path), original)
    for issue in result.issues:
        if issue.example_ids:
            implicated = set(issue.example_ids)
            issue.source_rows = [line for row, line in zip(rows, source_rows) if row.get("example_id") in implicated]
    if legacy_alias:
        result.issues.insert(0, make_issue(
            "legacy_id_column", "warning", [],
            "Input uses ticket_id instead of example_id. Only the separate review copy uses the canonical example_id header; IDs and raw bytes are unchanged.",
            source_rows=[1],
        ))
    result.issues.extend(parse_issues)
    return result
