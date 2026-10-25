"""Freeze reviewed category data without moving held-out examples into training."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import random

from feedback_analyzer.dataset_audit import COLUMNS, LABELS, normalize_duplicate_text


VALIDATION_FRACTION = 90 / 510


class SplitPreparationError(ValueError):
    """The reviewed data cannot safely produce the requested split."""


def _validate_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise SplitPreparationError("The reviewed dataset is empty.")
    seen_ids: set[str] = set()
    seen_messages: dict[str, str] = {}
    pools_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if set(row) != set(COLUMNS):
            raise SplitPreparationError("Every reviewed row must contain exactly the eight canonical dataset columns.")
        if any(not isinstance(row[column], str) or not row[column].strip() for column in COLUMNS):
            raise SplitPreparationError("Reviewed rows require nonblank string values in every dataset column.")
        identifier = row["example_id"]
        if identifier in seen_ids:
            raise SplitPreparationError(f"Duplicate example_id: {identifier}.")
        seen_ids.add(identifier)
        if row["review_status"] != "reviewed":
            raise SplitPreparationError(f"{identifier} has not been marked reviewed; no final splits were prepared.")
        if row["label"] not in LABELS:
            raise SplitPreparationError(f"{identifier} has an unsupported category label.")
        if row["split_pool"] not in {"development", "test"}:
            raise SplitPreparationError(f"{identifier} has an unsupported split_pool.")
        if row["is_synthetic"] not in {"true", "false"}:
            raise SplitPreparationError(f"{identifier} must record is_synthetic as true or false.")
        normalized = normalize_duplicate_text(row["message"])
        if not normalized:
            raise SplitPreparationError(f"{identifier} has no message content after duplicate normalization.")
        if normalized in seen_messages:
            raise SplitPreparationError(
                f"Normalized exact duplicates {seen_messages[normalized]} and {identifier} require correction before freezing."
            )
        seen_messages[normalized] = identifier
        pools_by_group[row["scenario_group"]].add(row["split_pool"])
    if any(len(pools) > 1 for pools in pools_by_group.values()):
        raise SplitPreparationError("A scenario_group crosses development/test pools; replace conflicting test scenarios first.")


def _component_groups(
    rows: Sequence[Mapping[str, str]], related_pairs: Iterable[tuple[str, str]],
) -> list[list[dict[str, str]]]:
    by_id = {row["example_id"]: row for row in rows}
    parents = {row["scenario_group"]: row["scenario_group"] for row in rows}

    def root(group: str) -> str:
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for pair in related_pairs:
        if len(pair) != 2 or pair[0] not in by_id or pair[1] not in by_id:
            raise SplitPreparationError("Every confirmed related pair must name two existing example IDs.")
        left, right = (by_id[identifier] for identifier in pair)
        if left["split_pool"] != right["split_pool"]:
            raise SplitPreparationError("A confirmed related pair crosses pools; test content cannot be moved into training.")
        first, second = sorted((root(left["scenario_group"]), root(right["scenario_group"])))
        parents[second] = first

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[root(row["scenario_group"])].append(dict(row))
    return [sorted(grouped[key], key=lambda row: row["example_id"]) for key in sorted(grouped)]


def _choose_validation(groups: Sequence[Sequence[Mapping[str, str]]], seed: int) -> set[int]:
    """Minimize row/class count deviations using whole groups and seeded search.

    This is balancing, not a claim of an exact stratification optimum. Counts
    may differ from the target when keeping related examples together needs it.
    """
    vectors = [tuple(Counter(row["label"] for row in group)[label] for label in LABELS) for group in groups]
    totals = tuple(sum(vector[index] for vector in vectors) for index in range(len(LABELS)))
    group_coverage = [sum(vector[index] > 0 for vector in vectors) for index in range(len(LABELS))]
    if any(total < 2 or coverage < 2 for total, coverage in zip(totals, group_coverage)):
        raise SplitPreparationError("Each category needs at least two independent development groups for train/validation coverage.")
    targets = tuple(total * VALIDATION_FRACTION for total in totals)
    target_size = sum(targets)

    def score(counts: Sequence[int]) -> float:
        missing = sum(count == 0 or count == total for count, total in zip(counts, totals))
        per_class = sum((count - target) ** 2 / max(target, 1) for count, target in zip(counts, targets))
        return missing * 1_000_000 + per_class + (sum(counts) - target_size) ** 2 / max(target_size, 1)

    rng = random.Random(seed)
    best_selection: set[int] | None = None
    best_score = float("inf")
    indices = list(range(len(groups)))
    for attempt in range(48):
        selected = set() if attempt == 0 else {index for index in indices if rng.random() < VALIDATION_FRACTION}
        counts = [sum(vectors[index][label] for index in selected) for label in range(len(LABELS))]
        current_score = score(counts)
        # Whole-group add/remove descent; random ties make restarts useful.
        for _ in range(len(groups) * 2):
            rng.shuffle(indices)
            candidate = None
            candidate_counts = None
            candidate_score = current_score
            for index in indices:
                direction = -1 if index in selected else 1
                trial = [count + direction * value for count, value in zip(counts, vectors[index])]
                trial_score = score(trial)
                if trial_score < candidate_score - 1e-10:
                    candidate, candidate_counts, candidate_score = index, trial, trial_score
            if candidate is None:
                break
            selected.symmetric_difference_update({candidate})
            counts = candidate_counts
            current_score = candidate_score
        if current_score < best_score - 1e-10:
            best_selection, best_score = selected.copy(), current_score
        if best_score < 1e-10:
            break
    if best_selection is None:
        raise SplitPreparationError("No group-preserving validation assignment could be prepared.")
    final_counts = [sum(vectors[index][label] for index in best_selection) for label in range(len(LABELS))]
    if any(count == 0 or count == total for count, total in zip(final_counts, totals)):
        raise SplitPreparationError("No six-category train/validation assignment was found without splitting related groups.")
    return best_selection


def prepare_grouped_splits(
    rows: Sequence[Mapping[str, str]], seed: int = 42, related_pairs: Iterable[tuple[str, str]] = (),
) -> dict[str, list[dict[str, str]]]:
    """Return reviewed train/validation/test rows, retaining all audit metadata.

    The test pool is preserved in full. Scenario groups and explicitly confirmed
    related pairs stay together; their row/class counts take priority over an
    exact 420/90 development subdivision. Input rows are never modified.
    """
    _validate_rows(rows)
    groups = _component_groups(rows, related_pairs)
    development = [group for group in groups if group[0]["split_pool"] == "development"]
    test = [dict(row) for row in rows if row["split_pool"] == "test"]
    if {row["label"] for row in test} != set(LABELS):
        raise SplitPreparationError("The held-out test pool must contain all six category labels.")
    selected = _choose_validation(development, seed)
    result = {"train": [], "validation": [], "test": test}
    for index, group in enumerate(development):
        result["validation" if index in selected else "train"].extend(group)
    return {name: sorted(split, key=lambda row: row["example_id"]) for name, split in result.items()}


def dataset_fingerprint(rows: Sequence[Mapping[str, str]]) -> str:
    """Hash canonical contents, including labels/provenance/review/pool metadata."""
    canonical = [{column: row[column] for column in COLUMNS} for row in sorted(rows, key=lambda row: row["example_id"])]
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_records(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Only message text is model input; category labels are prediction targets."""
    if any(row["review_status"] != "reviewed" for row in rows):
        raise SplitPreparationError("Only reviewed rows may be converted to model training records.")
    return [{"text": row["message"], "label": row["label"]} for row in rows]
