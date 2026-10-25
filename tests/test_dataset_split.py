"""Synthetic fixtures verify pool isolation and group-aware split behavior."""

from copy import deepcopy

import pytest

from feedback_analyzer.dataset_audit import LABELS
from feedback_analyzer.dataset_split import (
    SplitPreparationError, dataset_fingerprint, model_records, prepare_grouped_splits,
)


def make_rows(development_per_label=4, test_per_label=1):
    rows = []
    for pool, count in (("development", development_per_label), ("test", test_per_label)):
        for category, label in enumerate(LABELS):
            for number in range(count):
                identifier = f"{pool}_{category}_{number}"
                rows.append({
                    "example_id": identifier,
                    "message": f"Distinct {pool} scenario number {number} for category {category}.",
                    "label": label, "source": "authored_unit_fixture", "is_synthetic": "true",
                    "scenario_group": identifier, "review_status": "reviewed", "split_pool": pool,
                })
    return rows


def split_ids(splits):
    return {name: {row["example_id"] for row in rows} for name, rows in splits.items()}


def test_balanced_singleton_groups_reach_full_targets_without_pool_migration():
    rows = make_rows(85, 15)
    before = deepcopy(rows)
    splits = prepare_grouped_splits(rows)
    assert {name: len(split) for name, split in splits.items()} == {"train": 420, "validation": 90, "test": 90}
    for name, per_label in (("train", 70), ("validation", 15), ("test", 15)):
        assert {label: sum(row["label"] == label for row in splits[name]) for label in LABELS} == {
            label: per_label for label in LABELS
        }
    ids = split_ids(splits)
    assert not (ids["train"] & ids["validation"] | ids["train"] & ids["test"] | ids["validation"] & ids["test"])
    assert ids["test"] == {row["example_id"] for row in rows if row["split_pool"] == "test"}
    assert all(row["split_pool"] == "development" for name in ("train", "validation") for row in splits[name])
    assert rows == before


def test_reproducible_assignment_does_not_depend_on_input_order():
    rows = make_rows(8, 2)
    first = prepare_grouped_splits(rows, seed=42)
    second = prepare_grouped_splits(list(reversed(rows)), seed=42)
    assert first == second


def test_scenario_groups_and_confirmed_related_pairs_cannot_cross_train_validation():
    rows = make_rows(6, 1)
    # Contrasting labels can share one real scenario family.
    rows[0]["scenario_group"] = rows[6]["scenario_group"] = "shared_mixed_label_family"
    linked = (rows[12]["example_id"], rows[18]["example_id"])
    splits = prepare_grouped_splits(rows, related_pairs=[linked])
    assignment = {row["example_id"]: name for name, split in splits.items() for row in split}
    assert assignment[rows[0]["example_id"]] == assignment[rows[6]["example_id"]]
    assert assignment[linked[0]] == assignment[linked[1]]
    groups = {name: {row["scenario_group"] for row in split} for name, split in splits.items()}
    assert not groups["train"] & groups["validation"]
    for split in splits.values():
        assert {row["label"] for row in split} == set(LABELS)


def test_row_targets_yield_to_related_groups_without_losing_examples():
    rows = make_rows(6, 1)
    for row in rows:
        if row["split_pool"] == "development":
            number = int(row["example_id"].rsplit("_", 1)[1])
            row["scenario_group"] = f"{row['label']}_{'small' if number < 2 else 'large'}"
    splits = prepare_grouped_splits(rows)
    assert len(splits["validation"]) == 12  # Whole two-row families, rather than roughly six rows.
    assert len(splits["train"]) == 24
    assert sum(map(len, splits.values())) == len(rows)
    assert {row["example_id"] for split in splits.values() for row in split} == {
        row["example_id"] for row in rows
    }


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda rows: rows[0].update(review_status="unreviewed"), "not been marked reviewed"),
        (lambda rows: rows[0].update(label="Unknown"), "unsupported category"),
        (lambda rows: rows[0].update(is_synthetic="maybe"), "is_synthetic"),
        (lambda rows: rows[0].update(source=""), "nonblank"),
        (lambda rows: rows[0].update(split_pool="train"), "unsupported split_pool"),
        (lambda rows: rows[1].update(example_id=rows[0]["example_id"]), "Duplicate example_id"),
        (lambda rows: rows[-1].update(scenario_group=rows[0]["scenario_group"]), "crosses development/test"),
        (lambda rows: rows[-1].update(message=rows[0]["message"].upper() + "!"), "Normalized exact duplicates"),
    ],
)
def test_unsafe_reviewed_data_is_rejected_before_splitting(change, match):
    rows = make_rows()
    change(rows)
    with pytest.raises(SplitPreparationError, match=match):
        prepare_grouped_splits(rows)


def test_confirmed_related_pair_across_pools_cannot_reassign_test_to_training():
    rows = make_rows()
    with pytest.raises(SplitPreparationError, match="crosses pools"):
        prepare_grouped_splits(rows, related_pairs=[(rows[0]["example_id"], rows[-1]["example_id"])])


def test_infeasible_category_coverage_does_not_split_a_scenario_group():
    rows = make_rows()
    for row in rows:
        if row["split_pool"] == "development" and row["label"] == LABELS[0]:
            row["scenario_group"] = "only_group_for_this_category"
    with pytest.raises(SplitPreparationError, match="two independent development groups"):
        prepare_grouped_splits(rows)


def test_model_input_projection_omits_ids_and_all_provenance_metadata():
    rows = make_rows()
    assert model_records(rows[:1]) == [{"text": rows[0]["message"], "label": rows[0]["label"]}]
    rows[0]["review_status"] = "unreviewed"
    with pytest.raises(SplitPreparationError, match="Only reviewed"):
        model_records(rows)


def test_fingerprint_ignores_row_order_but_captures_a_reviewed_label_correction():
    rows = make_rows()
    original = dataset_fingerprint(rows)
    assert dataset_fingerprint(list(reversed(rows))) == original
    rows[0]["label"] = LABELS[1]
    assert dataset_fingerprint(rows) != original
