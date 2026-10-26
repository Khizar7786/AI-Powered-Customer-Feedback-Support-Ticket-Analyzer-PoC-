"""Offline Stage 4 contracts; all CSV examples here are synthetic fixtures."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from feedback_analyzer.dataset_audit import COLUMNS, LABELS
from feedback_analyzer.intent_training import (
    LABEL2ID,
    TrainingConfig,
    compute_metrics,
    inspect_training_device,
    load_training_splits,
    memory_adjusted_config,
    run_training,
    tokenize_rows,
    training_arguments,
)


def fixture_splits():
    return {
        name: [
            {
                "example_id": f"{name}_{index}",
                "message": hashlib.sha256(f"{name}_{index}".encode()).hexdigest(),
                "label": label,
                "source": "unit_test_metadata_only",
                "is_synthetic": "true",
                "scenario_group": f"group_{name}_{index}",
                "review_status": "reviewed",
                "split_pool": "development",
            }
            for index, label in enumerate(LABELS)
        ]
        for name in ("train", "validation")
    }


def freeze(directory, splits):
    """Write fixture train/validation only; no test data is needed to train."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_sha256": "a" * 64, "splits": {},
                "review_origin": {"method": "unit_test_fixture"}}
    for name, rows in splits.items():
        path = directory / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        manifest["splits"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": len(rows), "example_ids": [row["example_id"] for row in rows],
        }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_frozen_inputs_load_without_ever_opening_held_out_file(tmp_path, monkeypatch):
    expected = fixture_splits()
    freeze(tmp_path, expected)
    test_path = tmp_path / "test.csv"
    test_path.write_text("DO NOT READ: fixture poison", encoding="utf-8")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == test_path:
            pytest.fail("Stage 4 must not open held-out content")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    actual, metadata = load_training_splits(tmp_path)
    assert actual == expected
    assert set(metadata["splits"]) == {"train", "validation"}
    assert metadata["held_out_test_opened"] is False
    assert metadata["review_origin"] == {"method": "unit_test_fixture"}
    assert len(metadata["development_fingerprint"]) == 64


def test_changed_csv_is_rejected_before_training(tmp_path):
    freeze(tmp_path, fixture_splits())
    with (tmp_path / "train.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered content\n")
    with pytest.raises(ValueError, match="frozen hash"):
        load_training_splits(tmp_path)


@pytest.mark.parametrize("content,reason", [
    ("ticket_id,message\nT1,hello\n", "eight-column training schema"),
    (",".join(COLUMNS) + "\nmissing,columns\n", "Malformed train CSV record"),
])
def test_schema_and_malformed_rows_rejected_with_matching_file_hash(tmp_path, content, reason):
    manifest = freeze(tmp_path, fixture_splits())
    path = tmp_path / "train.csv"
    path.write_text(content, encoding="utf-8")
    manifest["splits"]["train"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        load_training_splits(tmp_path)


@pytest.mark.parametrize("manifest", [[], {"dataset_sha256": None}, {"dataset_sha256": "bad"},
                                      {"dataset_sha256": "a" * 64, "splits": None}])
def test_malformed_manifest_is_an_actionable_validation_error(tmp_path, manifest):
    freeze(tmp_path, fixture_splits())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|fingerprint|frozen train"):
        load_training_splits(tmp_path)


@pytest.mark.parametrize("field,value,reason", [
    ("review_status", "pending", "reviewed development"),
    ("split_pool", "test", "reviewed development"),
    ("label", "Unknown", "invalid category"),
    ("is_synthetic", "maybe", "provenance"),
    ("message", "  ", "blank required"),
])
def test_invalid_rows_rejected_even_with_matching_hash(tmp_path, field, value, reason):
    splits = fixture_splits()
    splits["train"][0][field] = value
    freeze(tmp_path, splits)
    with pytest.raises(ValueError, match=reason):
        load_training_splits(tmp_path)


def test_manifest_membership_cannot_be_changed_independently(tmp_path):
    manifest = freeze(tmp_path, fixture_splits())
    manifest["splits"]["validation"]["example_ids"][0] = "wrong_id"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="membership"):
        load_training_splits(tmp_path)


@pytest.mark.parametrize("kind,reason", [
    ("id", "Duplicate IDs"), ("message", "Normalized duplicate"),
    ("group", "scenario group crosses"),
])
def test_development_leakage_is_rejected(tmp_path, kind, reason):
    splits = fixture_splits()
    train, validation = splits["train"][0], splits["validation"][0]
    if kind == "id":
        validation["example_id"] = train["example_id"]
    elif kind == "message":
        train["message"] = "Checkout CRASHES!"
        validation["message"] = "  checkout crashes. "
    else:
        validation["scenario_group"] = train["scenario_group"]
    freeze(tmp_path, splits)
    with pytest.raises(ValueError, match=reason):
        load_training_splits(tmp_path)


def test_all_six_classes_required_in_validation(tmp_path):
    splits = fixture_splits()
    splits["validation"].pop()
    freeze(tmp_path, splits)
    with pytest.raises(ValueError, match="all six categories"):
        load_training_splits(tmp_path)


def test_tokenization_excludes_metadata_and_preserves_original_messages():
    rows = fixture_splits()["train"][:2]
    rows[0]["message"] = "Hi, checkout crashes!"
    rows[1]["message"] = "one two three four five six seven eight nine ten"
    original = [row.copy() for row in rows]
    calls = []

    def tokenizer(texts, **options):
        calls.append((list(texts), options))
        ids = [list(range(len(text.split()) + 2)) for text in texts]
        if options["truncation"]:
            ids = [tokens[:options["max_length"]] for tokens in ids]
        return {"input_ids": ids, "attention_mask": [[1] * len(tokens) for tokens in ids]}

    features, stats = tokenize_rows(rows, tokenizer, 8)
    assert all(texts == [row["message"] for row in original] for texts, _ in calls)
    assert set(features) == {"input_ids", "attention_mask", "labels"}
    assert features["labels"] == [LABEL2ID[row["label"]] for row in rows]
    assert stats["truncated_rows"] == 1
    assert max(map(len, features["input_ids"])) == 8
    assert all(options["padding"] is False for _, options in calls)
    assert rows == original


def test_macro_f1_counts_each_class_instead_of_following_majority_accuracy():
    targets = np.array([0] * 5 + [1, 2, 3, 4, 5])
    logits = np.tile(np.array([1, 0, 0, 0, 0, 0]), (len(targets), 1))
    result = compute_metrics(SimpleNamespace(predictions=(logits,), label_ids=targets))
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(1 / 9)


def test_explicit_cpu_never_queries_cuda(monkeypatch):
    import torch

    def unexpected_cuda_check():
        pytest.fail("Explicit CPU should work without querying CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_check)
    device = inspect_training_device("cpu")
    assert device["device"] == "cpu"
    assert device["fp16"] is False


def test_cuda_unavailable_has_cpu_fallback_and_explicit_cuda_error(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert inspect_training_device("auto")["device"] == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        inspect_training_device("cuda")


def test_low_vram_preserves_effective_batch_and_does_not_mutate_request():
    requested = TrainingConfig()
    adjusted, reason = memory_adjusted_config(requested, {"device": "cuda", "free_mib": 2000})
    assert (adjusted.train_batch_size, adjusted.gradient_accumulation_steps, adjusted.eval_batch_size) == (1, 8, 1)
    assert adjusted.train_batch_size * adjusted.gradient_accumulation_steps == 8
    assert requested.train_batch_size == 2
    assert reason
    with pytest.raises(ValueError, match="Unload other GPU models"):
        memory_adjusted_config(requested, {"device": "cuda", "free_mib": 1000})
    assert memory_adjusted_config(requested, {"device": "cpu"}) == (requested, None)


def test_real_trainer_arguments_select_validation_macro_f1_on_cpu(tmp_path):
    full = training_arguments(TrainingConfig(output_dir=tmp_path / "full"), {"device": "cpu", "fp16": False})
    assert full.device.type == "cpu"
    assert full.fp16 is False
    assert full.num_train_epochs == 3
    assert full.learning_rate == 2e-5
    assert full.optim.value == "adamw_torch"
    assert full.weight_decay == 0.01
    assert (full.per_device_train_batch_size, full.per_device_eval_batch_size, full.gradient_accumulation_steps) == (2, 2, 4)
    assert full.eval_strategy.value == full.save_strategy.value == "epoch"
    assert full.load_best_model_at_end and full.greater_is_better
    assert full.metric_for_best_model == "macro_f1"
    assert full.seed == full.data_seed == 42
    assert full.warmup_steps == 10
    smoke = training_arguments(TrainingConfig(smoke=True, output_dir=tmp_path / "smoke"), {"device": "cpu", "fp16": False})
    assert smoke.num_train_epochs == 1
    assert smoke.warmup_steps == 0


def test_existing_model_run_is_not_overwritten_or_loaded(tmp_path, monkeypatch):
    import feedback_analyzer.intent_training as training

    split_dir, output_dir = tmp_path / "splits", tmp_path / "existing_model"
    freeze(split_dir, fixture_splits())
    output_dir.mkdir()
    marker = output_dir / "user_model.txt"
    marker.write_text("keep me", encoding="utf-8")

    def unexpected_device(_):
        pytest.fail("Existing run must be rejected before model/device work")

    monkeypatch.setattr(training, "inspect_training_device", unexpected_device)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run_training(TrainingConfig(split_dir=split_dir, output_dir=output_dir, report_dir=tmp_path / "reports"))
    assert marker.read_text(encoding="utf-8") == "keep me"
