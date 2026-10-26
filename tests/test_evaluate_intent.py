"""One-pass evaluation guards and measured metrics on explicit temporary fixtures only."""

from collections import Counter
import csv
import hashlib
from io import StringIO
import json
from types import SimpleNamespace

import pytest

from feedback_analyzer import intent_evaluation as evaluation
from feedback_analyzer.config import load_config
from feedback_analyzer.dataset_audit import COLUMNS, LABELS
from feedback_analyzer.intent import IntentPrediction
from feedback_analyzer.intent_evaluation import EvaluationError, calculate_metrics, evaluate_intent, qualification_errors
from scripts import evaluate_intent as cli


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evaluation_setup(tmp_path):
    (tmp_path / ".env").write_text("LLM_PROVIDER=openai\nLLM_MODEL=unit-mock-model\n")
    config = load_config(tmp_path, environ={})
    split_dir = tmp_path / "data/splits"
    split_dir.mkdir(parents=True)
    model_dir = config.classifier_dir
    model_dir.mkdir(parents=True)
    manifest = {"dataset_sha256": "a" * 64,
                "review_origin": {"method": "assistant_user_authorized", "authorization": "Explicit unit fixture only", "human_review_performed": False}, "splits": {}}
    rowsets = {}
    for name in ("train", "validation", "test"):
        rows = [{"example_id": f"{name}-{i}", "message": f"{name} message only {i}", "label": label,
                 "source": "explicit unit fixture", "is_synthetic": "true", "scenario_group": f"{name}-group-{i}",
                 "review_status": "reviewed", "split_pool": "test" if name == "test" else "development"}
                for i, label in enumerate(LABELS)]
        rowsets[name] = rows

    def sync():
        for name, rows in rowsets.items():
            stream = StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            path = split_dir / (name + ".csv")
            path.write_bytes(stream.getvalue().encode("utf-8"))
            manifest["splits"][name] = {"count": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
                                        "sha256": sha(path), "example_ids": [r["example_id"] for r in rows],
                                        "scenario_groups": sorted({r["scenario_group"] for r in rows})}
        write(split_dir / "manifest.json", manifest)
        training = {"dataset_sha256": manifest["dataset_sha256"], "frozen_manifest_sha256": sha(split_dir / "manifest.json"),
                    "splits": {name: manifest["splits"][name] for name in ("train", "validation")}}
        write(model_dir / "training_data_manifest.json", training)
        write(model_dir / "training_summary.json", {"status": "complete", "mode": "full", "data": training,
                                                   "best_epoch": 3, "best_validation_macro_f1": .9})

    sync()
    write(model_dir / "config.json", {"id2label": {str(i): label for i, label in enumerate(LABELS)}})
    (model_dir / "model.safetensors").write_bytes(b"explicit fake weights; factory must be mocked")
    received, loads = [], []

    class Classifier:
        def predict(self, message):
            received.append(message)
            row = next(r for r in rowsets["test"] if r["message"] == message)
            return IntentPrediction(category=row["label"], confidence=.5, input_tokens=8, used_tokens=8,
                                    max_length=128, truncated=False, device="cpu")

    def factory(*args):
        loads.append(True)
        return Classifier()

    return SimpleNamespace(root=tmp_path, config=config, split_dir=split_dir, model_dir=model_dir,
                           output=tmp_path / "reports/intent_evaluation", registry=tmp_path / "reports/intent_evaluation/registry",
                           manifest=manifest, rowsets=rowsets, sync=sync, factory=factory, received=received, loads=loads)


def run(s, **kwargs):
    return evaluate_intent(s.config, s.split_dir, s.model_dir, s.output, s.registry,
                           classifier_factory=s.factory, **kwargs)


def test_freeze_does_not_decode_test_or_load_predictor(evaluation_setup, monkeypatch):
    s = evaluation_setup
    monkeypatch.setattr(evaluation, "_load_test", lambda *a: pytest.fail("Freeze must not decode test messages"))
    record = run(s, freeze_only=True)
    assert record["status"] == "choices_frozen" and not s.loads and not s.received
    saved = json.loads((s.output / "frozen_choices.json").read_text())
    assert saved["held_out_messages_decoded"] is False


def test_one_pass_persists_metrics_and_repeated_command_never_predicts_again(evaluation_setup):
    s = evaluation_setup
    manifest_before = sha(s.split_dir / "manifest.json")
    weights_before = sha(s.model_dir / "model.safetensors")
    record = run(s)
    assert record["metrics"]["accuracy"] == record["metrics"]["macro_f1"] == 1
    assert len(s.received) == 6 and s.received == [r["message"] for r in s.rowsets["test"]]
    assert record["test_used_for_tuning"] is False and record["api_calls"] == 0
    assert (s.output / "confusion_matrix.png").is_file()
    assert len((s.output / "prediction_journal.jsonl").read_text().splitlines()) == 6
    again = run(s)
    assert again["reused_saved_evaluation"] and again["new_predictions"] == 0
    assert len(s.received) == 6 and len(s.loads) == 1
    assert sha(s.split_dir / "manifest.json") == manifest_before and sha(s.model_dir / "model.safetensors") == weights_before


def test_alternate_output_or_changed_model_cannot_bypass_test_consumption(evaluation_setup):
    s = evaluation_setup
    run(s)
    other = evaluate_intent(s.config, s.split_dir, s.model_dir, s.root / "other-output", s.registry, classifier_factory=s.factory)
    assert other["reused_saved_evaluation"] and len(s.received) == 6
    (s.model_dir / "model.safetensors").write_bytes(b"changed fake model")
    with pytest.raises(EvaluationError, match="already evaluated under different choices"):
        run(s)
    assert len(s.received) == 6


@pytest.mark.parametrize("change", ["test_bytes", "development_bytes", "training_identity", "unreviewed", "wrong_pool", "overlap"])
def test_invalid_prerequisites_never_produce_predictions_or_metrics(evaluation_setup, change):
    s = evaluation_setup
    if change.endswith("bytes"):
        path = s.split_dir / ("test.csv" if change == "test_bytes" else "train.csv")
        path.write_bytes(path.read_bytes() + b"tampered")
    elif change == "training_identity":
        summary = json.loads((s.model_dir / "training_summary.json").read_text())
        summary["data"]["dataset_sha256"] = "b" * 64
        write(s.model_dir / "training_summary.json", summary)
    elif change == "overlap":
        s.rowsets["test"][0]["scenario_group"] = s.rowsets["train"][0]["scenario_group"]
        s.sync()
    else:
        s.rowsets["test"][0]["review_status" if change == "unreviewed" else "split_pool"] = "unreviewed" if change == "unreviewed" else "development"
        s.sync()
    with pytest.raises(EvaluationError):
        run(s)
    assert not s.received and not (s.output / "metrics.json").exists()


def test_choices_changed_after_freeze_refuse_before_test_read(evaluation_setup):
    s = evaluation_setup
    run(s, freeze_only=True)
    (s.model_dir / "model.safetensors").write_bytes(b"changed after freeze")
    with pytest.raises(EvaluationError, match="changed after freezing"):
        run(s)
    assert not s.loads and not s.received


def test_interruption_retains_journal_and_refuses_automatic_second_pass(evaluation_setup):
    s = evaluation_setup
    normal = s.factory

    def factory(*args):
        model = normal(*args)
        original = model.predict

        def predict(message):
            if len(s.received) == 2:
                raise KeyboardInterrupt
            return original(message)

        model.predict = predict
        return model

    s.factory = factory
    with pytest.raises(KeyboardInterrupt):
        run(s)
    assert len((s.output / "prediction_journal.jsonl").read_text().splitlines()) == 2
    with pytest.raises(EvaluationError, match="unfinished/failed"):
        run(s)
    assert len(s.received) == 2 and not (s.output / "metrics.json").exists()


def test_metrics_include_failures_and_qualification_mistakes():
    def record(label, actual):
        return {"expected_category": label, "prediction": {"category": actual} if actual else None, "rule_ignore_reason": None}

    rows = [record(label, label) for label in LABELS]
    rows[0]["prediction"] = None
    metrics = calculate_metrics(rows)
    assert metrics["accuracy"] == pytest.approx(5 / 6) and metrics["macro_f1"] == pytest.approx(5 / 6)
    assert len(metrics["confusion_matrix"]) == 7 and sum(map(sum, metrics["confusion_matrix"])) == 6
    assert qualification_errors(rows)["inference_errors_count"] == 1
    rows = [record("Bug Report", "Non-Qualifying"), record("Non-Qualifying", "Feature Request")]
    errors = qualification_errors(rows)
    assert errors["valid_tickets_incorrectly_ignored_count"] == errors["nonqualifying_incorrectly_accepted_count"] == 1


def test_cli_freezes_then_evaluates_and_reuses_saved_result(evaluation_setup, monkeypatch, capsys):
    s = evaluation_setup
    monkeypatch.setattr(cli, "PROJECT_ROOT", s.root)
    monkeypatch.setattr(cli, "load_config", lambda: s.config)
    monkeypatch.setattr(evaluation.IntentClassifier, "from_checkpoint", lambda *args, **kwargs: s.factory())
    assert cli.main(["--freeze-only"]) == 0
    assert not s.received
    capsys.readouterr()
    assert cli.main([]) == 0
    assert json.loads(capsys.readouterr().out)["metrics"]["row_count"] == 6
    assert cli.main([]) == 0 and len(s.received) == 6
    assert (s.root / "reports/intent_metrics.json").is_file()
