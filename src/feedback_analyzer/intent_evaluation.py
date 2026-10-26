"""Frozen, one-pass held-out intent evaluation. Never trains or calls a generative API."""

from collections import Counter
from dataclasses import replace
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import time

from feedback_analyzer.dataset_audit import COLUMNS, LABELS, normalized_text
from feedback_analyzer.filtering import ignore_reason, normalize_message
from feedback_analyzer.intent import IntentClassifier, IntentPrediction
from feedback_analyzer.pipeline_identity import build_identity, digest, model_fingerprint


class EvaluationError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError):
        raise EvaluationError("Required evaluation/training metadata is missing or invalid.") from None


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_context(config, split_dir: Path, checkpoint: Path, device="cpu") -> dict:
    """Verify frozen identities without decoding or displaying any held-out messages."""
    split_dir, checkpoint = Path(split_dir), Path(checkpoint)
    manifest = _read_json(split_dir / "manifest.json")
    summary = _read_json(checkpoint / "training_summary.json")
    trained_data = _read_json(checkpoint / "training_data_manifest.json")
    try:
        if summary["status"] != "complete" or summary["mode"] != "full":
            raise EvaluationError("Evaluate the selected full-training checkpoint, not a smoke/base model.")
        if (summary["data"]["dataset_sha256"] != manifest["dataset_sha256"]
                or trained_data["dataset_sha256"] != manifest["dataset_sha256"]
                or summary["data"]["frozen_manifest_sha256"] != _hash(split_dir / "manifest.json")
                or trained_data["frozen_manifest_sha256"] != _hash(split_dir / "manifest.json")):
            raise EvaluationError("Checkpoint and frozen dataset/manifest identities do not match.")
        memberships, groups = {}, {}
        for name in ("train", "validation", "test"):
            info = manifest["splits"][name]
            if _hash(split_dir / f"{name}.csv") != info["sha256"]:
                raise EvaluationError(f"Frozen {name} bytes changed. Do not evaluate altered/unreviewed data.")
            memberships[name], groups[name] = set(info["example_ids"]), set(info["scenario_groups"])
            if len(memberships[name]) != info["count"] or set(info["labels"]) != set(LABELS):
                raise EvaluationError("Frozen membership/count/class metadata is inconsistent.")
            if name != "test" and trained_data["splits"][name]["sha256"] != info["sha256"]:
                raise EvaluationError("Development split differs from the training record.")
        if any(memberships[a] & memberships[b] or groups[a] & groups[b]
               for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise EvaluationError("Frozen IDs or scenario groups overlap across splits.")
        origin = manifest.get("review_origin", {})
        if origin.get("method") == "assistant_user_authorized" and not origin.get("authorization"):
            raise EvaluationError("Assistant review requires its recorded user delegation.")
        labels = _read_json(checkpoint / "config.json")
        if set(labels["id2label"].values()) != set(LABELS) or len(labels["id2label"]) != 6:
            raise EvaluationError("Saved classifier labels do not match the six-category contract.")
        pipeline = build_identity(replace(config, classifier_dir=checkpoint), device).metadata
        return {"evaluation_version": "intent-final-v1", "evaluation_code_sha256": _hash(Path(__file__)), "device": device,
                "dataset_sha256": manifest["dataset_sha256"], "manifest_sha256": _hash(split_dir / "manifest.json"),
                "test": manifest["splits"]["test"], "review_origin": origin,
                "checkpoint": model_fingerprint(checkpoint), "label_mapping": labels["id2label"],
                "selected_epoch": summary.get("best_epoch"), "selected_by": summary.get("best_checkpoint_metric", "validation macro-F1"),
                "validation_macro_f1": summary.get("best_validation_macro_f1"),
                "pipeline": pipeline, "classifier_input": "message only; whitespace normalized; saved max length",
                "frozen_choices_attestation": "User requested final evaluation after choices are fixed; no test-driven model/prompt tuning."}
    except (KeyError, TypeError, OSError):
        raise EvaluationError("Frozen split/checkpoint metadata is incomplete; finish reviewed preparation/training first.") from None


def freeze_choices(context: dict, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "frozen_choices.json"
    if path.exists():
        saved = _read_json(path)
        if saved.get("choices_sha256") != digest(context):
            raise EvaluationError("Model/prompt/data choices changed after freezing. Do not silently reuse or tune against the held-out set.")
        return saved
    saved = {"locked_at_utc": _now(), "choices_sha256": digest(context), "held_out_messages_decoded": False, "choices": context}
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(saved, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        return freeze_choices(context, output_dir)
    return saved


def _load_test(split_dir, context):
    data = (Path(split_dir) / "test.csv").read_bytes()
    if hashlib.sha256(data).hexdigest() != context["test"]["sha256"]:
        raise EvaluationError("Held-out data changed after choices were frozen.")
    try:
        reader = csv.reader(StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        if next(reader) != list(COLUMNS):
            raise EvaluationError("Held-out CSV requires the canonical eight-column schema.")
        rows = []
        while True:
            start_line = reader.line_num + 1
            try:
                fields = next(reader)
            except StopIteration:
                break
            if len(fields) != len(COLUMNS) or any(not value.strip() for value in fields):
                raise EvaluationError("Held-out data has malformed/blank required fields.")
            row = dict(zip(COLUMNS, fields))
            row["_source_row"] = start_line
            if row["review_status"] != "reviewed" or row["split_pool"] != "test":
                raise EvaluationError("Only reviewed, held-out test-pool rows can be evaluated.")
            if row["label"] not in LABELS or row["is_synthetic"] not in {"true", "false"}:
                raise EvaluationError("Invalid held-out label/provenance.")
            rows.append(row)
        info = context["test"]
        if ([r["example_id"] for r in rows] != info["example_ids"] or len(rows) != info["count"]
                or dict(Counter(r["label"] for r in rows)) != info["labels"]
                or set(r["scenario_group"] for r in rows) != set(info["scenario_groups"])):
            raise EvaluationError("Held-out membership/groups/counts differ from the frozen manifest.")
        if len({normalized_text(r["message"]) for r in rows}) != len(rows):
            raise EvaluationError("Normalized duplicate messages occur in the test split.")
        return rows
    except (csv.Error, UnicodeError, StopIteration):
        raise EvaluationError("Held-out CSV could not be parsed safely.") from None


def calculate_metrics(predictions):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    labels = list(LABELS)
    expected = [r["expected_category"] for r in predictions]
    actual = [r["prediction"]["category"] if r["prediction"] else "__ERROR__" for r in predictions]
    successful = sum(r["prediction"] is not None for r in predictions)
    matrix_labels = labels + (["__ERROR__"] if successful != len(predictions) else [])
    report = classification_report(expected, actual, labels=labels, output_dict=True, zero_division=0)
    return {"row_count": len(predictions), "successful_predictions": successful,
            "prediction_coverage": successful / len(predictions),
            "accuracy": float(accuracy_score(expected, actual)),
            "macro_f1": float(f1_score(expected, actual, labels=labels, average="macro", zero_division=0)),
            "per_class": {label: report[label] for label in labels},
            "confusion_matrix": confusion_matrix(expected, actual, labels=matrix_labels).tolist(),
            "matrix_labels": matrix_labels, "matrix_orientation": "rows=true labels, columns=predicted labels",
            "failure_policy": "Inference errors count as wrong and as class false negatives; never ignored or dropped."}


def qualification_errors(predictions):
    valid_ignored, nonqual_accepted, failures = [], [], []
    for row in predictions:
        prediction = row["prediction"]
        if prediction is None:
            failures.append(row)
            continue
        ignored = bool(row["rule_ignore_reason"]) or prediction["category"] == "Non-Qualifying"
        if row["expected_category"] != "Non-Qualifying" and ignored:
            valid_ignored.append(row)
        elif row["expected_category"] == "Non-Qualifying" and not ignored:
            nonqual_accepted.append(row)
    valid = sum(r["expected_category"] != "Non-Qualifying" for r in predictions)
    return {"valid_ticket_count": valid, "valid_tickets_incorrectly_ignored_count": len(valid_ignored),
            "valid_tickets_incorrectly_ignored_rate": len(valid_ignored) / valid if valid else None,
            "valid_tickets_incorrectly_ignored": valid_ignored,
            "nonqualifying_incorrectly_accepted_count": len(nonqual_accepted),
            "nonqualifying_incorrectly_accepted": nonqual_accepted,
            "inference_errors_count": len(failures), "inference_errors": failures,
            "qualification_rule": "Exact blank/greeting rules followed by classifier semantic qualification; low confidence alone is not spam."}


def _save_report(directory, summary, predictions, qualification):
    _write(directory / "evaluation.json", summary)
    _write(directory / "metrics.json", summary["metrics"])
    _write(directory / "predictions.json", predictions)
    _write(directory / "errors.json", [r for r in predictions if not r["match"]])
    _write(directory / "qualification_errors.json", qualification)
    _write(directory / "confusion_matrix.json", {key: summary["metrics"][key] for key in ("confusion_matrix", "matrix_labels", "matrix_orientation")})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = summary["metrics"]["confusion_matrix"]
    labels = summary["metrics"]["matrix_labels"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    for i, line in enumerate(matrix):
        for j, value in enumerate(line):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > max(map(max, matrix)) / 2 else "black")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    m = summary["metrics"]
    lines = ["# Final held-out intent evaluation", "", f"Status: {summary['status']}",
             f"Rows: {m['row_count']}; accuracy: {m['accuracy']:.4f}; macro-F1: {m['macro_f1']:.4f}.",
             f"Coverage: {m['prediction_coverage']:.4f}; duration: {summary['duration_seconds']:.2f} seconds.", "",
             "| Category | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label, metric in m["per_class"].items():
        lines.append(f"| {label} | {metric['precision']:.4f} | {metric['recall']:.4f} | {metric['f1-score']:.4f} | {metric['support']} |")
    lines += ["", f"Valid tickets incorrectly ignored: {qualification['valid_tickets_incorrectly_ignored_count']}/{qualification['valid_ticket_count']}.",
              f"Non-qualifying tickets incorrectly accepted: {qualification['nonqualifying_incorrectly_accepted_count']}.", "",
              summary["limitations"], "", "No model/prompt tuning, training or API calls occurred during this category evaluation. "
              "The held-out set is now evaluated; a future tuning cycle needs a new independent test set."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_intent(config, split_dir: Path, checkpoint: Path, output_dir: Path, registry_dir: Path,
                    *, device="cpu", freeze_only=False, classifier_factory=None) -> dict:
    output_dir, registry_dir = Path(output_dir), Path(registry_dir)
    context = prepare_context(config, split_dir, checkpoint, device)
    registry_dir.mkdir(parents=True, exist_ok=True)
    claim_path = registry_dir / (context["test"]["sha256"] + ".json")
    if claim_path.exists():
        claim = _read_json(claim_path)
        if claim.get("status") in {"complete", "incomplete_inference"}:
            if claim.get("choices_sha256") != digest(context):
                raise EvaluationError("This test split was already evaluated under different choices. Do not tune/re-evaluate against it; use a new independent held-out set.")
            saved = _read_json(Path(claim["output_dir"]) / "evaluation.json")
            return {**saved, "reused_saved_evaluation": True, "new_predictions": 0}
        raise EvaluationError("This held-out split already has an unfinished/failed evaluation reservation. Inspect saved predictions; no automatic second pass is allowed.")
    frozen = freeze_choices(context, output_dir)
    if freeze_only:
        return {"status": "choices_frozen", "choices_sha256": frozen["choices_sha256"], "held_out_messages_decoded": False}
    factory = classifier_factory or (lambda path, device: IntentClassifier.from_checkpoint(path, device=device))
    classifier = factory(checkpoint, device)
    claim = {"status": "started", "started_at_utc": _now(), "choices_sha256": frozen["choices_sha256"],
             "output_dir": str(output_dir.resolve()), "test_sha256": context["test"]["sha256"], "test_messages_decoded": False}
    try:
        with claim_path.open("x", encoding="utf-8") as stream:
            json.dump(claim, stream, indent=2)
    except FileExistsError:
        raise EvaluationError("Another process reserved this held-out evaluation. No predictions were made by this process.") from None
    started = time.perf_counter()
    predictions = []
    try:
        claim["test_messages_decoded"] = True
        _write(claim_path, claim)
        rows = _load_test(split_dir, context)
        with (output_dir / "prediction_journal.jsonl").open("x", encoding="utf-8") as journal:
            for row in rows:
                record = {"source_row": row["_source_row"], "example_id": row["example_id"], "message": row["message"],
                          "expected_category": row["label"], "rule_ignore_reason": ignore_reason(row["message"])}
                try:
                    prediction = IntentPrediction.model_validate(classifier.predict(normalize_message(row["message"])))
                    record.update(status="predicted", prediction=prediction.model_dump(mode="json"), match=prediction.category == row["label"])
                except Exception:
                    record.update(status="inference_error", prediction=None, match=False, error="Local category inference failed; no substituted or ignored result.")
                predictions.append(record)
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                journal.flush()
        metrics, qualification = calculate_metrics(predictions), qualification_errors(predictions)
        summary = {"status": "complete" if metrics["prediction_coverage"] == 1 else "incomplete_inference",
                   "evaluated_at_utc": _now(), "duration_seconds": time.perf_counter() - started,
                   "choices_sha256": frozen["choices_sha256"], "identity": context,
                   "metrics": metrics, "qualification_counts": {k: v for k, v in qualification.items() if k.endswith("count") or k.endswith("rate")},
                   "versions": {name: version(name) for name in ("torch", "transformers", "scikit-learn", "pydantic")},
                   "held_out_evaluated_once": True, "test_used_for_tuning": False, "training_performed": False, "api_calls": 0,
                   "limitations": "Small synthetic English corpus; review was assistant-performed under explicit user delegation where recorded, not independently human-validated. "
                   "Held out from training/model selection since freezing, but assistant-authored/reviewed scenarios are not guaranteed independent real-customer data. "
                   "These category metrics do not measure sentiment, priority, reply quality or end-to-end application accuracy."}
        _save_report(output_dir, summary, predictions, qualification)
        claim.update(status=summary["status"], completed_at_utc=_now(), prediction_count=len(predictions))
        _write(claim_path, claim)
        return summary
    except BaseException:
        claim.update(status="interrupted_or_failed", saved_prediction_count=len(predictions))
        _write(claim_path, claim)
        raise
