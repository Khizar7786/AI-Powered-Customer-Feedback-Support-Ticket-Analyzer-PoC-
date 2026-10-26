"""Reviewed sentiment evaluation, separate from unlabeled development checks."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform
import time
import unicodedata
from uuid import uuid4

from feedback_analyzer.config import SENTIMENT_MODEL_ID


SENTIMENTS = ("Negative", "Neutral", "Positive")
EVALUATION_COLUMNS = ("message", "expected_sentiment")
csv.field_size_limit(2**31 - 1)
DOMAIN_LIMITATION = (
    "This existing English Twitter/TweetEval model may interpret support-ticket wording differently. "
    "Polite feature requests, sarcasm, mixed feelings and domain-specific complaints need reviewed checks. "
    "Confidence scores are not calibrated guarantees. No sentiment model was trained."
)


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _normalized_message(message: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", message).casefold().split())


def audit_sentiment_data(path: Path) -> dict:
    """Read only the explicit two-column sentiment file, retaining source lines."""
    result = {"input_path": str(path.resolve()), "input_sha256": None, "rows": [], "issues": [],
              "class_counts": {label: 0 for label in SENTIMENTS}, "warnings": []}
    if not path.is_file():
        result["missing"] = True
        return result
    try:
        data = path.read_bytes()
        result["input_sha256"] = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8-sig")
    except (OSError, UnicodeError):
        result["issues"].append({"source_row": 1, "source_line_end": 1,
                                 "message": "Cannot read UTF-8 sentiment evaluation CSV."})
        return result
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        if next(reader) != list(EVALUATION_COLUMNS):
            raise ValueError
    except (StopIteration, csv.Error, ValueError):
        result["issues"].append({"source_row": 1, "source_line_end": max(reader.line_num, 1),
                                 "message": "Expected exactly the header message,expected_sentiment."})
        return result
    while True:
        start = reader.line_num + 1
        try:
            fields = next(reader)
        except StopIteration:
            break
        except csv.Error:
            result["issues"].append({"source_row": start, "source_line_end": max(reader.line_num, start),
                                     "message": "Malformed quoted CSV record; later record boundaries are uncertain."})
            break
        end = reader.line_num
        if len(fields) != 2:
            result["issues"].append({"source_row": start, "source_line_end": end,
                                     "message": "Each record must have exactly two fields."})
            continue
        message, expected = fields
        if not message.strip() or expected not in SENTIMENTS:
            result["issues"].append({"source_row": start, "source_line_end": end,
                                     "message": "Message must be nonblank and expected_sentiment must be Negative, Neutral or Positive."})
            continue
        result["rows"].append({"source_row": start, "source_line_end": end,
                               "message": message, "expected_sentiment": expected})
    groups = defaultdict(list)
    for row in result["rows"]:
        groups[_normalized_message(row["message"])].append(row)
    for rows in groups.values():
        if len(rows) > 1:
            conflict = len({row["expected_sentiment"] for row in rows}) > 1
            for row in rows:
                result["issues"].append({"source_row": row["source_row"], "source_line_end": row["source_line_end"],
                                         "message": "Normalized duplicate message" + (" with conflicting labels." if conflict else ".")})
    counts = Counter(row["expected_sentiment"] for row in result["rows"])
    result["class_counts"] = {label: counts[label] for label in SENTIMENTS}
    if len(result["rows"]) != 30 or any(counts[label] != 10 for label in SENTIMENTS):
        result["warnings"].append("Learning target is about 30 independently authored/reviewed messages, around 10 per sentiment; actual counts are reported.")
    return result


def _new_run(output_dir: Path, kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / "runs" / f"{stamp}_{kind}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _versions() -> dict:
    return {"python": platform.python_version(), **{name: version(name) for name in ("torch", "transformers", "scikit-learn")}}


def _status(output_dir: Path, status: str, message: str, **details) -> dict:
    record = {"status": status, "message": message, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_id": SENTIMENT_MODEL_ID, "held_out_intent_test_opened": False,
              "sentiment_training_performed": False, **details}
    _write_json(output_dir / "status.json", record)
    return record


def _report_text(record: dict) -> str:
    lines = ["# Sentiment evaluation", "", f"Status: {record['status']}", "", record["message"], ""]
    if "row_count" in record:
        lines += [f"Actual messages: {record['row_count']}", f"Class counts: {record['class_counts']}", ""]
    for issue in record.get("issues", []):
        lines.append(f"- Source lines {issue['source_row']}–{issue['source_line_end']}: {issue['message']}")
    for warning in record.get("warnings", []):
        lines.append(f"- {warning}")
    if record.get("metrics"):
        lines += ["", f"Accuracy counting inference failures as wrong: {record['metrics']['all_rows_accuracy']:.6f}",
                  f"Prediction coverage: {record['metrics']['prediction_coverage']:.6f}"]
        if record["metrics"]["successful_subset"]:
            lines.append(f"Successful-subset macro-F1: {record['metrics']['successful_subset']['macro_f1']:.6f}")
        else:
            lines.append("No successful predictions; no subset macro-F1 is available.")
    lines += ["", DOMAIN_LIMITATION, ""]
    return "\n".join(lines)


def evaluate_sentiment(input_path: Path, model_dir: Path, output_dir: Path, *, reviewed: bool = False,
                       device: str = "cpu") -> dict:
    """Require caller review attestation; never turn provisional examples into gold labels."""
    started = time.perf_counter()
    audit = audit_sentiment_data(input_path)
    details = {key: audit[key] for key in ("input_path", "input_sha256", "class_counts", "issues", "warnings")}
    details.update(row_count=len(audit["rows"]), review_attested=reviewed)
    if audit.get("missing") or not audit["rows"] and not audit["issues"]:
        record = _status(output_dir, "pending_reviewed_data", "Supply your independently written and labeled messages in data/sentiment_eval.csv, then run with --reviewed after checking every label. No evaluation metrics were calculated.", **details)
    elif audit["issues"]:
        record = _status(output_dir, "invalid_data", "Correct the source-row issues before sentiment evaluation. No metrics were calculated.", **details)
    elif not reviewed:
        record = _status(output_dir, "pending_review_confirmation", "Data is present. Review each message/label, then explicitly run --reviewed. No model was loaded and no metrics were calculated.", **details)
    elif any(count == 0 for count in audit["class_counts"].values()):
        record = _status(output_dir, "invalid_data", "Reviewed evaluation must include all three sentiments before macro-F1 evaluation. Actual class counts are shown; no metrics were calculated.", **details)
    else:
        from feedback_analyzer.sentiment import SentimentClassifier, SentimentPrediction

        directory = _new_run(output_dir, "reviewed")
        details["run_directory"] = str(directory.resolve())
        try:
            classifier = SentimentClassifier.from_checkpoint(model_dir, device=device)
        except (RuntimeError, ValueError, OSError):
            record = _status(output_dir, "model_unavailable", "The local sentiment checkpoint could not be verified. Run python -m scripts.evaluate_sentiment --download-model or check --model-dir. No metrics were calculated.", **details)
        else:
            predictions = []
            errors = []
            for row in audit["rows"]:
                record_row = dict(row)
                try:
                    prediction = SentimentPrediction.model_validate(classifier.predict(row["message"]))
                    record_row.update(prediction=prediction.model_dump(mode="json"), status="predicted",
                                      match=prediction.sentiment == row["expected_sentiment"])
                    if not record_row["match"]:
                        errors.append(record_row)
                except Exception:
                    record_row.update(prediction=None, status="inference_error", match=False,
                                      error="Local sentiment inference failed; no label was substituted.")
                    errors.append(record_row)
                predictions.append(record_row)
            successful = [row for row in predictions if row["prediction"] is not None]
            metrics = {"all_rows_accuracy": sum(row["match"] for row in predictions) / len(predictions),
                       "prediction_coverage": len(successful) / len(predictions), "successful_subset": None}
            if successful:
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

                expected = [row["expected_sentiment"] for row in successful]
                actual = [row["prediction"]["sentiment"] for row in successful]
                metrics = {
                    "all_rows_accuracy": sum(row["match"] for row in predictions) / len(predictions),
                    "prediction_coverage": len(successful) / len(predictions),
                    "successful_subset": {
                        "row_count": len(successful), "accuracy": float(accuracy_score(expected, actual)),
                        "macro_f1": float(f1_score(expected, actual, labels=list(SENTIMENTS), average="macro", zero_division=0)),
                        "classification_report": classification_report(expected, actual, labels=list(SENTIMENTS), output_dict=True, zero_division=0),
                        "confusion_matrix": confusion_matrix(expected, actual, labels=list(SENTIMENTS)).tolist(),
                        "label_order": list(SENTIMENTS),
                    },
                }
            complete = len(successful) == len(predictions)
            record = _status(
                output_dir, "complete" if complete else "incomplete_inference",
                "Reviewed evaluation finished using existing sentiment weights." if complete else "Some reviewed rows could not be predicted. Failed rows count as wrong in all-row accuracy; subset metrics are explicitly separated.",
                **details, model_revision=classifier.revision, device=classifier.device, device_warning=classifier.device_warning,
                unused_checkpoint_keys=list(getattr(classifier, "unused_checkpoint_keys", ())),
                successful_rows=len(successful), inference_error_count=len(predictions) - len(successful),
                mismatch_count=sum(row["status"] == "predicted" and not row["match"] for row in predictions),
                metrics=metrics, duration_seconds=time.perf_counter() - started, versions=_versions(),
                review_attestation="Caller explicitly passed --reviewed; no authorship or human review inferred from file existence.",
                domain_limitations=DOMAIN_LIMITATION,
            )
            _write_json(directory / "predictions.json", predictions)
            _write_json(directory / "errors.json", errors)
            _write_json(directory / "evaluation.json", record)
            (directory / "report.md").write_text(_report_text(record), encoding="utf-8")
        if record["status"] == "model_unavailable":
            _write_json(directory / "evaluation.json", record)
    (output_dir / "report.md").write_text(_report_text(record), encoding="utf-8")
    return record


def run_development(model_dir: Path, output_dir: Path, *, device: str = "cpu") -> dict:
    """Unlabeled assistant development fixtures plus separate published sample checks."""
    from feedback_analyzer.sentiment import SentimentClassifier, SentimentPrediction

    started = time.perf_counter()
    directory = _new_run(output_dir, "development")
    try:
        classifier = SentimentClassifier.from_checkpoint(model_dir, device=device)
    except (RuntimeError, ValueError, OSError):
        result = _status(output_dir, "model_unavailable", "The local sentiment checkpoint could not be verified. Run python -m scripts.evaluate_sentiment --download-model or check --model-dir. No development predictions or metrics were created.",
                         run_directory=str(directory.resolve()), reviewed_evaluation=False, metrics_calculated=False)
        _write_json(directory / "development.json", result)
        (output_dir / "report.md").write_text(_report_text(result), encoding="utf-8")
        return result
    messages = [
        "Thank you, the new search makes finding my saved reports much easier.",
        "Where can I find the list of supported export formats?",
        "The app lost my changes again and I am frustrated.",
        "@support Please explain the charge listed at https://example.com/invoice.",
        "The report export keeps failing. " * 180,
    ]
    development = []
    for number, message in enumerate(messages, 1):
        row = {"development_id": f"DEV-SENT-{number}", "message": message}
        try:
            prediction = SentimentPrediction.model_validate(classifier.predict(message))
            row.update(status="predicted", prediction=prediction.model_dump(mode="json"))
        except Exception:
            row.update(status="inference_error", prediction=None,
                       error="Local sentiment inference failed; no label was substituted.")
        development.append(row)
    development_record = {"status": "development_check", "fixture_origin": "assistant-authored development examples; no expected labels assigned",
                          "model_id": SENTIMENT_MODEL_ID, "model_revision": classifier.revision,
                          "device": classifier.device, "device_warning": classifier.device_warning,
                          "unused_checkpoint_keys": list(getattr(classifier, "unused_checkpoint_keys", ())),
                          "results": development, "metrics_calculated": False, "reviewed_evaluation": False,
                          "held_out_intent_test_opened": False, "sentiment_training_performed": False,
                          "versions": _versions(), "duration_seconds": time.perf_counter() - started,
                          "domain_limitations": DOMAIN_LIMITATION}
    _write_json(directory / "development.json", development_record)
    examples = [
        {"ticket_id": "T001", "message": "The payment page crashes every time I click 'Checkout' on Chrome. Please fix this ASAP!", "expected_sentiment": "Negative"},
        {"ticket_id": "T002", "message": "Hi team, it would be awesome if you could add a dark mode to the mobile app.", "expected_sentiment": "Positive"},
    ]
    assessment = []
    for row in examples:
        item = {**row, "expectation_origin": "published assessment fixture, not user-labeled evaluation data"}
        try:
            prediction = SentimentPrediction.model_validate(classifier.predict(row["message"]))
            item.update(status="predicted", prediction=prediction.model_dump(mode="json"),
                        match=prediction.sentiment == row["expected_sentiment"])
        except Exception:
            item.update(status="inference_error", prediction=None, match=False,
                        error="Local sentiment inference failed; no label was substituted.")
        assessment.append(item)
    assessment.append({"ticket_id": "T003", "message": "Hello, hope you are having a nice day!", "status": "ignored",
                       "reason": "Published assessment marks this greeting Non-Qualifying; sentiment intentionally not run.", "prediction": None})
    assessment_record = {"status": "assessment_fixture_check", "model_id": SENTIMENT_MODEL_ID,
                         "model_revision": classifier.revision, "results": assessment,
                         "metrics_calculated": False, "reviewed_evaluation": False,
                         "held_out_intent_test_opened": False, "sentiment_training_performed": False,
                         "domain_limitations": DOMAIN_LIMITATION}
    _write_json(directory / "assessment_examples.json", assessment_record)
    failures = sum(row["status"] == "inference_error" for row in development + assessment)
    result = _status(output_dir, "incomplete_development" if failures else "development_complete",
                     "Unlabeled development inference and separate published assessment checks finished. Reviewed evaluation data is still required; no accuracy claim was made." if not failures else "Some development/assessment predictions failed. Failures are recorded with no substitute labels; no accuracy claim was made.",
                     run_directory=str(directory.resolve()), model_revision=classifier.revision,
                     reviewed_evaluation=False, metrics_calculated=False, inference_error_count=failures)
    (output_dir / "report.md").write_text(_report_text(result), encoding="utf-8")
    return result
