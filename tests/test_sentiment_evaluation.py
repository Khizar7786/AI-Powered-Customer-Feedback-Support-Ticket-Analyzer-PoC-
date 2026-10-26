"""Sentiment evaluation fixtures are synthetic test doubles, never human gold data."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from feedback_analyzer import sentiment as sentiment_module
from feedback_analyzer.sentiment import SentimentPrediction
from feedback_analyzer.sentiment_evaluation import audit_sentiment_data, evaluate_sentiment, run_development
from scripts.evaluate_sentiment import main


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["message", "expected_sentiment"])
        writer.writerows(rows)


def fixture_rows():
    return [("Bad export", "Negative"), ("How to export", "Neutral"), ("Great export", "Positive")]


@pytest.fixture
def mock_classifier(monkeypatch):
    state = SimpleNamespace(loads=[], messages=[], failures=set(), unavailable=False)

    class FakeClassifier:
        revision = "a" * 40
        device = "cpu"
        device_warning = None

        def predict(self, message):
            state.messages.append(message)
            if message in state.failures:
                raise RuntimeError("explicit mock failure")
            label = {message: label for message, label in fixture_rows()}.get(message, "Neutral")
            count = len(message.split()) + 2
            return SentimentPrediction(sentiment=label, confidence=0.7, input_tokens=count,
                                       used_tokens=min(count, 512), max_length=512, truncated=count > 512,
                                       device="cpu", model_input=message)

    def load(path, device="cpu"):
        state.loads.append((path, device))
        if state.unavailable:
            raise RuntimeError("mock model missing")
        return FakeClassifier()

    monkeypatch.setattr(sentiment_module.SentimentClassifier, "from_checkpoint", staticmethod(load))
    return state


@pytest.mark.parametrize("exists", [False, True])
def test_pending_empty_data_never_loads_model_or_calculates_metrics(tmp_path, mock_classifier, exists):
    source = tmp_path / "sentiment_eval.csv"
    if exists:
        write_csv(source, [])
    result = evaluate_sentiment(source, tmp_path / "model", tmp_path / "report")
    assert result["status"] == "pending_reviewed_data"
    assert "metrics" not in result
    assert not mock_classifier.loads
    assert (tmp_path / "report/report.md").exists()


def test_existing_labels_need_explicit_review_confirmation(tmp_path, mock_classifier):
    source = tmp_path / "sentiment_eval.csv"
    write_csv(source, fixture_rows())
    result = evaluate_sentiment(source, tmp_path / "model", tmp_path / "report")
    assert result["status"] == "pending_review_confirmation"
    assert result["row_count"] == 3
    assert not result["review_attested"]
    assert "metrics" not in result
    assert not mock_classifier.loads


@pytest.mark.parametrize("content", [
    "message,label\nhello,Positive\n",
    "message,expected_sentiment\nhello,positive\n",
    "message,expected_sentiment\n  ,Positive\n",
    "message,expected_sentiment\nhello,Positive,extra\n",
    'message,expected_sentiment\n"unterminated,Positive\n',
    "",
])
def test_invalid_csv_blocks_model_with_readable_source_issues(tmp_path, mock_classifier, content):
    source = tmp_path / "sentiment_eval.csv"
    source.write_text(content, encoding="utf-8")
    result = evaluate_sentiment(source, tmp_path / "model", tmp_path / "report", reviewed=True)
    assert result["status"] == "invalid_data"
    assert result["issues"]
    assert result["issues"][0]["source_row"] >= 1
    assert "metrics" not in result
    assert not mock_classifier.loads


def test_invalid_utf8_is_an_issue_not_silently_replaced_text(tmp_path):
    source = tmp_path / "sentiment_eval.csv"
    source.write_bytes(b"message,expected_sentiment\n\xff,Positive\n")
    assert audit_sentiment_data(source)["issues"]


def test_multiline_source_rows_and_original_text_preserved(tmp_path):
    source = tmp_path / "sentiment_eval.csv"
    source.write_bytes('\ufeffmessage,expected_sentiment\n"  Hello,\nthank you!  ",Positive\nQuestion,Neutral\n'.encode("utf-8"))
    audit = audit_sentiment_data(source)
    assert not audit["issues"]
    assert audit["rows"][0] == {"source_row": 2, "source_line_end": 3,
                                 "message": "  Hello,\nthank you!  ", "expected_sentiment": "Positive"}
    assert audit["rows"][1]["source_row"] == 4


def test_conflicting_normalized_duplicates_mark_every_source_row(tmp_path, mock_classifier):
    source = tmp_path / "sentiment_eval.csv"
    write_csv(source, [("  GREAT export  ", "Positive"), ("great export", "Negative")])
    result = evaluate_sentiment(source, tmp_path / "model", tmp_path / "report", reviewed=True)
    assert result["status"] == "invalid_data"
    assert [issue["source_row"] for issue in result["issues"]] == [2, 3]
    assert all("conflicting labels" in issue["message"] for issue in result["issues"])
    assert not mock_classifier.loads


def test_reviewed_evaluation_requires_all_three_sentiments(tmp_path, mock_classifier):
    source = tmp_path / "sentiment_eval.csv"
    write_csv(source, fixture_rows()[:2])
    result = evaluate_sentiment(source, tmp_path / "model", tmp_path / "report", reviewed=True)
    assert result["status"] == "invalid_data"
    assert "all three" in result["message"]
    assert not mock_classifier.loads


def test_reviewed_evaluation_records_actual_metrics_counts_and_immutable_run(tmp_path, mock_classifier):
    source = tmp_path / "sentiment_eval.csv"
    write_csv(source, fixture_rows())
    report_dir = tmp_path / "report"
    result = evaluate_sentiment(source, tmp_path / "model", report_dir, reviewed=True)
    assert result["status"] == "complete"
    assert result["metrics"]["all_rows_accuracy"] == 1
    assert result["metrics"]["successful_subset"]["macro_f1"] == 1
    assert result["class_counts"] == {"Negative": 1, "Neutral": 1, "Positive": 1}
    assert result["warnings"]  # actual three rows, not claimed to be the target thirty
    assert result["model_revision"] == "a" * 40
    assert result["review_attested"] is True
    assert len(result["input_sha256"]) == 64
    run = Path(result["run_directory"])
    previous = (run / "evaluation.json").read_bytes()
    assert json.loads((run / "errors.json").read_text()) == []
    assert len(json.loads((run / "predictions.json").read_text())) == 3
    pending = evaluate_sentiment(source, tmp_path / "model", report_dir)
    assert pending["status"] == "pending_review_confirmation"
    assert "metrics" not in pending and "run_directory" not in pending
    assert (run / "evaluation.json").read_bytes() == previous
    assert json.loads((report_dir / "status.json").read_text())["status"] == pending["status"]


def test_inference_failure_is_counted_in_coverage_and_all_row_accuracy(tmp_path, mock_classifier):
    source = tmp_path / "sentiment_eval.csv"
    write_csv(source, fixture_rows())
    mock_classifier.failures.add("Great export")
    result = evaluate_sentiment(source, tmp_path / "model", tmp_path / "report", reviewed=True)
    assert result["status"] == "incomplete_inference"
    assert result["inference_error_count"] == 1
    assert result["metrics"]["all_rows_accuracy"] == pytest.approx(2 / 3)
    assert result["metrics"]["prediction_coverage"] == pytest.approx(2 / 3)
    assert result["metrics"]["successful_subset"]["accuracy"] == 1
    errors = json.loads((Path(result["run_directory"]) / "errors.json").read_text())
    assert errors[0]["status"] == "inference_error"
    assert errors[0]["source_row"] == 4
    assert errors[0]["prediction"] is None


def test_all_inference_failures_never_create_success_subset_metrics(tmp_path, mock_classifier):
    source = tmp_path / "sentiment_eval.csv"
    write_csv(source, fixture_rows())
    mock_classifier.failures.update(message for message, _ in fixture_rows())
    result = evaluate_sentiment(source, tmp_path / "model", tmp_path / "report", reviewed=True)
    assert result["status"] == "incomplete_inference"
    assert result["metrics"] == {"all_rows_accuracy": 0, "prediction_coverage": 0, "successful_subset": None}


def test_model_setup_failure_produces_no_metrics(tmp_path, mock_classifier):
    source = tmp_path / "sentiment_eval.csv"
    write_csv(source, fixture_rows())
    mock_classifier.unavailable = True
    result = evaluate_sentiment(source, tmp_path / "model", tmp_path / "report", reviewed=True)
    assert result["status"] == "model_unavailable"
    assert "metrics" not in result
    assert "--download-model" in result["message"]


def test_development_model_failure_is_reported_without_stale_success(tmp_path, mock_classifier):
    mock_classifier.unavailable = True
    result = run_development(tmp_path / "model", tmp_path / "report")
    assert result["status"] == "model_unavailable"
    assert result["metrics_calculated"] is False
    assert not mock_classifier.messages
    assert json.loads((tmp_path / "report/status.json").read_text())["status"] == "model_unavailable"


def test_development_is_unlabeled_and_assessment_mismatches_are_not_forced(tmp_path, mock_classifier):
    result = run_development(tmp_path / "model", tmp_path / "report")
    run = Path(result["run_directory"])
    development = json.loads((run / "development.json").read_text())
    assessment = json.loads((run / "assessment_examples.json").read_text())
    assert development["metrics_calculated"] is False
    assert development["reviewed_evaluation"] is False
    assert all("expected_sentiment" not in row for row in development["results"])
    assert assessment["results"][0]["prediction"]["sentiment"] == "Neutral"
    assert assessment["results"][1]["prediction"]["sentiment"] == "Neutral"
    assert assessment["results"][1]["match"] is False
    assert assessment["results"][2]["prediction"] is None
    assert assessment["results"][2]["message"] not in mock_classifier.messages
    assert len(mock_classifier.messages) == 7  # five development, two qualifying assessment fixtures


def test_cli_pending_path_does_not_require_a_model(tmp_path, mock_classifier, capsys):
    source = tmp_path / "sentiment_eval.csv"
    write_csv(source, [])
    assert main(["--input", str(source), "--output-dir", str(tmp_path / "report"), "--model-dir", str(tmp_path / "missing")]) == 0
    assert "pending_reviewed_data" in capsys.readouterr().out
    assert not mock_classifier.loads


def test_download_cli_only_calls_explicit_setup_helper(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "feedback_analyzer.sentiment_setup", SimpleNamespace(
        download_sentiment_model=lambda **kwargs: calls.append(kwargs) or {"status": "ready"},
    ))
    assert main(["--download-model", "--model-dir", str(tmp_path / "model")]) == 0
    assert calls == [{"output_dir": tmp_path / "model", "revision": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"}]
