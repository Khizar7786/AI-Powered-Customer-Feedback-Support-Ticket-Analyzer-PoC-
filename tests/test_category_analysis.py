"""Category workflow boundaries using explicit predictions, without model weights."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from feedback_analyzer.category_analysis import CategoryRow, analyze_categories, checkpoint_signature
from feedback_analyzer.ingestion import load_ticket_csv
from feedback_analyzer.intent import IntentPrediction


def prediction(category="Bug Report", confidence=0.18, truncated=False):
    return IntentPrediction(category=category, confidence=confidence, input_tokens=200 if truncated else 6,
                            used_tokens=128 if truncated else 6, max_length=128, truncated=truncated, device="cpu")


def test_candidates_only_and_low_confidence_is_partial_with_original_preserved():
    preview = load_ticket_csv(b'ticket_id,message\nA,"  Hi,\ncheckout crashes  "\nB,Hello!\n,App crashes\n')
    calls = []

    class FakeClassifier:
        def predict(self, text):
            calls.append(text)
            return prediction()

    result = analyze_categories(preview, FakeClassifier())
    assert calls == ["Hi, checkout crashes"]
    assert result.counts == {"partial": 1, "ignored": 1, "error": 1}
    assert result.fully_processed == 0
    row = result.rows[0]
    assert (row.source_row, row.source_line_end) == (2, 3)
    assert row.original_message == "  Hi,\ncheckout crashes  "
    assert row.normalized_message == calls[0]
    assert row.prediction.confidence == 0.18
    assert row.status == "partial"
    assert "pending" in row.reason


def test_semantic_nonqualifying_is_ignored_with_explicit_reason_and_truncation():
    preview = load_ticket_csv(b'ticket_id,message\nA,"Hello, hope you are having a nice day!"\n')

    class FakeClassifier:
        def predict(self, text):
            return prediction("Non-Qualifying", confidence=0.7, truncated=True)

    row = analyze_categories(preview, FakeClassifier()).rows[0]
    assert row.status == "ignored"
    assert "classifier predicted Non-Qualifying" in row.reason
    assert row.prediction.truncated
    assert row.prediction.input_tokens == 200
    assert row.original_message == preview.rows[0].original_message


def test_failed_row_is_error_and_later_rows_still_classified():
    preview = load_ticket_csv(b'ticket_id,message\nA,App crashes\nB,Add dark mode\n')

    class FakeClassifier:
        def predict(self, text):
            if "crashes" in text:
                raise RuntimeError("explicit mock failure")
            return prediction("Feature Request")

    result = analyze_categories(preview, FakeClassifier())
    assert [row.status for row in result.rows] == ["error", "partial"]
    assert result.rows[0].prediction is None
    assert "failed" in result.rows[0].reason
    assert result.counts["ignored"] == 0
    assert result.fully_processed == 0


def test_invalid_prediction_is_an_error_not_a_category_or_spam():
    class FakeClassifier:
        def predict(self, text):
            return {"category": "invented category", "confidence": 0.9}

    result = analyze_categories(load_ticket_csv(b"ticket_id,message\nA,App crashes\n"), FakeClassifier())
    assert result.rows[0].status == "error"
    assert result.rows[0].prediction is None


def test_file_errors_stop_before_any_candidate_inference():
    preview = load_ticket_csv(b'ticket_id,message\nA,App crashes\nB,"unterminated')

    class NeverClassifier:
        def predict(self, text):
            pytest.fail("Uncertain CSV boundaries must block inference")

    with pytest.raises(ValueError, match="file-level errors"):
        analyze_categories(preview, NeverClassifier())


@pytest.mark.parametrize("status,category", [("partial", "Non-Qualifying"), ("ignored", "Bug Report"), ("error", "Bug Report")])
def test_outcome_schema_rejects_inconsistent_category_status(status, category):
    with pytest.raises(ValidationError):
        CategoryRow(source_row=2, source_line_end=2, status=status, reason="fixture", prediction=prediction(category))


def test_checkpoint_change_invalidates_signature_without_reading_weights(tmp_path, monkeypatch):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"mock")
    before = checkpoint_signature(tmp_path)
    weights.write_bytes(b"updated mock")

    def never_read(_):
        pytest.fail("Rerun cache keys must not read model weights")

    monkeypatch.setattr(Path, "read_bytes", never_read)
    assert checkpoint_signature(tmp_path) != before
