"""Shared sentiment workflow using synthetic rows and explicit fake predictions."""

import pytest
from pydantic import ValidationError

from feedback_analyzer.category_analysis import analyze_categories
from feedback_analyzer.ingestion import load_ticket_csv
from feedback_analyzer.intent import IntentPrediction
from feedback_analyzer.sentiment import SentimentPrediction
from feedback_analyzer.sentiment_analysis import SentimentRow, analyze_sentiments


def category_batch():
    preview = load_ticket_csv(
        b'ticket_id,message\nA,"  Hi,\ncheckout crashes  "\nB,Add an export option\n'
        b'C,"Hello, have a lovely afternoon!"\nD,Hello!\n,App crashes\n'
    )

    class FakeIntent:
        def predict(self, text):
            return IntentPrediction(
                category="Non-Qualifying" if "afternoon" in text else "Bug Report",
                confidence=0.3, input_tokens=8, used_tokens=8, max_length=128, truncated=False, device="cpu",
            )

    return analyze_categories(preview, FakeIntent())


def prediction(sentiment="Neutral", truncated=False):
    return SentimentPrediction(sentiment=sentiment, confidence=0.6, input_tokens=700 if truncated else 8,
                               used_tokens=512 if truncated else 8, max_length=512,
                               truncated=truncated, device="cpu", model_input="fixture model input")


def test_sentiment_only_reaches_qualifying_rows_and_originals_survive():
    categories = category_batch()
    before = categories.model_dump()
    calls = []

    class FakeSentiment:
        def predict(self, text):
            calls.append(text)
            return prediction("Negative" if "crashes" in text else "Neutral")

    result = analyze_sentiments(categories, FakeSentiment())
    assert calls == ["Hi, checkout crashes", "Add an export option"]
    assert result.counts == {"partial": 2, "ignored": 2, "error": 1}
    assert result.fully_processed == 0
    assert result.rows[0].prediction.sentiment == "Negative"
    assert result.rows[1].prediction.sentiment == "Neutral"  # A feature request is not forced positive.
    assert result.rows[0].category_row.original_message == "  Hi,\ncheckout crashes  "
    assert result.rows[0].category_row.source_line_end == 3
    assert all(row.prediction is None for row in result.rows[2:])
    assert categories.model_dump() == before


def test_failure_retains_category_and_continues_without_spam_relabeling():
    categories = category_batch()

    class FakeSentiment:
        def predict(self, text):
            if "crashes" in text:
                raise RuntimeError("explicit mock failure")
            return prediction()

    result = analyze_sentiments(categories, FakeSentiment())
    first = result.rows[0]
    assert first.status == "error"
    assert first.prediction is None
    assert first.category_row.prediction.category == "Bug Report"
    assert first.category_row.original_message == categories.rows[0].original_message
    assert result.rows[1].status == "partial"
    assert result.counts["ignored"] == categories.counts["ignored"]
    assert "category is retained" in first.reason


def test_invalid_sentiment_output_is_error_not_replacement_label():
    class FakeSentiment:
        def predict(self, text):
            return {"sentiment": "Happy", "confidence": 0.8}

    result = analyze_sentiments(category_batch(), FakeSentiment())
    assert all(row.prediction is None and row.status == "error" for row in result.rows[:2])


def test_sentiment_truncation_does_not_truncate_original_text_or_category_input():
    class FakeSentiment:
        def predict(self, text):
            return prediction(truncated=True)

    categories = category_batch()
    result = analyze_sentiments(categories, FakeSentiment())
    assert result.rows[0].prediction.truncated
    assert result.rows[0].prediction.used_tokens == 512
    assert result.rows[0].category_row == categories.rows[0]
    assert not result.rows[0].category_row.prediction.truncated


@pytest.mark.parametrize("index", [2, 3, 4])
def test_schema_never_assigns_sentiment_to_ignored_or_invalid_category(index):
    with pytest.raises(ValidationError, match="qualifying"):
        SentimentRow(category_row=category_batch().rows[index], prediction=prediction(), status="partial", reason="fixture")


def test_schema_does_not_turn_failed_sentiment_into_ignored():
    with pytest.raises(ValidationError, match="failure must not cause ignoring"):
        SentimentRow(category_row=category_batch().rows[0], status="ignored", reason="fixture")
