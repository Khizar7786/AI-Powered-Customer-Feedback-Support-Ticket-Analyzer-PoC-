"""Single-ticket orchestration with explicit local-model/provider doubles."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from feedback_analyzer.generation import GeneratedResult, GenerationError, GenerationReceipt
from feedback_analyzer.intent import IntentPrediction
from feedback_analyzer.schemas import GeneratedDetails, TicketInput
from feedback_analyzer.sentiment import SentimentPrediction
from feedback_analyzer.ticket_analysis import TicketAnalysis, analyze_ticket


class FakeCategory:
    def __init__(self, category="Bug Report", fail=False):
        self.category, self.fail, self.messages = category, fail, []

    def predict(self, message):
        self.messages.append(message)
        if self.fail:
            raise RuntimeError("sensitive-model-debug-data")
        return IntentPrediction(category=self.category, confidence=0.2, input_tokens=15,
                                used_tokens=15, max_length=128, truncated=False, device="cpu")


class FakeSentiment:
    def __init__(self, fail=False):
        self.fail, self.messages = fail, []

    def predict(self, message):
        self.messages.append(message)
        if self.fail:
            raise RuntimeError("sensitive-sentiment-debug-data")
        return SentimentPrediction(sentiment="Negative", confidence=0.8, input_tokens=15, used_tokens=15,
                                   max_length=512, truncated=False, device="cpu", model_input=message)


def fake_result(product=None):
    return GeneratedResult(details=GeneratedDetails(priority="Medium", product_feature=product,
        suggested_reply="Thanks for reporting this. Could you share your browser version?"),
        receipt=GenerationReceipt(model="mock-model", response_id="mock", attempts=1,
                                  duration_seconds=0.1, completed_at_utc="test-only"))


def test_complete_ticket_preserves_local_authority_original_text_and_one_call():
    ticket = TicketInput(ticket_id=" A1 ", message="  Hi,\ncheckout crashes  ")
    category, sentiment, contexts = FakeCategory(), FakeSentiment(), []
    provider = SimpleNamespace(generate=lambda context: contexts.append(context) or fake_result("checkout"))
    result = analyze_ticket(ticket, category, sentiment, lambda: provider)
    assert result.status == "processed" and result.reply_is_draft
    assert result.ticket.ticket_id == "A1"
    assert result.ticket.message == "  Hi,\ncheckout crashes  "
    assert category.messages == sentiment.messages == ["Hi, checkout crashes"]
    assert contexts[0].message == ticket.message
    assert result.category.category == "Bug Report" and result.sentiment.sentiment == "Negative"
    assert len(contexts) == 1


@pytest.mark.parametrize("message", ["", "  \n ", "Hi!"])
def test_conservative_ignores_never_load_predict_or_construct_provider(message):
    def forbidden():
        pytest.fail("No provider for ignored messages")

    result = analyze_ticket(TicketInput(ticket_id="A", message=message), None, None, forbidden)
    assert result.status == "ignored" and result.category is None and result.generated is None


def test_semantic_nonqualifying_never_calls_sentiment_or_provider():
    sentiment = FakeSentiment()
    result = analyze_ticket(TicketInput(ticket_id="A", message="Long social greeting"), FakeCategory("Non-Qualifying"),
                            sentiment, lambda: pytest.fail("No paid call for Non-Qualifying"))
    assert result.status == "ignored" and not sentiment.messages


@pytest.mark.parametrize("stage", ["category", "sentiment", "generation"])
def test_errors_remain_errors_and_keep_available_local_predictions(stage):
    category = FakeCategory(fail=stage == "category")
    sentiment = FakeSentiment(fail=stage == "sentiment")
    calls = []

    def provider():
        calls.append(True)
        raise GenerationError("authentication", "Check OpenAI credentials.")

    result = analyze_ticket(TicketInput(ticket_id="A", message="Checkout crashes"), category, sentiment, provider)
    assert result.status == "error" and result.error.stage == stage
    assert result.generated is None
    assert (result.category is not None) == (stage != "category")
    assert (result.sentiment is not None) == (stage == "generation")
    assert len(calls) == int(stage == "generation")
    assert "sensitive" not in result.model_dump_json()


def test_provider_cannot_add_override_fields_or_ungrounded_product():
    category, sentiment = FakeCategory(), FakeSentiment()
    provider = SimpleNamespace(generate=lambda ctx: fake_result("Invented Product"))
    result = analyze_ticket(TicketInput(ticket_id="A", message="App crashes"), category, sentiment, lambda: provider)
    assert result.status == "error" and result.error.stage == "generation"
    assert result.category.category == "Bug Report" and result.sentiment.sentiment == "Negative"


def test_api_input_limit_stops_before_constructing_provider_and_preserves_full_message():
    message = "App crashes " * 1100
    result = analyze_ticket(TicketInput(ticket_id="A", message=message), FakeCategory(), FakeSentiment(),
                            lambda: pytest.fail("Oversize message must not reach API"))
    assert result.status == "error" and result.error.code == "validation"
    assert result.ticket.message == message


def test_invalid_input_and_false_processed_or_ignored_status_are_rejected():
    with pytest.raises(ValidationError):
        TicketInput(ticket_id=" ", message="App crashes")
    ticket = TicketInput(ticket_id="A", message="App crashes")
    with pytest.raises(ValidationError, match="Processed requires"):
        TicketAnalysis(ticket=ticket, status="processed")
    with pytest.raises(ValidationError, match="qualifying category"):
        TicketAnalysis(ticket=ticket, status="ignored", reason="fake", category=FakeCategory().predict(ticket.message))
