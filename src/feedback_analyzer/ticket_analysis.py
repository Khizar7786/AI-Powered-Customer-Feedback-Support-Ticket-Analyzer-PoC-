"""One complete ticket: local authoritative models followed by one structured API request."""

from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from feedback_analyzer.filtering import ignore_reason, normalize_message
from feedback_analyzer.generation import DetailsProvider, GeneratedResult, GenerationError, validate_generated_details
from feedback_analyzer.intent import IntentPrediction
from feedback_analyzer.schemas import GenerationContext, TicketInput
from feedback_analyzer.sentiment import SentimentPrediction


class TicketError(BaseModel):
    stage: Literal["category", "sentiment", "generation"]
    code: str
    message: str


class TicketAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket: TicketInput
    status: Literal["processed", "ignored", "error"]
    category: IntentPrediction | None = None
    sentiment: SentimentPrediction | None = None
    generated: GeneratedResult | None = None
    reason: str | None = None
    error: TicketError | None = None
    reply_is_draft: Literal[True] = True

    @model_validator(mode="after")
    def complete_outcomes(self):
        if self.status == "processed":
            if not self.category or self.category.category == "Non-Qualifying" or not self.sentiment or not self.generated or self.error:
                raise ValueError("Processed requires every validated analysis field and no error.")
            validate_generated_details(self.generated.details, self.ticket.message)
        if self.status == "ignored":
            if not self.reason or self.sentiment or self.generated or self.error:
                raise ValueError("Ignored requires a reason and no subsequent analysis/error.")
            if self.category and self.category.category != "Non-Qualifying":
                raise ValueError("A qualifying category cannot be ignored.")
        if self.status == "error" and (not self.error or self.generated):
            raise ValueError("Errors require an explicit failure and no complete generated result.")
        return self


def analyze_ticket(ticket: TicketInput, category_classifier, sentiment_classifier,
                   provider_factory: Callable[[], DetailsProvider]) -> TicketAnalysis:
    reason = ignore_reason(ticket.message)
    if reason:
        return TicketAnalysis(ticket=ticket, status="ignored", reason=reason)
    category = None
    sentiment = None
    stage = "category"
    try:
        category = IntentPrediction.model_validate(category_classifier.predict(normalize_message(ticket.message)))
        if category.category == "Non-Qualifying":
            return TicketAnalysis(ticket=ticket, status="ignored", category=category, reason="Category classifier predicted Non-Qualifying.")
        stage = "sentiment"
        sentiment = SentimentPrediction.model_validate(sentiment_classifier.predict(normalize_message(ticket.message)))
        stage = "generation"
        context = GenerationContext(message=ticket.message, category=category.category, sentiment=sentiment.sentiment)
        generated = GeneratedResult.model_validate(provider_factory().generate(context))
        details = validate_generated_details(generated.details, ticket.message)
        generated = generated.model_copy(update={"details": details})
        return TicketAnalysis(ticket=ticket, status="processed", category=category, sentiment=sentiment, generated=generated)
    except GenerationError as error:
        failure = TicketError(stage=stage, code=error.code, message=str(error))
    except ValidationError:
        failure = TicketError(stage=stage, code="validation", message="Analysis output or API input limit failed validation. Original text and available local predictions are retained.")
    except Exception:
        failure = TicketError(stage=stage, code="inference_failure", message="Analysis failed at this stage. No fallback fields or reply were invented.")
    return TicketAnalysis(ticket=ticket, status="error", category=category, sentiment=sentiment, error=failure)
