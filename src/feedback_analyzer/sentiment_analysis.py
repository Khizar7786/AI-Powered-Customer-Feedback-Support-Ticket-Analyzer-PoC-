"""Add sentiment to qualifying category rows while preserving partial outcomes."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from feedback_analyzer.category_analysis import CategoryBatch, CategoryRow
from feedback_analyzer.sentiment import SentimentPrediction


class SentimentPredictor(Protocol):
    def predict(self, message: str) -> SentimentPrediction: ...


class SentimentRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category_row: CategoryRow
    prediction: SentimentPrediction | None = None
    status: Literal["partial", "ignored", "error"]
    reason: str

    @model_validator(mode="after")
    def qualifying_only(self):
        if self.prediction is not None and self.category_row.status != "partial":
            raise ValueError("Sentiment is permitted only for qualifying category rows.")
        if self.status == "partial" and (self.category_row.status != "partial" or self.prediction is None):
            raise ValueError("Partial sentiment results require both category and sentiment.")
        if self.status == "ignored" and self.category_row.status != "ignored":
            raise ValueError("Sentiment failure must not cause ignoring.")
        if self.status == "error" and self.prediction is not None:
            raise ValueError("Failed sentiment must not have a substituted prediction.")
        return self


class SentimentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[SentimentRow]

    @property
    def counts(self) -> dict[str, int]:
        return {status: sum(row.status == status for row in self.rows) for status in ("partial", "ignored", "error")}

    @property
    def fully_processed(self) -> int:
        return 0


def analyze_sentiments(categories: CategoryBatch, classifier: SentimentPredictor) -> SentimentBatch:
    """No category decision is changed by a sentiment prediction or failure."""
    rows = []
    for category_row in categories.rows:
        if category_row.status != "partial":
            rows.append(SentimentRow(category_row=category_row, status=category_row.status, reason=category_row.reason))
            continue
        try:
            prediction = SentimentPrediction.model_validate(classifier.predict(category_row.normalized_message))
        except Exception:
            rows.append(SentimentRow(category_row=category_row, status="error", reason=(
                "Sentiment inference failed; category is retained. Retry explicitly or select CPU."
            )))
            continue
        rows.append(SentimentRow(category_row=category_row, prediction=prediction, status="partial", reason=(
            "Category and sentiment only: priority, product/feature and reply are still pending."
        )))
    return SentimentBatch(rows=rows)
