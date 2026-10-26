"""Category-only application workflow; no training, API calls or persistence."""

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedback_analyzer.ingestion import CsvPreview
from feedback_analyzer.intent import IntentPrediction


class CategoryPredictor(Protocol):
    def predict(self, message: str) -> IntentPrediction: ...


class CategoryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_row: int = Field(ge=1)
    source_line_end: int = Field(ge=1)
    ticket_id: str | None = None
    original_message: str | None = None
    normalized_message: str | None = None
    raw_record: str | None = None
    status: Literal["partial", "ignored", "error"]
    reason: str
    prediction: IntentPrediction | None = None

    @model_validator(mode="after")
    def valid_outcome(self):
        if self.status == "partial" and (self.prediction is None or self.prediction.category == "Non-Qualifying"):
            raise ValueError("Partial analysis requires a qualifying category prediction.")
        if self.status == "ignored" and self.prediction and self.prediction.category != "Non-Qualifying":
            raise ValueError("A qualifying prediction must not be ignored.")
        if self.status == "error" and self.prediction is not None:
            raise ValueError("Failed inference must not have a category prediction.")
        return self


class CategoryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[CategoryRow]

    @property
    def counts(self) -> dict[str, int]:
        return {status: sum(row.status == status for row in self.rows) for status in ("partial", "ignored", "error")}

    @property
    def fully_processed(self) -> int:
        # Sentiment, priority, feature and reply fields do not exist in Stage 5.
        return 0


def checkpoint_signature(directory: Path) -> tuple:
    """Invalidate cached resources/results when saved files change, without reading weights."""
    names = ("config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
             "vocab.txt", "vocab.json", "merges.txt", "special_tokens_map.json", "training_summary.json", "sentiment_source.json")
    files = []
    for name in names:
        try:
            stat = (directory / name).stat()
            files.append((name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            files.append((name, None, None))
    return (str(directory.resolve()), tuple(files))


def analyze_categories(preview: CsvPreview, classifier: CategoryPredictor) -> CategoryBatch:
    """Classify only candidates; never interpret failures/low confidence as spam."""
    if preview.file_errors:
        raise ValueError("Correct the CSV file-level errors before category analysis.")
    results = []
    for row in preview.rows:
        fields = {name: getattr(row, name) for name in (
            "source_row", "source_line_end", "ticket_id", "original_message", "normalized_message", "raw_record",
        )}
        if row.status != "candidate":
            results.append(CategoryRow(**fields, status=row.status, reason="; ".join(row.issues)))
            continue
        try:
            prediction = IntentPrediction.model_validate(classifier.predict(row.normalized_message))
        except Exception:
            # A bad row must not terminate a batch or be silently called spam.
            results.append(CategoryRow(**fields, status="error", reason=(
                "Category inference failed. No category was assigned. Retry explicitly or select CPU."
            )))
            continue
        ignored = prediction.category == "Non-Qualifying"
        results.append(CategoryRow(
            **fields, status="ignored" if ignored else "partial", prediction=prediction,
            reason="Category classifier predicted Non-Qualifying." if ignored else (
                "Category only: sentiment, priority, product/feature and reply are still pending."
            ),
        ))
    return CategoryBatch(rows=results)
