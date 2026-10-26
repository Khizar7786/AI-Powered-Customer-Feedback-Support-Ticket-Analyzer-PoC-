"""Validated batch outcomes, including invalid source records and storage failures."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedback_analyzer.filtering import normalize_message
from feedback_analyzer.ticket_analysis import TicketAnalysis


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BatchFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["input", "reprocessing", "storage", "coordination"]
    code: str
    message: str


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    ticket_id: str | None = None
    original_message: str | None = None
    normalized_message: str | None = None
    raw_record: str | None = None
    status: Literal["processed", "ignored", "error"]
    analysis: TicketAnalysis | None = None
    failure: BatchFailure | None = None

    @model_validator(mode="after")
    def validated_status(self):
        if self.normalized_message != (normalize_message(self.original_message) if self.original_message is not None else None):
            raise ValueError("Normalized text must correspond to the original message.")
        if self.analysis:
            # Revalidate even if a caller passed an already constructed/model_copy instance.
            self.analysis = TicketAnalysis.model_validate(self.analysis.model_dump())
            if (self.analysis.ticket.ticket_id, self.analysis.ticket.message) != (self.ticket_id, self.original_message):
                raise ValueError("Analysis belongs to a different input.")
        if self.failure:
            if self.status != "error" or (self.analysis and self.failure.stage != "storage"):
                raise ValueError("Batch failures are errors; only storage errors may retain an analysis.")
        elif self.analysis is None or self.status != self.analysis.status:
            raise ValueError("Outcome requires matching validated analysis or an explicit failure.")
        return self

    @property
    def reason(self) -> str | None:
        return self.analysis.reason if self.analysis else None

    @property
    def error(self):
        return self.failure or (self.analysis.error if self.analysis else None)


class BatchRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ordinal: int = Field(ge=1)
    source_row: int = Field(ge=1)
    source_line_end: int = Field(ge=1)
    identity: str
    ticket_key: str
    input_hash: str
    config_hash: str
    outcome: Outcome
    reused: bool = False
    persisted: bool = True
    recorded_at: str = Field(default_factory=utc_now)


class BatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    source_hash: str
    config_hash: str
    rows: list[BatchRow]
    file_errors: list[str] = Field(default_factory=list)
    persistence_errors: list[str] = Field(default_factory=list)
    started_at: str
    finished_at: str

    @property
    def counts(self) -> dict[str, int]:
        return {s: sum(row.outcome.status == s for row in self.rows) for s in ("processed", "ignored", "error")}

    @property
    def fully_processed(self) -> int:
        return self.counts["processed"]

    @property
    def reused_count(self) -> int:
        return sum(row.reused for row in self.rows)
