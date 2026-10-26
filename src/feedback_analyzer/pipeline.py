"""Shared batch orchestration for Streamlit and CLI; every parsed record has one outcome."""

import hashlib

from feedback_analyzer.batch_contracts import BatchFailure, BatchResult, BatchRow, Outcome, utc_now
from feedback_analyzer.generation import GenerationError, create_provider
from feedback_analyzer.ingestion import CsvPreview, load_ticket_csv
from feedback_analyzer.intent import IntentClassifier
from feedback_analyzer.pipeline_identity import PipelineIdentity, digest
from feedback_analyzer.schemas import TicketInput
from feedback_analyzer.sentiment import SentimentClassifier
from feedback_analyzer.storage import OutcomeStore, StorageError
from feedback_analyzer.ticket_analysis import analyze_ticket


class LazyPredictor:
    def __init__(self, loader):
        self.loader, self.model = loader, None

    def predict(self, text):
        if self.model is None:
            self.model = self.loader()
        return self.model.predict(text)


class BatchRuntime:
    def __init__(self, category, sentiment, provider_factory):
        self.category, self.sentiment, self.provider_factory = category, sentiment, provider_factory

    @classmethod
    def local(cls, config, device="cpu", category_loader=None, sentiment_loader=None):
        return cls(LazyPredictor(category_loader or (lambda: IntentClassifier.from_checkpoint(config.classifier_dir, device=device))),
                   LazyPredictor(sentiment_loader or (lambda: SentimentClassifier.from_checkpoint(config.sentiment_dir, device=device))),
                   lambda: create_provider(config.project_root))


def _failure(fields, stage, code, message, analysis=None):
    return Outcome(**fields, status="error", analysis=analysis, failure=BatchFailure(stage=stage, code=code, message=message))


def analyze_batch(preview: CsvPreview, runtime: BatchRuntime, store: OutcomeStore, identity: PipelineIdentity,
                  *, source_hash=None, source_name="input.csv", reprocess=False, retry_errors=False,
                  recover_incomplete=False) -> BatchResult:
    # Revalidate caller-constructed previews before any work or paid requests.
    preview = CsvPreview.model_validate(preview.model_dump())
    config_hash = identity.config_hash
    metadata = identity.model_dump()["metadata"]
    def checked_provider():
        provider = runtime.provider_factory()
        options = getattr(provider, "public_options", None)
        if options is not None and options != metadata.get("generation"):
            raise GenerationError("configuration_changed", "API configuration changed during this batch. No request was made; submit explicitly under the new configuration.")
        return provider

    source_hash = source_hash or digest(preview.model_dump())
    run_id, started = store.start_run(source_name, source_hash, config_hash, preview.file_errors)
    rows, persistence_errors = [], []
    for ordinal, source in enumerate(preview.rows, 1):
        fields = {name: getattr(source, name) for name in ("ticket_id", "original_message", "normalized_message", "raw_record")}
        invalid = source.status == "error" or bool(preview.file_errors)
        ticket_key = f"invalid:{source_hash}:{source.source_row}" if invalid else "ticket:" + source.ticket_id
        input_hash = digest({**fields, "issues": source.issues}) if invalid else hashlib.sha256(source.original_message.encode()).hexdigest()
        key = digest([ticket_key, input_hash, config_hash])
        base = dict(ordinal=ordinal, source_row=source.source_row, source_line_end=source.source_line_end,
                    identity=key, ticket_key=ticket_key, input_hash=input_hash, config_hash=config_hash)
        outcome = None
        reused = False
        try:
            reservation = store.reserve(key, ticket_key, reprocess=reprocess, retry_errors=retry_errors,
                                        recover_incomplete=recover_incomplete)
            if reservation.action == "reuse":
                outcome, reused = reservation.outcome, True
                if any(getattr(outcome, name) != value for name, value in fields.items()):
                    outcome = None  # Never display another input's cached analysis as a useful partial.
                    raise StorageError("Stored outcome does not match this input. No automatic model/API retry was made.")
            elif reservation.action == "changed":
                outcome = _failure(fields, "reprocessing", "reprocess_required",
                                   "This ticket has a different saved input/configuration. Enable reprocessing explicitly; no model/API call was made.")
            elif reservation.action == "busy":
                outcome = _failure(fields, "coordination", "in_progress",
                                   "This input has an unfinished reservation. No duplicate work was started. Recover only after the previous worker has stopped.")
            elif invalid:
                message = "; ".join(source.issues or preview.file_errors)
                outcome = _failure(fields, "input", "invalid_csv", message)
            else:
                analysis = analyze_ticket(TicketInput(ticket_id=source.ticket_id, message=source.original_message),
                                          runtime.category, runtime.sentiment, checked_provider)
                outcome = Outcome(**fields, status=analysis.status, analysis=analysis)
            row = BatchRow(**base, outcome=outcome, reused=reused)
            if reservation.action == "run":
                store.finish_row(run_id, row, reservation.owner, metadata)
            else:
                store.record_row(run_id, row)
        except StorageError as error:
            # If paid work happened but the commit failed, its claim remains. A rerun cannot silently pay twice.
            partial = outcome.analysis if outcome else None
            row = BatchRow(**base, outcome=_failure(fields, "storage", "persistence_failure", str(error), partial), persisted=False)
            persistence_errors.append(f"Source row {source.source_row}: {error}")
        rows.append(row)
    result = BatchResult(run_id=run_id, source_hash=source_hash, config_hash=config_hash, rows=rows,
                         file_errors=preview.file_errors, persistence_errors=persistence_errors, started_at=started, finished_at=utc_now())
    try:
        store.finish_run(result)
    except StorageError as error:
        result.persistence_errors.append(str(error))
    return result


def analyze_csv(data: bytes, runtime: BatchRuntime, store: OutcomeStore, identity: PipelineIdentity, **options) -> BatchResult:
    return analyze_batch(load_ticket_csv(data), runtime, store, identity,
                         source_hash=hashlib.sha256(data).hexdigest(), **options)
