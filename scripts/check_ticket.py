"""Check Stage 7 configuration, or explicitly run one live sample ticket."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

from feedback_analyzer.config import PROJECT_ROOT, load_config
from feedback_analyzer.generation import GenerationError, create_provider, load_generation_settings
from feedback_analyzer.ingestion import load_ticket_csv
from feedback_analyzer.intent import IntentClassifier
from feedback_analyzer.schemas import TicketInput
from feedback_analyzer.sentiment import SentimentClassifier
from feedback_analyzer.ticket_analysis import analyze_ticket


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make one real structured request after local qualification.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/sample_tickets.csv")
    parser.add_argument("--ticket-id", default="T001")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/single_ticket/live_check.json")
    args = parser.parse_args(argv)
    record = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "provider": "openai",
              "held_out_intent_test_opened": False, "reply_sent": False, "live_api_verified": False}
    try:
        settings = load_generation_settings()
    except GenerationError as error:
        record.update(status="pending_configuration", error_code=error.code, message=str(error))
    else:
        record.update(status="configuration_present_unverified", model=settings.model,
                      message="Configuration is present; no API request has been made.")
        if args.live:
            try:
                preview = load_ticket_csv(args.input.read_bytes())
                candidates = [row for row in preview.rows if row.ticket_id == args.ticket_id and row.status == "candidate"]
                if preview.file_errors or len(candidates) != 1:
                    raise ValueError("Choose exactly one valid candidate from the input sample.")
                row = candidates[0]
                config = load_config()
                category = IntentClassifier.from_checkpoint(config.classifier_dir, device="cpu")
                sentiment = SentimentClassifier.from_checkpoint(config.sentiment_dir, device="cpu")
                result = analyze_ticket(TicketInput(ticket_id=row.ticket_id, message=row.original_message), category, sentiment, create_provider)
                record.update(status=result.status, verification_kind="real_local_models_and_openai", result=result.model_dump(mode="json"),
                              live_api_verified=result.status == "processed", sdk_version=version("openai"),
                              message="Actual local model and OpenAI result; no reply was sent.")
            except Exception:
                record.update(status="local_check_error", message="Live check could not complete. Check local models and valid sample input. No live success is claimed.")
    # A configuration-only check must not overwrite a previously successful live result.
    target = args.output if args.live else args.output.with_name("configuration_check.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: value for key, value in record.items() if key != "result"}, indent=2), flush=True)
    if record.get("live_api_verified"):
        print("Actual reply DRAFT: " + record["result"]["generated"]["details"]["suggested_reply"], flush=True)
    return 0 if record["status"] in {"configuration_present_unverified", "processed", "ignored"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
