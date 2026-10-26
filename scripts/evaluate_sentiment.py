"""Evaluate existing local sentiment weights only after explicit data review."""

import argparse
from pathlib import Path

from feedback_analyzer.config import PROJECT_ROOT, load_config
from feedback_analyzer.sentiment_evaluation import evaluate_sentiment, run_development


DEFAULT_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/sentiment_eval.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/sentiment_evaluation")
    parser.add_argument("--model-dir", type=Path, default=load_config().sentiment_dir)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--reviewed", action="store_true", help="Confirm you independently checked all message/sentiment labels before computing metrics.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--download-model", action="store_true", help="Download the pinned existing weights/tokenizer; do not train/evaluate.")
    actions.add_argument("--development", action="store_true", help="Run identified unlabeled development examples and separate published assessment checks; no accuracy metrics.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Official model revision for an explicit download.")
    args = parser.parse_args(argv)
    try:
        if args.download_model:
            from feedback_analyzer.sentiment_setup import download_sentiment_model

            result = download_sentiment_model(output_dir=args.model_dir, revision=args.revision)
            print(f"Local sentiment model ready: {args.model_dir}", flush=True)
            return 0
        if args.development:
            result = run_development(args.model_dir, args.output_dir, device=args.device)
        else:
            result = evaluate_sentiment(args.input, args.model_dir, args.output_dir, reviewed=args.reviewed, device=args.device)
        print(f"{result['status']}: {result['message']}", flush=True)
        print(f"Report: {args.output_dir / 'status.json'}", flush=True)
        if result.get("run_directory"):
            print(f"Run artifacts: {result['run_directory']}", flush=True)
        return 0 if result["status"] in {"complete", "development_complete", "pending_reviewed_data", "pending_review_confirmation"} else 2
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Sentiment check stopped ({type(error).__name__}): {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
