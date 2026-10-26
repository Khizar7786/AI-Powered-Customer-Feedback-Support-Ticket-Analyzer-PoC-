"""Evaluate the frozen held-out category split once, or return its recorded result."""

import argparse
import json
from pathlib import Path

from feedback_analyzer.config import PROJECT_ROOT, load_config
from feedback_analyzer.intent_evaluation import EvaluationError, evaluate_intent


def main(argv=None):
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--model-dir", type=Path, default=config.classifier_dir)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/intent_evaluation")
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--freeze-only", action="store_true", help="Record final choices without decoding test messages or making predictions.")
    args = parser.parse_args(argv)
    try:
        record = evaluate_intent(config, args.split_dir, args.model_dir, args.output_dir,
                                 PROJECT_ROOT / "reports/intent_evaluation/registry", device=args.device, freeze_only=args.freeze_only)
    except (EvaluationError, OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation stopped: {error}")
        return 2
    if record.get("metrics") and args.output_dir.resolve() == (PROJECT_ROOT / "reports/intent_evaluation").resolve():
        (PROJECT_ROOT / "reports/intent_metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("status", "choices_sha256", "duration_seconds", "metrics", "qualification_counts", "reused_saved_evaluation", "new_predictions") if key in record}, indent=2))
    return 0 if record["status"] in {"choices_frozen", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
