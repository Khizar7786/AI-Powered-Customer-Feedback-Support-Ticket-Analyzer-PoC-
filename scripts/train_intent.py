"""Train DistilBERT on frozen train/validation splits, or verify a saved model."""

import argparse
from pathlib import Path

from feedback_analyzer.intent_training import ROOT, TrainingConfig, run_training, verify_checkpoint, write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One epoch on 12 train/6 validation rows; separate output and no final metrics.")
    parser.add_argument("--verify-only", action="store_true", help="Load saved model/tokenizer on CPU and run a single forward check.")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/huggingface/hub")
    parser.add_argument("--revision", default="main", help="Base-model commit/revision; the resolved commit is saved.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        config = TrainingConfig(
            split_dir=args.split_dir, output_dir=args.output_dir or ROOT / ("models/intent_smoke" if args.smoke else "models/ticket_classifier"),
            report_dir=args.report_dir or ROOT / ("reports/intent_smoke" if args.smoke else "reports/intent_training"),
            cache_dir=args.cache_dir, revision=args.revision, local_files_only=args.local_files_only,
            device=args.device, smoke=args.smoke, epochs=args.epochs, train_batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        )
        if args.verify_only:
            result = verify_checkpoint(config.output_dir)
            write_json(config.report_dir / "reload_check.json", result)
            print(result, flush=True)
        else:
            run_training(config)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Training/verification stopped ({type(error).__name__}): {error}", flush=True)
        if "out of memory" in str(error).lower():
            print("Unload other GPU models. Retry in NEW output/report directories with --batch-size 1 --eval-batch-size 1 --gradient-accumulation 8, or --device cpu.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
