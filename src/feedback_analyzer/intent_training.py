"""Train one local category classifier from frozen development splits only."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from feedback_analyzer.dataset_audit import COLUMNS, LABELS, normalized_text


MODEL_ID = "distilbert/distilbert-base-uncased"
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
ROOT = Path(__file__).resolve().parents[2]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_dir: Path = ROOT / "data/splits"
    output_dir: Path = ROOT / "models/ticket_classifier"
    report_dir: Path = ROOT / "reports/intent_training"
    cache_dir: Path = ROOT / ".cache/huggingface/hub"
    revision: str = "main"
    local_files_only: bool = False
    device: Literal["auto", "cuda", "cpu"] = "auto"
    smoke: bool = False
    epochs: int = Field(default=3, ge=1, le=5)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_length: int = Field(default=128, ge=8, le=512)
    train_batch_size: int = Field(default=2, ge=1)
    eval_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    seed: int = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def load_training_splits(directory: Path) -> tuple[dict[str, list[dict]], dict]:
    """Read train/validation and manifest metadata. Never open the test CSV."""
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Frozen split manifest is missing/invalid. Complete scripts.prepare_data first.") from error
    if not isinstance(manifest, dict):
        raise ValueError("Frozen split manifest must be a JSON object.")
    fingerprint = manifest.get("dataset_sha256", "")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
        raise ValueError("Manifest dataset fingerprint must be a SHA-256 digest.")
    splits = {}
    metadata = {
        "dataset_sha256": fingerprint, "frozen_manifest_sha256": sha256(manifest_path),
        "review_origin": manifest.get("review_origin", {}), "splits": {},
        "held_out_test_opened": False,
    }
    for name in ("train", "validation"):
        path = directory / (name + ".csv")
        try:
            data = path.read_bytes()
            info = manifest["splits"][name]
            digest = hashlib.sha256(data).hexdigest()
            if digest != info["sha256"]:
                raise ValueError(f"{name} CSV differs from its frozen hash; re-freeze a reviewed dataset version.")
            reader = csv.reader(StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
            if next(reader) != list(COLUMNS):
                raise ValueError(f"{name} requires the canonical eight-column training schema.")
            rows = []
            for fields in reader:
                if len(fields) != len(COLUMNS):
                    raise ValueError(f"Malformed {name} CSV record.")
                row = dict(zip(COLUMNS, fields))
                if any(not value.strip() for value in row.values()):
                    raise ValueError(f"{name} contains blank required fields.")
                if row["review_status"] != "reviewed" or row["split_pool"] != "development":
                    raise ValueError(f"{name} must contain reviewed development-pool rows only.")
                if row["label"] not in LABELS or row["is_synthetic"] not in {"true", "false"}:
                    raise ValueError(f"{name} has invalid category/provenance values.")
                rows.append(row)
            if len(rows) != info["count"] or [row["example_id"] for row in rows] != info["example_ids"]:
                raise ValueError(f"{name} membership differs from the frozen manifest.")
            if set(row["label"] for row in rows) != set(LABELS):
                raise ValueError(f"{name} must cover all six categories.")
            splits[name] = rows
            metadata["splits"][name] = {
                "sha256": digest, "count": len(rows), "example_ids": [row["example_id"] for row in rows],
                "labels": dict(Counter(row["label"] for row in rows)),
            }
        except (OSError, UnicodeError, csv.Error, StopIteration, KeyError, TypeError) as error:
            raise ValueError(f"Cannot read valid frozen {name} data; run scripts.prepare_data first.") from error
    combined = splits["train"] + splits["validation"]
    if len({row["example_id"] for row in combined}) != len(combined):
        raise ValueError("Duplicate IDs occur within/across training and validation.")
    if len({normalized_text(row["message"]) for row in combined}) != len(combined):
        raise ValueError("Normalized duplicate messages occur within/across development splits.")
    if {row["scenario_group"] for row in splits["train"]} & {row["scenario_group"] for row in splits["validation"]}:
        raise ValueError("A scenario group crosses training and validation.")
    metadata["development_fingerprint"] = hashlib.sha256(json.dumps(
        {name: metadata["splits"][name]["sha256"] for name in splits}, sort_keys=True,
    ).encode()).hexdigest()
    return splits, metadata


def smoke_subset(rows: list[dict], per_class: int) -> list[dict]:
    selected = []
    for label in LABELS:
        candidates = [row for row in rows if row["label"] == label]
        selected.extend(candidates[:per_class])
    return selected


def inspect_training_device(requested: str) -> dict:
    import torch

    if requested == "cpu":
        return {"device": "cpu", "fp16": False, "reason": "CPU explicitly selected."}
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise ValueError("CUDA was requested but is unavailable; use --device cpu or repair CUDA.")
        return {"device": "cpu", "fp16": False, "reason": "CUDA unavailable; CPU fallback."}
    free, total = torch.cuda.mem_get_info()
    capability = torch.cuda.get_device_capability()
    fp16 = False
    with torch.inference_mode():
        matrix = torch.ones((2, 2), device="cuda")
        if not bool(torch.all(matrix @ matrix == 2).item()):
            raise ValueError("CUDA arithmetic check failed; do not train on this device.")
        if capability[0] >= 7:
            half = matrix.half()
            fp16 = bool(torch.all(half @ half == 2).item())
            del half
        del matrix
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return {
        "device": "cuda", "fp16": fp16, "name": torch.cuda.get_device_name(),
        "capability": list(capability), "free_mib": round(free / 1024**2, 2),
        "total_mib": round(total / 1024**2, 2), "cuda_runtime": torch.version.cuda,
        "float32_probe_passed": True, "float16_probe_passed": fp16,
    }


def memory_adjusted_config(config: TrainingConfig, device: dict) -> tuple[TrainingConfig, str | None]:
    if device["device"] != "cuda":
        return config, None
    if device["free_mib"] < 1500:
        raise ValueError("Less than 1500 MiB free CUDA memory. Unload other GPU models or use --device cpu.")
    if device["free_mib"] < 2500 and config.train_batch_size == 2 and config.gradient_accumulation_steps == 4:
        return config.model_copy(update={"train_batch_size": 1, "eval_batch_size": 1, "gradient_accumulation_steps": 8}), \
            "Low free VRAM: batch 1, accumulation 8, evaluation batch 1; effective batch remains 8."
    return config, None


def tokenize_rows(rows: list[dict], tokenizer, max_length: int) -> tuple[dict, dict]:
    """Only message is tokenized; labels are targets. Metadata is never a feature."""
    texts = [row["message"] for row in rows]
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False, padding=False)["input_ids"]]
    encoded = dict(tokenizer(texts, truncation=True, max_length=max_length, padding=False))
    encoded["labels"] = [LABEL2ID[row["label"]] for row in rows]
    return encoded, {"rows": len(rows), "truncated_rows": sum(length > max_length for length in lengths),
                     "max_original_tokens": max(lengths, default=0), "max_length": max_length}


def compute_metrics(prediction) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    predicted = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(prediction.label_ids, predicted)),
            "macro_f1": float(f1_score(prediction.label_ids, predicted, labels=list(range(6)), average="macro", zero_division=0))}


def training_arguments(config: TrainingConfig, device: dict):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"), num_train_epochs=1 if config.smoke else config.epochs,
        learning_rate=config.learning_rate, optim="adamw_torch", weight_decay=config.weight_decay,
        per_device_train_batch_size=config.train_batch_size, per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, fp16=device["fp16"], bf16=False,
        use_cpu=device["device"] == "cpu", eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=2, lr_scheduler_type="linear", warmup_steps=0 if config.smoke else 10,
        seed=config.seed, data_seed=config.seed, dataloader_num_workers=0,
        dataloader_pin_memory=device["device"] == "cuda", report_to=[], disable_tqdm=True,
        logging_strategy="steps", logging_steps=1 if config.smoke else 10,
    )


def run_training(config: TrainingConfig) -> dict:
    """Train from the base encoder; a smoke checkpoint is never reused for full training."""
    started = time.perf_counter()
    splits, split_info = load_training_splits(config.split_dir)
    for directory in (config.output_dir, config.report_dir):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Refusing to overwrite nonempty run directory: {directory}. Select a new output/report path.")
    device = inspect_training_device(config.device)
    config, adjustment = memory_adjusted_config(config, device)
    if config.smoke:
        splits = {"train": smoke_subset(splits["train"], 2), "validation": smoke_subset(splits["validation"], 1)}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    versions = {name: version(name) for name in ("torch", "transformers", "datasets", "accelerate", "scikit-learn", "pydantic")}
    versions["python"] = platform.python_version()
    record = {"status": "running", "mode": "smoke" if config.smoke else "full", "model_id": MODEL_ID,
              "config": config.model_dump(mode="json"), "device": device, "memory_adjustment": adjustment,
              "versions": versions, "data": split_info, "actual_rows": {name: len(rows) for name, rows in splits.items()},
              "used_example_ids": {name: [row["example_id"] for row in rows] for name, rows in splits.items()},
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "held_out_test_evaluated": False}
    write_json(config.report_dir / "run_status.json", record)
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.metrics import classification_report, confusion_matrix
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, set_seed

        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=config.revision, cache_dir=str(config.cache_dir),
                                                  local_files_only=config.local_files_only)
        datasets = {}
        truncation = {}
        for name, rows in splits.items():
            encoded, truncation[name] = tokenize_rows(rows, tokenizer, config.max_length)
            datasets[name] = Dataset.from_dict(encoded)
        if device["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=config.revision, cache_dir=str(config.cache_dir), local_files_only=config.local_files_only,
            num_labels=6, label2id=LABEL2ID, id2label=ID2LABEL, use_safetensors=True,
        )
        head_before = hashlib.sha256(model.classifier.weight.detach().cpu().numpy().tobytes()).hexdigest()
        args = training_arguments(config, device)
        record.update(resolved_model_revision=model.config._commit_hash, truncation=truncation,
                      optimizer="AdamW (torch)", effective_batch_size=config.train_batch_size * config.gradient_accumulation_steps,
                      warmup_steps=args.warmup_steps, actual_epochs=args.num_train_epochs,
                      dynamic_padding=True, best_checkpoint_metric="validation macro-F1",
                      trainer_arguments=args.to_dict())
        write_json(config.report_dir / "training_config.json", record)
        trainer = Trainer(
            model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
            processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if device["fp16"] else None),
            compute_metrics=compute_metrics,
        )
        training_started = time.perf_counter()
        print(f"Training {record['mode']}: {record['actual_rows']}; device={device['device']}; fp16={device['fp16']}; effective batch={record['effective_batch_size']}", flush=True)
        trained = trainer.train()
        if device["device"] == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        predictions = trainer.predict(datasets["validation"], metric_key_prefix="validation")
        predicted = np.argmax(predictions.predictions, axis=-1)
        targets = predictions.label_ids
        validation = {
            "dataset_role": "validation", "mode": record["mode"], "metrics": predictions.metrics,
            "classification_report": classification_report(targets, predicted, labels=list(range(6)), target_names=list(LABELS),
                                                           output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(targets, predicted, labels=list(range(6))).tolist(), "label_order": list(LABELS),
            "held_out_test_evaluated": False,
        }
        trainer.save_model(str(config.output_dir))
        tokenizer.model_max_length = config.max_length
        tokenizer.save_pretrained(str(config.output_dir))
        best_step = getattr(trainer.state, "best_global_step", None)
        if best_step is None and trainer.state.best_model_checkpoint:
            best_step = int(Path(trainer.state.best_model_checkpoint).name.rsplit("-", 1)[-1])
        best_log = next((entry for entry in trainer.state.log_history if entry.get("step") == best_step and "eval_macro_f1" in entry), {})
        record.update(status="complete", training_seconds=training_seconds, pipeline_seconds=time.perf_counter() - started,
                      train_metrics=trained.metrics, best_validation_macro_f1=trainer.state.best_metric,
                      best_checkpoint=trainer.state.best_model_checkpoint, best_global_step=best_step, best_epoch=best_log.get("epoch"),
                      optimizer_steps=trainer.state.global_step,
                      classifier_head_changed=head_before != hashlib.sha256(model.classifier.weight.detach().cpu().numpy().tobytes()).hexdigest())
        if device["device"] == "cuda":
            record["gpu_peak_memory"] = {"allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                                         "reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 2)}
        write_json(config.report_dir / "validation_metrics.json", validation)
        write_json(config.report_dir / "training_history.json", trainer.state.log_history)
        write_json(config.report_dir / "training_summary.json", record)
        write_json(config.report_dir / "run_status.json", record)
        write_json(config.output_dir / "training_summary.json", record)
        write_json(config.output_dir / "training_data_manifest.json", split_info)
        write_json(config.output_dir / "label_mappings.json", {"label2id": LABEL2ID, "id2label": ID2LABEL})
        with (config.report_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["example_id", "true_label", "predicted_label"])
            writer.writerows((row["example_id"], row["label"], ID2LABEL[int(pred)]) for row, pred in zip(splits["validation"], predicted))
        print(json.dumps({"status": "complete", "mode": record["mode"], "training_seconds": training_seconds,
                          "best_validation_macro_f1": record["best_validation_macro_f1"], "best_epoch": record["best_epoch"]}), flush=True)
        return record
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, pipeline_seconds=time.perf_counter() - started)
        write_json(config.report_dir / "run_status.json", record)
        raise


def verify_checkpoint(directory: Path) -> dict:
    """CPU-only fresh-process reload/forward check; never read an evaluation CSV."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    summary = json.loads((directory / "training_summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Checkpoint lacks a completed training record.")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(directory), local_files_only=True, use_safetensors=True)
    if model.config.label2id != LABEL2ID or model.config.id2label != ID2LABEL:
        raise ValueError("Checkpoint label mappings do not match the six categories.")
    model.eval()
    inputs = tokenizer("Please help me update my account settings.", return_tensors="pt", truncation=True,
                       max_length=summary["config"]["max_length"])
    with torch.inference_mode():
        logits = model(**inputs).logits
    result = {"load_success": True, "device": "cpu", "mode": summary["mode"], "num_labels": model.config.num_labels,
              "forward_shape": list(logits.shape), "logits_finite": bool(torch.isfinite(logits).all().item()),
              "label_mappings_match": True, "held_out_test_opened": False}
    if result["forward_shape"] != [1, 6] or not result["logits_finite"]:
        raise ValueError("Reloaded checkpoint forward check failed.")
    return result
