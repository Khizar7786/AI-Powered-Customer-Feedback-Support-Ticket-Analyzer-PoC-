"""Strictly local, trained-checkpoint category inference without training or APIs."""

import json
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feedback_analyzer.dataset_audit import LABELS


SETUP_COMMAND = "python -m scripts.train_intent"


class IntentSetupError(RuntimeError):
    """The local trained classifier cannot be used safely."""


class IntentInferenceError(RuntimeError):
    """A message could not receive a valid model prediction."""


class IntentPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    category: str
    confidence: float = Field(ge=0, le=1)
    input_tokens: int = Field(ge=1)
    used_tokens: int = Field(ge=1)
    max_length: int = Field(ge=1)
    truncated: bool
    device: Literal["cpu", "cuda"]

    @field_validator("category")
    @classmethod
    def known_category(cls, value: str) -> str:
        if value not in LABELS:
            raise ValueError("Unknown category from classifier.")
        return value

    @model_validator(mode="after")
    def consistent_token_counts(self):
        if self.used_tokens > min(self.input_tokens, self.max_length):
            raise ValueError("Invalid classifier-input token counts.")
        if self.truncated != (self.input_tokens > self.used_tokens):
            raise ValueError("Truncation flag does not match token counts.")
        return self


def _setup_error(detail: str) -> IntentSetupError:
    return IntentSetupError(f"{detail} Train or restore the complete local checkpoint. Run: {SETUP_COMMAND}. See docs/training.md.")


def _checkpoint_metadata(directory: Path) -> tuple[dict, int]:
    """Inspect only local model artifacts; never read a dataset or fetch a model."""
    try:
        required = ("config.json", "tokenizer_config.json", "model.safetensors", "training_summary.json")
        if not directory.is_dir() or any(not (directory / name).is_file() or (directory / name).stat().st_size == 0 for name in required):
            raise _setup_error("The trained classifier checkpoint is missing or incomplete.")
        summary = json.loads((directory / "training_summary.json").read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("status") != "complete" or summary.get("mode") != "full":
            raise _setup_error("The classifier has no completed full-training record; a smoke/base model is insufficient.")
        limit = summary["config"]["max_length"]
        if type(limit) is not int or not 8 <= limit <= 512:
            raise _setup_error("The checkpoint's trained input length is invalid.")
        return summary, limit
    except IntentSetupError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise _setup_error("The checkpoint training metadata is unreadable or invalid.") from None


def _saved_labels(config) -> dict[int, str]:
    try:
        labels = {int(index): name for index, name in config.id2label.items()}
        if config.model_type != "distilbert" or config.num_labels != 6:
            raise ValueError
        if set(labels) != set(range(6)) or set(labels.values()) != set(LABELS):
            raise ValueError
        if config.label2id != {name: index for index, name in labels.items()}:
            raise ValueError
        return labels
    except (AttributeError, TypeError, ValueError):
        raise _setup_error("The saved configuration must contain matching six-category label mappings.") from None


class IntentClassifier:
    """One thread-safe model shared by UI sessions; predictions use batch size one."""

    def __init__(self, model, tokenizer, labels: dict[int, str], max_length: int, device: str,
                 device_warning: str | None = None):
        self._model = model
        self._tokenizer = tokenizer
        self._labels = dict(labels)
        self.max_length = max_length
        self.device = device
        self.device_warning = device_warning
        self._lock = RLock()
        self._tokenizer.truncation_side = "right"
        self._model.eval()

    @classmethod
    def from_checkpoint(cls, directory: Path, device: str = "auto") -> "IntentClassifier":
        if device not in {"auto", "cpu", "cuda"}:
            raise _setup_error("Choose auto, cpu or cuda for classifier inference.")
        directory = Path(directory).resolve()
        _, limit = _checkpoint_metadata(directory)
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

            config = AutoConfig.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
            labels = _saved_labels(config)
            if limit > config.max_position_embeddings:
                raise _setup_error("The trained input length exceeds the saved model capacity.")
            tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
            model, loading = AutoModelForSequenceClassification.from_pretrained(
                str(directory), config=config, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, output_loading_info=True,
            )
            if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise _setup_error("Saved weights do not fully match the classifier; uninitialized weights are not allowed.")
            selected, warning = "cpu", None
            if device != "cpu":
                try:
                    if torch.cuda.is_available():
                        model.to("cuda")
                        selected = "cuda"
                    else:
                        warning = "CUDA is unavailable; category inference is using CPU."
                except (RuntimeError, AssertionError, OSError):
                    warning = "CUDA initialization failed; category inference is using CPU."
            if selected == "cpu":
                model.to("cpu")
            return cls(model, tokenizer, labels, limit, selected, warning)
        except IntentSetupError:
            raise
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError):
            raise _setup_error("The local model/tokenizer could not be loaded and verified.") from None

    def _forward(self, encoded):
        import torch

        with torch.inference_mode():
            inputs = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            return self._model(**inputs).logits.detach().float().cpu()

    def _use_cpu(self) -> None:
        import torch

        self._model.to("cpu")
        self._model.eval()
        self.device = "cpu"
        self.device_warning = "CUDA inference failed; this cached classifier has switched to CPU."
        try:
            torch.cuda.empty_cache()
        except (RuntimeError, AssertionError, OSError):
            pass

    def predict(self, message: str) -> IntentPrediction:
        """Predict from supplied text only; never infer qualification from confidence."""
        if not isinstance(message, str) or not message.strip():
            raise IntentInferenceError("Classifier input must be a nonblank message.")
        with self._lock:
            try:
                import torch

                # This pass counts tokens only; the model receives the capped encoding below.
                full = self._tokenizer(message, add_special_tokens=True, truncation=False, padding=False, verbose=False)
                input_tokens = len(full["input_ids"])
                encoded = self._tokenizer(message, add_special_tokens=True, truncation=True, max_length=self.max_length,
                                          padding=False, return_tensors="pt")
                used_tokens = int(encoded["input_ids"].shape[-1])
                try:
                    logits = self._forward(encoded)
                except (RuntimeError, AssertionError, OSError):
                    if self.device != "cuda":
                        raise
                    self._use_cpu()
                    logits = self._forward(encoded)
                if tuple(logits.shape) != (1, 6) or not bool(torch.isfinite(logits).all().item()):
                    raise IntentInferenceError("The classifier returned invalid category scores; this row was not classified.")
                probabilities = torch.softmax(logits, dim=-1)
                index = int(probabilities.argmax(dim=-1).item())
                return IntentPrediction(
                    category=self._labels[index], confidence=float(probabilities[0, index].item()),
                    input_tokens=input_tokens, used_tokens=used_tokens, max_length=self.max_length,
                    truncated=input_tokens > used_tokens, device=self.device,
                )
            except IntentInferenceError:
                raise
            except (OSError, RuntimeError, AssertionError, ValueError, TypeError, KeyError, AttributeError):
                raise IntentInferenceError("Local category inference failed for this row. Retry or select CPU; no category was substituted.") from None
