"""Local pretrained Cardiff sentiment inference; this module never trains."""

import json
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENTS = ("Negative", "Neutral", "Positive")
SETUP_COMMAND = "python -m scripts.evaluate_sentiment --download-model"


class SentimentSetupError(RuntimeError):
    """The configured local pretrained sentiment model cannot be used."""


class SentimentInferenceError(RuntimeError):
    """A message could not receive a valid sentiment prediction."""


class SentimentPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sentiment: Literal["Positive", "Neutral", "Negative"]
    confidence: float = Field(ge=0, le=1)
    input_tokens: int = Field(ge=1)
    used_tokens: int = Field(ge=1)
    max_length: int = Field(ge=1)
    truncated: bool
    device: Literal["cpu", "cuda"]
    model_input: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_token_counts(self):
        if self.used_tokens != min(self.input_tokens, self.max_length):
            raise ValueError("Invalid sentiment-input token counts.")
        if self.truncated != (self.input_tokens > self.used_tokens):
            raise ValueError("Sentiment truncation flag does not match token counts.")
        return self


def preprocess_message(message: str) -> str:
    """Apply the model card's username/link placeholders to model input only."""
    tokens = []
    for token in message.split():
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        tokens.append(token)
    return " ".join(tokens)


def _setup_error(detail: str) -> SentimentSetupError:
    return SentimentSetupError(
        f"{detail} Restore the complete local pretrained sentiment model or run: {SETUP_COMMAND}. "
        "See docs/sentiment.md. No sentiment training is required."
    )


def _source_metadata(directory: Path) -> tuple[str, bool]:
    """Require explicit acquisition identity; do not fetch anything from the UI."""
    try:
        required = ("config.json", "tokenizer_config.json", "sentiment_source.json")
        if not directory.is_dir() or any(not (directory / name).is_file() or (directory / name).stat().st_size == 0
                                         for name in required):
            raise _setup_error("The local sentiment checkpoint is missing or incomplete.")
        safe = directory / "model.safetensors"
        binary = directory / "pytorch_model.bin"
        use_safe = safe.is_file() and safe.stat().st_size > 0
        if not use_safe and not (binary.is_file() and binary.stat().st_size > 0):
            raise _setup_error("The pretrained sentiment weights are missing.")
        source = json.loads((directory / "sentiment_source.json").read_text(encoding="utf-8"))
        revision = source["revision"]
        if source["model_id"] != MODEL_ID or not isinstance(revision, str) or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise _setup_error("The sentiment model identity/revision is invalid.")
        return revision, use_safe
    except SentimentSetupError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise _setup_error("The sentiment acquisition metadata is unreadable or invalid.") from None


def _saved_labels(config) -> dict[int, str]:
    try:
        original = {int(index): label for index, label in config.id2label.items()}
        labels = {index: label.strip().capitalize() for index, label in original.items()}
        reverse = {label.strip().capitalize(): int(index) for label, index in config.label2id.items()}
        if config.model_type != "roberta" or config.num_labels != 3:
            raise ValueError
        if set(labels) != set(range(3)) or set(labels.values()) != set(SENTIMENTS):
            raise ValueError
        if len(reverse) != 3 or reverse != {label: index for index, label in labels.items()}:
            raise ValueError
        return labels
    except (AttributeError, TypeError, ValueError):
        raise _setup_error("Saved labels must explicitly map the three sentiments with matching reverse IDs.") from None


def _input_limit(config, tokenizer) -> int:
    try:
        model_capacity = config.max_position_embeddings - config.pad_token_id - 1
        tokenizer_capacity = tokenizer.model_max_length
        if type(model_capacity) is not int or type(tokenizer_capacity) is not int:
            raise ValueError
        limit = min(512, model_capacity, tokenizer_capacity)
        if limit < 8:
            raise ValueError
        return limit
    except (AttributeError, TypeError, ValueError):
        raise _setup_error("The sentiment model/tokenizer input capacity is invalid.") from None


class SentimentClassifier:
    """Locked, batch-one inference; CPU is the conservative default."""

    def __init__(self, model, tokenizer, labels: dict[int, str], max_length: int, revision: str,
                 device: str, device_warning: str | None = None, unused_checkpoint_keys: tuple[str, ...] = ()):
        self._model = model
        self._tokenizer = tokenizer
        self._labels = dict(labels)
        self.max_length = max_length
        self.revision = revision
        self.device = device
        self.device_warning = device_warning
        self.unused_checkpoint_keys = unused_checkpoint_keys
        self._lock = RLock()
        self._tokenizer.truncation_side = "right"
        self._model.eval()

    @classmethod
    def from_checkpoint(cls, directory: Path, device: str = "cpu") -> "SentimentClassifier":
        if device not in {"cpu", "auto", "cuda"}:
            raise _setup_error("Choose cpu, auto or cuda for sentiment inference.")
        directory = Path(directory).resolve()
        revision, use_safe = _source_metadata(directory)
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

            config = AutoConfig.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
            labels = _saved_labels(config)
            tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
            limit = _input_limit(config, tokenizer)
            model, loading = AutoModelForSequenceClassification.from_pretrained(
                str(directory), config=config, local_files_only=True, trust_remote_code=False,
                use_safetensors=use_safe, weights_only=True, output_loading_info=True,
            )
            # This classification class constructs RobertaModel with add_pooling_layer=False.
            # Cardiff's original checkpoint includes that unused pooler, plus possible old buffers.
            # Every weight used by the encoder/classifier must still load with no missing keys.
            unused_allowed = {"roberta.embeddings.position_ids", "roberta.embeddings.token_type_ids",
                              "roberta.pooler.dense.weight", "roberta.pooler.dense.bias"}
            unused = set(loading.get("unexpected_keys", []))
            unexpected = unused - unused_allowed
            if unexpected or any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
                raise _setup_error("Pretrained weights do not fully match the sentiment classifier; random weights are forbidden.")
            selected, warning = "cpu", None
            if device != "cpu":
                try:
                    if torch.cuda.is_available():
                        model.to("cuda")
                        selected = "cuda"
                    else:
                        warning = "CUDA is unavailable; sentiment inference is using CPU."
                except (RuntimeError, AssertionError, OSError):
                    warning = "CUDA initialization failed; sentiment inference is using CPU."
            if selected == "cpu":
                model.to("cpu")
            return cls(model, tokenizer, labels, limit, revision, selected, warning, tuple(sorted(unused)))
        except SentimentSetupError:
            raise
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError):
            raise _setup_error("The local sentiment model/tokenizer could not be loaded and verified.") from None

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
        self.device_warning = "CUDA sentiment inference failed; this cached model has switched to CPU."
        try:
            torch.cuda.empty_cache()
        except (RuntimeError, AssertionError, OSError):
            pass

    def predict(self, message: str) -> SentimentPrediction:
        if not isinstance(message, str) or not message.strip():
            raise SentimentInferenceError("Sentiment input must be a nonblank message.")
        model_input = preprocess_message(message)
        with self._lock:
            try:
                import torch

                full = self._tokenizer(model_input, add_special_tokens=True, truncation=False, padding=False, verbose=False)
                input_tokens = len(full["input_ids"])
                encoded = self._tokenizer(model_input, add_special_tokens=True, truncation=True, max_length=self.max_length,
                                          padding=False, return_tensors="pt")
                used_tokens = int(encoded["input_ids"].shape[-1])
                try:
                    logits = self._forward(encoded)
                except (RuntimeError, AssertionError, OSError):
                    if self.device != "cuda":
                        raise
                    self._use_cpu()
                    logits = self._forward(encoded)
                if tuple(logits.shape) != (1, 3) or not bool(torch.isfinite(logits).all().item()):
                    raise SentimentInferenceError("The model returned invalid sentiment scores; this row was not analyzed.")
                scores = torch.softmax(logits, dim=-1)
                index = int(scores.argmax(dim=-1).item())
                return SentimentPrediction(
                    sentiment=self._labels[index], confidence=float(scores[0, index].item()), input_tokens=input_tokens,
                    used_tokens=used_tokens, max_length=self.max_length, truncated=input_tokens > used_tokens,
                    device=self.device, model_input=model_input,
                )
            except SentimentInferenceError:
                raise
            except (OSError, RuntimeError, AssertionError, ValueError, TypeError, KeyError, AttributeError):
                raise SentimentInferenceError("Local sentiment inference failed. Retry or select CPU; no sentiment was substituted.") from None
