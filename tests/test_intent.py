"""Explicit model/tokenizer doubles; no downloaded model, network or GPU needed."""

import json
from types import SimpleNamespace

import pytest
import torch
import transformers

from feedback_analyzer.dataset_audit import LABELS
from feedback_analyzer.intent import IntentClassifier, IntentInferenceError, IntentPrediction, IntentSetupError


class FakeTokenizer:
    truncation_side = "left"

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        count = len(text.split()) + 2
        if kwargs["truncation"]:
            count = min(count, kwargs["max_length"])
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.ones((1, count), dtype=torch.long),
                    "attention_mask": torch.ones((1, count), dtype=torch.long)}
        return {"input_ids": list(range(count))}


class FakeModel:
    def __init__(self):
        self.scores = [0, 0, 0, 0, 0, 0]
        self.eval_calls = 0
        self.moves = []
        self.forward_checks = []
        self.cuda_load_error = False

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.moves.append(device)
        if device == "cuda" and self.cuda_load_error:
            raise RuntimeError("simulated CUDA initialization failure")
        return self

    def __call__(self, **inputs):
        self.forward_checks.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled(), set(inputs)))
        return SimpleNamespace(logits=torch.tensor([self.scores], dtype=torch.float32))


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    directory = tmp_path / "classifier"
    directory.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (directory / name).write_bytes(b"fake-local-artifact")
    summary = {"status": "complete", "mode": "full", "config": {"max_length": 128}}
    (directory / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    # A deliberate permutation proves predictions follow saved IDs, not tuple position.
    labels = dict(enumerate(reversed(LABELS)))
    config = SimpleNamespace(model_type="distilbert", num_labels=6, id2label=labels,
                             label2id={label: index for index, label in labels.items()}, max_position_embeddings=512)
    tokenizer, model = FakeTokenizer(), FakeModel()
    loading = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    calls = []

    def load_config(path, **options):
        calls.append(("config", path, options))
        return config

    def load_tokenizer(path, **options):
        calls.append(("tokenizer", path, options))
        return tokenizer

    def load_model(path, **options):
        calls.append(("model", path, options))
        return model, loading

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", load_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return SimpleNamespace(directory=directory, summary=summary, config=config, model=model,
                           tokenizer=tokenizer, loading=loading, calls=calls)


def test_local_saved_labels_and_eval_inference_mode_are_used(checkpoint, monkeypatch):
    def no_cuda_query():
        pytest.fail("Explicit CPU must not inspect CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", no_cuda_query)
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory, device="cpu")
    checkpoint.model.scores = [4, 0, 0, 0, 0, 0]
    prediction = classifier.predict("App crashes")
    assert prediction.category == LABELS[-1]  # saved ID 0 is Non-Qualifying
    assert prediction.device == "cpu"
    assert prediction.confidence > .9
    assert classifier.device_warning is None
    assert checkpoint.model.eval_calls == 1
    assert checkpoint.model.forward_checks == [(True, False, {"input_ids", "attention_mask"})]
    assert checkpoint.model.moves == ["cpu"]
    for _, path, options in checkpoint.calls:
        assert path == str(checkpoint.directory.resolve())
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
    assert checkpoint.calls[-1][2]["output_loading_info"] is True
    assert checkpoint.calls[-1][2]["use_safetensors"] is True


def test_low_confidence_preserves_model_category_without_spam_rule(checkpoint):
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory, device="cpu")
    checkpoint.model.scores = [0, 0, 0, 0, 0, .01]
    prediction = classifier.predict("Hi, checkout crashes")
    assert prediction.category == "Bug Report"  # saved ID 5
    assert prediction.confidence < .2


def test_original_text_reaches_tokenizer_and_only_prefix_is_used_when_truncated(checkpoint):
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory, device="cpu")
    message = "  Hi,\n" + "checkout " * 140 + "crashes!  "
    prediction = classifier.predict(message)
    assert all(text == message for text, _ in checkpoint.tokenizer.calls)
    assert checkpoint.tokenizer.truncation_side == "right"
    assert prediction.input_tokens == len(message.split()) + 2
    assert prediction.used_tokens == prediction.max_length == 128
    assert prediction.truncated
    short = classifier.predict("App crashes")
    assert short.input_tokens == short.used_tokens == 4
    assert not short.truncated


def test_missing_checkpoint_stops_before_model_loading_and_shows_training_command(tmp_path, checkpoint):
    with pytest.raises(IntentSetupError, match="python -m scripts.train_intent"):
        IntentClassifier.from_checkpoint(tmp_path / "missing")
    assert not checkpoint.calls


@pytest.mark.parametrize("field,value", [("status", "running"), ("mode", "smoke")])
def test_untrained_or_smoke_record_is_rejected(checkpoint, field, value):
    checkpoint.summary[field] = value
    (checkpoint.directory / "training_summary.json").write_text(json.dumps(checkpoint.summary), encoding="utf-8")
    with pytest.raises(IntentSetupError, match="completed full-training"):
        IntentClassifier.from_checkpoint(checkpoint.directory)
    assert not checkpoint.calls


@pytest.mark.parametrize("bad", [0, 513, "128", True])
def test_invalid_saved_length_rejected(checkpoint, bad):
    checkpoint.summary["config"]["max_length"] = bad
    (checkpoint.directory / "training_summary.json").write_text(json.dumps(checkpoint.summary), encoding="utf-8")
    with pytest.raises(IntentSetupError, match="input length"):
        IntentClassifier.from_checkpoint(checkpoint.directory)


def test_invalid_or_default_label_mapping_never_runs_random_categories(checkpoint):
    checkpoint.config.id2label = {index: f"LABEL_{index}" for index in range(6)}
    with pytest.raises(IntentSetupError, match="label mappings"):
        IntentClassifier.from_checkpoint(checkpoint.directory)
    assert [name for name, _, _ in checkpoint.calls] == ["config"]


def test_inconsistent_reverse_label_mapping_rejected(checkpoint):
    checkpoint.config.label2id["Bug Report"] = 0
    with pytest.raises(IntentSetupError, match="label mappings"):
        IntentClassifier.from_checkpoint(checkpoint.directory)


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_weight_load_cannot_silently_initialize_head(checkpoint, key):
    checkpoint.loading[key] = ["classifier.weight"]
    with pytest.raises(IntentSetupError, match="uninitialized weights"):
        IntentClassifier.from_checkpoint(checkpoint.directory)
    assert not checkpoint.model.moves
    assert not checkpoint.model.forward_checks


def test_cuda_unavailable_uses_cpu_with_visible_reason(checkpoint):
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory)
    assert classifier.device == "cpu"
    assert "CUDA is unavailable" in classifier.device_warning
    assert classifier.predict("App crashes").device == "cpu"


def test_cuda_initialization_failure_falls_back_to_cpu(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    checkpoint.model.cuda_load_error = True
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory)
    assert classifier.device == "cpu"
    assert "initialization failed" in classifier.device_warning
    assert checkpoint.model.moves == ["cuda", "cpu"]


def test_gpu_forward_oom_retries_once_on_cpu_without_changing_prediction(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory)
    attempts = []

    def simulated_forward(encoded):
        attempts.append(classifier.device)
        if classifier.device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return torch.tensor([[0., 0., 0., 0., 0., 3.]])

    monkeypatch.setattr(classifier, "_forward", simulated_forward)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = classifier.predict("App crashes")
    assert result.category == "Bug Report"
    assert result.device == "cpu"
    assert attempts == ["cuda", "cpu"]
    assert "switched to CPU" in classifier.device_warning
    classifier.predict("Checkout is broken")
    assert attempts == ["cuda", "cpu", "cpu"]


@pytest.mark.parametrize("scores", [[float("nan")] * 6, [float("inf")] * 6, [0, 1]])
def test_nonfinite_or_wrong_shape_output_is_error_not_nonqualifying(checkpoint, scores):
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory, device="cpu")
    checkpoint.model.scores = scores
    with pytest.raises(IntentInferenceError, match="invalid category scores"):
        classifier.predict("App crashes")


def test_cpu_model_failure_is_explicit_and_not_replaced(checkpoint, monkeypatch):
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory, device="cpu")

    def failing_forward(encoded):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(classifier, "_forward", failing_forward)
    with pytest.raises(IntentInferenceError, match="no category was substituted"):
        classifier.predict("App crashes")


@pytest.mark.parametrize("message", [None, "", " \n "])
def test_non_message_inputs_rejected_without_model_work(checkpoint, message):
    classifier = IntentClassifier.from_checkpoint(checkpoint.directory, device="cpu")
    with pytest.raises(IntentInferenceError, match="nonblank"):
        classifier.predict(message)
    assert not checkpoint.model.forward_checks


def test_prediction_contract_rejects_inconsistent_truncation():
    with pytest.raises(ValueError, match="Truncation flag"):
        IntentPrediction(category="Bug Report", confidence=.5, input_tokens=20, used_tokens=10,
                         max_length=10, truncated=False, device="cpu")
