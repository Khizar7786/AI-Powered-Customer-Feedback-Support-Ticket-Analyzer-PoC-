"""Explicit sentiment model/tokenizer doubles; no network, weights, or GPU used."""

import json
from types import SimpleNamespace

import pytest
import torch
import transformers

from feedback_analyzer.sentiment import (
    MODEL_ID, SentimentClassifier, SentimentInferenceError, SentimentPrediction, SentimentSetupError, preprocess_message,
)


class FakeTokenizer:
    model_max_length = 512
    truncation_side = "left"

    def __init__(self):
        self.calls = []

    def __call__(self, text, **options):
        self.calls.append((text, options))
        count = len(text.split()) + 2
        if options["truncation"]:
            count = min(count, options["max_length"])
        if options.get("return_tensors") == "pt":
            return {"input_ids": torch.ones((1, count), dtype=torch.long),
                    "attention_mask": torch.ones((1, count), dtype=torch.long)}
        return {"input_ids": list(range(count))}


class FakeModel:
    def __init__(self):
        self.scores = [4., 0., 0.]
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
            raise RuntimeError("explicit mock initialization failure")
        return self

    def __call__(self, **inputs):
        self.forward_checks.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled(), set(inputs)))
        return SimpleNamespace(logits=torch.tensor([self.scores], dtype=torch.float32))


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    directory = tmp_path / "sentiment"
    directory.mkdir()
    for name in ("config.json", "tokenizer_config.json", "pytorch_model.bin"):
        (directory / name).write_bytes(b"explicit fake artifact")
    source = {"model_id": MODEL_ID, "revision": "1" * 40}
    (directory / "sentiment_source.json").write_text(json.dumps(source), encoding="utf-8")
    # Deliberate permutation tests that IDs are read rather than guessed.
    labels = {0: "positive", 1: "negative", 2: "neutral"}
    config = SimpleNamespace(model_type="roberta", num_labels=3, id2label=labels,
                             label2id={label: index for index, label in labels.items()},
                             max_position_embeddings=514, pad_token_id=1)
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
    return SimpleNamespace(directory=directory, source=source, config=config, tokenizer=tokenizer, model=model,
                           loading=loading, calls=calls)


def test_default_cpu_uses_saved_label_order_with_eval_and_inference_mode(checkpoint, monkeypatch):
    def no_cuda():
        pytest.fail("CPU default must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", no_cuda)
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    result = classifier.predict("Thanks, this worked beautifully!")
    assert result.sentiment == "Positive"  # saved ID 0, not official example's ID 2
    assert result.device == "cpu"
    assert result.confidence > .9
    assert classifier.revision == "1" * 40
    assert classifier.device_warning is None
    assert checkpoint.model.moves == ["cpu"]
    assert checkpoint.model.eval_calls == 1
    assert checkpoint.model.forward_checks == [(True, False, {"input_ids", "attention_mask"})]
    for _, path, options in checkpoint.calls:
        assert path == str(checkpoint.directory.resolve())
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
    assert checkpoint.calls[-1][2]["weights_only"] is True
    assert checkpoint.calls[-1][2]["use_safetensors"] is False


def test_model_card_placeholders_change_model_input_only(checkpoint):
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    original = "  @support\nmy invoice at https://example.com/i is wrong!  "
    result = classifier.predict(original)
    assert original == "  @support\nmy invoice at https://example.com/i is wrong!  "
    assert result.model_input == "@user my invoice at http is wrong!"
    assert all(text == result.model_input for text, _ in checkpoint.tokenizer.calls)
    assert preprocess_message("@ http://a @alice www.example.com") == "@ http @user www.example.com"


def test_long_input_is_right_truncated_and_full_count_recorded(checkpoint):
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    result = classifier.predict("good " * 600)
    assert result.input_tokens == 602
    assert result.used_tokens == result.max_length == 512
    assert result.truncated
    assert checkpoint.tokenizer.truncation_side == "right"
    assert checkpoint.tokenizer.calls[0][1]["verbose"] is False
    assert not classifier.predict("All good").truncated


def test_positive_sounding_feature_request_is_not_forced_to_positive(checkpoint):
    checkpoint.model.scores = [0., 0., 4.]  # saved neutral ID 2
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    result = classifier.predict("Hi team, it would be awesome if you could add a dark mode to the mobile app.")
    assert result.sentiment == "Neutral"


def test_missing_checkpoint_gives_download_command_before_transformers(tmp_path, checkpoint):
    with pytest.raises(SentimentSetupError, match="scripts.evaluate_sentiment --download-model"):
        SentimentClassifier.from_checkpoint(tmp_path / "missing")
    assert not checkpoint.calls


@pytest.mark.parametrize("field,value", [("model_id", "different-model"), ("revision", "main"), ("revision", True)])
def test_wrong_source_identity_or_unpinned_revision_is_rejected(checkpoint, field, value):
    checkpoint.source[field] = value
    (checkpoint.directory / "sentiment_source.json").write_text(json.dumps(checkpoint.source), encoding="utf-8")
    with pytest.raises(SentimentSetupError, match="identity/revision"):
        SentimentClassifier.from_checkpoint(checkpoint.directory)
    assert not checkpoint.calls


def test_missing_weights_is_setup_error_and_never_a_random_model(checkpoint):
    (checkpoint.directory / "pytorch_model.bin").unlink()
    with pytest.raises(SentimentSetupError, match="weights are missing"):
        SentimentClassifier.from_checkpoint(checkpoint.directory)
    assert not checkpoint.calls


def test_safetensors_is_preferred_when_present(checkpoint):
    (checkpoint.directory / "model.safetensors").write_bytes(b"explicit fake safer weights")
    SentimentClassifier.from_checkpoint(checkpoint.directory)
    assert checkpoint.calls[-1][2]["use_safetensors"] is True


def test_uninformative_default_labels_are_rejected_before_model_load(checkpoint):
    checkpoint.config.id2label = {index: f"LABEL_{index}" for index in range(3)}
    with pytest.raises(SentimentSetupError, match="Saved labels"):
        SentimentClassifier.from_checkpoint(checkpoint.directory)
    assert [name for name, _, _ in checkpoint.calls] == ["config"]


def test_inconsistent_reverse_labels_are_rejected(checkpoint):
    checkpoint.config.label2id["positive"] = 1
    with pytest.raises(SentimentSetupError, match="Saved labels"):
        SentimentClassifier.from_checkpoint(checkpoint.directory)


@pytest.mark.parametrize("field,value", [("model_type", "distilbert"), ("num_labels", 2)])
def test_wrong_architecture_or_label_count_is_rejected(checkpoint, field, value):
    setattr(checkpoint.config, field, value)
    with pytest.raises(SentimentSetupError, match="Saved labels"):
        SentimentClassifier.from_checkpoint(checkpoint.directory)


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_or_mismatched_trainable_weights_never_initialize_random_head(checkpoint, key):
    checkpoint.loading[key] = ["classifier.out_proj.weight"]
    with pytest.raises(SentimentSetupError, match="random weights are forbidden"):
        SentimentClassifier.from_checkpoint(checkpoint.directory)
    assert not checkpoint.model.moves


def test_legacy_nontrainable_position_buffers_do_not_block_existing_weights(checkpoint):
    checkpoint.loading["unexpected_keys"] = ["roberta.embeddings.position_ids"]
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    assert classifier.predict("Thanks!").sentiment == "Positive"


def test_official_unused_pooler_is_recorded_without_rejecting_trained_classifier(checkpoint):
    keys = ["roberta.pooler.dense.weight", "roberta.pooler.dense.bias"]
    checkpoint.loading["unexpected_keys"] = keys
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    assert classifier.unused_checkpoint_keys == tuple(sorted(keys))
    assert classifier.predict("A useful improvement").sentiment == "Positive"


def test_unused_pooler_never_excuses_a_missing_classifier_weight(checkpoint):
    checkpoint.loading["unexpected_keys"] = ["roberta.pooler.dense.weight"]
    checkpoint.loading["missing_keys"] = ["classifier.out_proj.weight"]
    with pytest.raises(SentimentSetupError, match="random weights"):
        SentimentClassifier.from_checkpoint(checkpoint.directory)


@pytest.mark.parametrize("tokenizer_limit,positions,expected", [(128, 514, 128), (512, 130, 128)])
def test_capacity_respects_both_tokenizer_and_roberta_position_offset(checkpoint, tokenizer_limit, positions, expected):
    checkpoint.tokenizer.model_max_length = tokenizer_limit
    checkpoint.config.max_position_embeddings = positions
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    assert classifier.max_length == expected


def test_auto_cuda_unavailable_uses_cpu_with_reason(checkpoint):
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory, device="auto")
    assert classifier.device == "cpu"
    assert "CUDA is unavailable" in classifier.device_warning


def test_cuda_initialization_failure_falls_back_to_same_model_on_cpu(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    checkpoint.model.cuda_load_error = True
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory, device="auto")
    assert classifier.device == "cpu"
    assert checkpoint.model.moves == ["cuda", "cpu"]
    assert "initialization failed" in classifier.device_warning


def test_cuda_forward_failure_retries_once_on_cpu_without_changing_model(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory, device="auto")
    attempts = []

    def forward(encoded):
        attempts.append(classifier.device)
        if classifier.device == "cuda":
            raise RuntimeError("simulated out of memory")
        return torch.tensor([[0., 4., 0.]])

    monkeypatch.setattr(classifier, "_forward", forward)
    result = classifier.predict("Payment failed again")
    assert result.sentiment == "Negative"
    assert result.device == "cpu"
    assert attempts == ["cuda", "cpu"]
    assert "switched to CPU" in classifier.device_warning


@pytest.mark.parametrize("scores", [[float("nan")] * 3, [float("inf")] * 3, [0., 1.]])
def test_invalid_scores_are_explicit_errors_not_fabricated_sentiments(checkpoint, scores):
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    checkpoint.model.scores = scores
    with pytest.raises(SentimentInferenceError, match="invalid sentiment scores"):
        classifier.predict("This failed")


def test_cpu_runtime_failure_is_never_replaced_with_a_rule(checkpoint, monkeypatch):
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)

    def fail(encoded):
        raise RuntimeError("explicit mock failure")

    monkeypatch.setattr(classifier, "_forward", fail)
    with pytest.raises(SentimentInferenceError, match="no sentiment was substituted"):
        classifier.predict("Please fix this")


@pytest.mark.parametrize("message", [None, "", " \n "])
def test_blank_or_nonmessage_input_is_rejected_without_forward(checkpoint, message):
    classifier = SentimentClassifier.from_checkpoint(checkpoint.directory)
    with pytest.raises(SentimentInferenceError, match="nonblank"):
        classifier.predict(message)
    assert not checkpoint.model.forward_checks


def test_prediction_schema_rejects_wrong_truncation_flag():
    with pytest.raises(ValueError, match="truncation flag"):
        SentimentPrediction(sentiment="Positive", confidence=.6, input_tokens=600, used_tokens=512,
                            max_length=512, truncated=False, device="cpu", model_input="fixture")
