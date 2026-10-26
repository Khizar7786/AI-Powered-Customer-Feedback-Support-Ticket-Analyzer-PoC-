"""Explicit acquisition guards; fake downloads never create usable model weights."""

import hashlib
import json
from pathlib import Path

import pytest

from feedback_analyzer.config import SENTIMENT_MODEL_ID
from feedback_analyzer.sentiment_setup import DEFAULT_REVISION, download_sentiment_model


@pytest.mark.parametrize("revision", ["main", "bad", "g" * 40])
def test_download_requires_immutable_revision_before_writing(tmp_path, revision):
    destination = tmp_path / "model"
    with pytest.raises(ValueError, match="immutable"):
        download_sentiment_model(destination, revision)
    assert not destination.exists()


def test_unrelated_directory_is_not_overwritten(tmp_path):
    marker = tmp_path / "user.txt"
    marker.write_text("retain this", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated"):
        download_sentiment_model(tmp_path)
    assert marker.read_text(encoding="utf-8") == "retain this"
    assert list(tmp_path.iterdir()) == [marker]


def test_existing_different_model_identity_is_rejected(tmp_path):
    (tmp_path / "sentiment_source.json").write_text(json.dumps({
        "model_id": "different/model", "revision": DEFAULT_REVISION,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        download_sentiment_model(tmp_path)


def test_download_only_acquires_existing_weights_and_records_their_hash(tmp_path, monkeypatch):
    import huggingface_hub
    import transformers

    calls = []
    original_weights = b"explicit fake pretrained bytes for setup testing only"

    def fake_snapshot(model_id, **options):
        calls.append((model_id, options))
        assert "tf_model.h5" not in options["allow_patterns"]
        (tmp_path / "model" / "pytorch_model.bin").write_bytes(original_weights)
        (tmp_path / "model" / "config.json").write_text("{}", encoding="utf-8")

    class FakeTokenizer:
        def save_pretrained(self, path):
            path = Path(path)
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    def fake_tokenizer(path, **options):
        assert options == {"local_files_only": True, "trust_remote_code": False}
        return FakeTokenizer()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tokenizer)
    destination = tmp_path / "model"
    record = download_sentiment_model(destination)
    assert calls[0][0] == SENTIMENT_MODEL_ID
    assert calls[0][1]["revision"] == DEFAULT_REVISION
    assert (destination / "pytorch_model.bin").read_bytes() == original_weights
    assert record["weights_sha256"] == hashlib.sha256(original_weights).hexdigest()
    assert record["trained_locally"] is False
    assert record["status"] == "ready"
    # A second explicit setup reuses complete artifacts without another download.
    assert download_sentiment_model(destination) == record
    assert len(calls) == 1


def test_nonobject_download_marker_is_rejected_cleanly(tmp_path):
    (tmp_path / "sentiment_download.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        download_sentiment_model(tmp_path)
