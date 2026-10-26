"""Identity changes when behavior changes, and excludes credentials/installation paths."""

import shutil

from feedback_analyzer.config import load_config
from feedback_analyzer.pipeline_identity import build_identity


def setup(root):
    root.mkdir(exist_ok=True)
    (root / ".env").write_text("LLM_PROVIDER=openai\nLLM_MODEL=test-model\nOPENAI_API_KEY=unit-test-key\n", encoding="utf-8")
    config = load_config(root, environ={})
    for directory in (config.classifier_dir, config.sentiment_dir):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text('{"labels":"explicit fake"}', encoding="utf-8")
        (directory / "model.safetensors").write_bytes(b"fake weights only")
    return config


def test_key_rotation_does_not_change_identity_or_expose_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = setup(tmp_path)
    first = build_identity(config)
    content = (tmp_path / ".env").read_text()
    (tmp_path / ".env").write_text(content.replace("unit-test-key", "different-unit-secret"))
    second = build_identity(config)
    assert first.config_hash == second.config_hash
    assert "unit-test-key" not in first.model_dump_json() and "different-unit-secret" not in second.model_dump_json()


def test_model_and_public_settings_changes_invalidate_reuse(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config = setup(tmp_path)
    first = build_identity(config)
    (config.classifier_dir / "model.safetensors").write_bytes(b"different fake weights")
    second = build_identity(config)
    assert first.config_hash != second.config_hash
    monkeypatch.setenv("LLM_MODEL", "another-test-model")
    assert build_identity(config).config_hash != second.config_hash


def test_identical_models_moved_to_another_directory_keep_identity(tmp_path):
    config = setup(tmp_path / "one")
    destination = tmp_path / "two"
    shutil.copytree(tmp_path / "one", destination)
    assert build_identity(config).config_hash == build_identity(load_config(destination, environ={})).config_hash


def test_missing_credentials_do_not_prevent_identity_or_change_it(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = setup(tmp_path)
    first = build_identity(config)
    (tmp_path / ".env").write_text("LLM_PROVIDER=openai\nLLM_MODEL=test-model\n")
    assert first.config_hash == build_identity(config).config_hash
