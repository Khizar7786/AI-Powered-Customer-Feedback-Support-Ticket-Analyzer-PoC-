"""Offline setup-contract checks; these do not verify real model/API access."""

import sys
from types import SimpleNamespace

import pytest

from feedback_analyzer.config import load_config
from feedback_analyzer.readiness import inspect_api_config, inspect_device, inspect_model_files


def test_config_paths_stay_project_relative_and_environment_overrides_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "CLASSIFIER_DIR=local/classifier\nSENTIMENT_DIR=local/sentiment\n"
        "LLM_PROVIDER=openai\nLLM_MODEL=dotenv-model\nOPENAI_API_KEY=dummy-dotenv-value\n",
        encoding="utf-8",
    )
    unrelated = tmp_path / "another-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    config = load_config(tmp_path, environ={"LLM_PROVIDER": "gemini"})

    assert config.classifier_dir == tmp_path / "local/classifier"
    assert config.sentiment_dir == tmp_path / "local/sentiment"
    assert config.api_provider == "gemini"
    assert config.api_model_configured
    assert not config.api_key_configured  # An OpenAI key cannot configure Gemini.


def test_api_presence_never_retains_or_claims_to_validate_credentials(tmp_path):
    key = "dummy-test-only-secret"
    model = "dummy-model-presence-only"
    config = load_config(
        tmp_path,
        environ={"LLM_PROVIDER": "openai", "LLM_MODEL": model, "OPENAI_API_KEY": key},
    )
    status = inspect_api_config(config)

    assert status.state == "Configured (unverified)"
    assert key not in repr(config) + repr(status)
    assert model not in repr(config) + repr(status)
    assert key not in vars(config).values()
    assert model not in vars(config).values()


@pytest.mark.parametrize(
    ("values", "state"),
    [
        ({}, "Not configured"),
        ({"LLM_PROVIDER": "other"}, "Unsupported provider"),
        (
            {"LLM_PROVIDER": "openai", "LLM_MODEL": "<choose-model>", "OPENAI_API_KEY": "dummy"},
            "Incomplete",
        ),
        (
            {"LLM_PROVIDER": "gemini", "LLM_MODEL": "dummy-model", "GEMINI_API_KEY": "your_key_here"},
            "Incomplete",
        ),
        (
            {"LLM_PROVIDER": "openai", "LLM_MODEL": "dummy-model", "GEMINI_API_KEY": "dummy"},
            "Incomplete",
        ),
    ],
)
def test_absent_placeholder_or_wrong_provider_values_are_not_configured(tmp_path, values, state):
    config = load_config(tmp_path, environ=values)
    assert inspect_api_config(config).state == state


def test_artifact_detection_requires_nonempty_files_and_never_proves_readiness(tmp_path):
    model_dir = tmp_path / "model"
    assert inspect_model_files(model_dir).state == "Missing"
    model_dir.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors", "tokenizer.json"):
        (model_dir / name).touch()
    assert inspect_model_files(model_dir).state == "Incomplete"

    # Deliberately invalid artifacts prove detection is not model validation.
    for path in model_dir.iterdir():
        path.write_text("not a real model", encoding="utf-8")
    status = inspect_model_files(model_dir)
    assert status.state == "Detected (unverified)"
    assert "not been validated" in status.detail


def test_device_inspection_supports_cpu_without_querying_gpu(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="test-only",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    status = inspect_device()
    assert status.device == "CPU"
    assert status.warning is None


def test_failed_cuda_probe_returns_explicit_cpu_fallback(monkeypatch):
    def fail_probe():
        raise RuntimeError("simulated unavailable driver")

    fake_torch = SimpleNamespace(
        __version__="test-only",
        version=SimpleNamespace(cuda="test-runtime"),
        cuda=SimpleNamespace(is_available=fail_probe),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    status = inspect_device()
    assert status.device == "CPU"
    assert "CUDA inspection failed" in status.warning
