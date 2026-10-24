"""Read local setup configuration without retaining API secrets in UI state."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_MODEL_ID = "distilbert/distilbert-base-uncased"
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@dataclass(frozen=True)
class AppConfig:
    """Only public paths and configuration-presence flags belong in readiness."""

    project_root: Path
    classifier_dir: Path
    sentiment_dir: Path
    api_provider: str | None
    invalid_provider: bool
    api_model_configured: bool
    api_key_configured: bool


def _configured(value: str | None) -> bool:
    """Treat blank values and obvious example placeholders as missing."""
    text = (value or "").strip().lower()
    return bool(text) and not (
        text in {"changeme", "change_me", "example", "placeholder", "none", "null"}
        or text.startswith(("your_", "your-", "replace_", "replace-", "<"))
    )


def _local_path(value: str | None, default: str, root: Path) -> Path:
    path = Path(value.strip() if value and value.strip() else default).expanduser()
    return path if path.is_absolute() else root / path


def load_config(
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Resolve paths relative to the project, with environment overriding .env.

    This does not choose a provider, contact a service, or put key values into
    the returned object. Configuration presence is not connection validation.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    values = dict(dotenv_values(root / ".env", interpolate=False))
    values.update(os.environ if environ is None else environ)
    provider_value = (values.get("LLM_PROVIDER") or "").strip().lower()
    provider = provider_value if provider_value in {"openai", "gemini"} else None
    key_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}.get(provider)
    return AppConfig(
        project_root=root,
        classifier_dir=_local_path(values.get("CLASSIFIER_DIR"), "models/ticket_classifier", root),
        sentiment_dir=_local_path(values.get("SENTIMENT_DIR"), "models/sentiment", root),
        api_provider=provider,
        invalid_provider=_configured(provider_value) and provider is None,
        api_model_configured=_configured(values.get("LLM_MODEL")),
        api_key_configured=bool(key_name and _configured(values.get(key_name))),
    )
