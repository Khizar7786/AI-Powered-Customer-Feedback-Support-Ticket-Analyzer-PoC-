"""Lightweight setup inspection; never load models or contact an API."""

from dataclasses import dataclass
from pathlib import Path

from feedback_analyzer.config import AppConfig


@dataclass(frozen=True)
class SetupStatus:
    state: str
    detail: str


@dataclass(frozen=True)
class DeviceStatus:
    device: str
    detail: str
    warning: str | None = None


def inspect_model_files(directory: Path) -> SetupStatus:
    """Detect common saved model/tokenizer files without validating weights.

    Detection cannot prove a classifier was trained or that its labels are
    correct. Loading and real inference checks belong to Stages 5 and 6.
    """
    try:
        if not directory.is_dir():
            return SetupStatus("Missing", "No local model directory was found.")

        def nonempty(name: str) -> bool:
            path = directory / name
            return path.is_file() and path.stat().st_size > 0

        missing = []
        if not nonempty("config.json"):
            missing.append("config.json")
        if not nonempty("tokenizer_config.json"):
            missing.append("tokenizer_config.json")
        if not any(nonempty(name) for name in ("model.safetensors", "pytorch_model.bin")):
            missing.append("model.safetensors or pytorch_model.bin")
        if not (
            nonempty("tokenizer.json")
            or nonempty("vocab.txt")
            or (nonempty("vocab.json") and nonempty("merges.txt"))
        ):
            missing.append("tokenizer.json or vocabulary files")
        if missing:
            return SetupStatus("Incomplete", "Missing saved files: " + "; ".join(missing) + ".")
        return SetupStatus(
            "Detected (unverified)",
            "Model and tokenizer files exist. Weights, labels, and inference have not been validated.",
        )
    except OSError:
        return SetupStatus("Unreadable", "The configured model directory could not be inspected.")


def inspect_api_config(config: AppConfig) -> SetupStatus:
    """Report presence only; never reveal keys or make a network request."""
    if config.invalid_provider:
        return SetupStatus("Unsupported provider", "Set LLM_PROVIDER to openai or gemini in Stage 7.")
    if config.api_provider is None:
        return SetupStatus("Not configured", "A generative API provider has not been selected.")
    missing = []
    if not config.api_model_configured:
        missing.append("LLM_MODEL")
    if not config.api_key_configured:
        missing.append({"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[config.api_provider])
    if missing:
        return SetupStatus("Incomplete", "Missing configuration: " + ", ".join(missing) + ".")
    return SetupStatus(
        "Configured (unverified)",
        "Provider, model, and key are present. Model availability and API access have not been checked.",
    )


def inspect_device() -> DeviceStatus:
    """Inspect PyTorch/CUDA support with an explicit CPU-safe fallback."""
    try:
        import torch
    except (ImportError, OSError):
        return DeviceStatus("Unavailable", "PyTorch could not be imported. Follow docs/setup.md.")

    version = f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda or 'not included'}."
    try:
        if not torch.cuda.is_available():
            return DeviceStatus("CPU", version + " CUDA is unavailable; CPU execution is supported.")
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        return DeviceStatus(
            "CUDA",
            f"{properties.name}. {version} Free VRAM: {free / 1024**2:.0f} MiB / {total / 1024**2:.0f} MiB.",
        )
    except (RuntimeError, AssertionError, OSError):
        return DeviceStatus(
            "CPU",
            version,
            "CUDA inspection failed. Use CPU until the driver/runtime setup is verified.",
        )
