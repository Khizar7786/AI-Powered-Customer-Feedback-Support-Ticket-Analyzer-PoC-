"""Explicit acquisition of existing Cardiff weights, separate from UI reruns."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from feedback_analyzer.config import SENTIMENT_MODEL_ID


DEFAULT_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"


def download_sentiment_model(output_dir: Path, revision: str = DEFAULT_REVISION) -> dict:
    """Download a pinned official snapshot; never initialize or train new weights."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full immutable model revision (40 hexadecimal characters).")
    output_dir = Path(output_dir).resolve()
    source_path = output_dir / "sentiment_source.json"
    progress_path = output_dir / "sentiment_download.json"
    identity = {"model_id": SENTIMENT_MODEL_ID, "revision": revision}
    if output_dir.exists() and any(output_dir.iterdir()):
        marker = source_path if source_path.is_file() else progress_path
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("Refusing to replace an unrelated nonempty model directory; select a new --model-dir.") from None
        if not isinstance(existing, dict) or any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing model identity differs; select a new --model-dir.")
        if source_path.is_file() and existing.get("status") == "ready" and all(
            (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0 for name in (
            "pytorch_model.bin", "config.json", "tokenizer_config.json", "tokenizer.json",
        )):
            return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress_path.write_text(json.dumps({**identity, "status": "downloading"}, indent=2) + "\n", encoding="utf-8")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot_download(
        SENTIMENT_MODEL_ID, revision=revision, local_dir=str(output_dir), max_workers=1,
        allow_patterns=["config.json", "pytorch_model.bin", "merges.txt", "vocab.json", "special_tokens_map.json", "README.md"],
    )
    # The official snapshot has no tokenizer_config.json; save the same tokenizer
    # locally so subsequent app loading needs neither downloads nor guesses.
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 512
    tokenizer.save_pretrained(str(output_dir))
    digest = hashlib.sha256()
    with (output_dir / "pytorch_model.bin").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    record = {
        **identity, "status": "ready", "trained_locally": False,
        "source_url": "https://huggingface.co/" + SENTIMENT_MODEL_ID,
        "license": "cc-by-4.0", "attribution": "Cardiff NLP / TimeLMs; see the downloaded README.md model card.",
        "weights_format": "pytorch_model.bin (loaded with weights_only=True)",
        "weights_sha256": digest.hexdigest(), "weights_bytes": (output_dir / "pytorch_model.bin").stat().st_size,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "download_seconds": time.perf_counter() - started,
    }
    source_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    progress_path.write_text(json.dumps({**identity, "status": "complete"}, indent=2) + "\n", encoding="utf-8")
    return record
