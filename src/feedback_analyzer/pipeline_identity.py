"""Content fingerprints for models, policies and public API settings; no datasets or secrets."""

from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from feedback_analyzer.config import CLASSIFIER_MODEL_ID, SENTIMENT_MODEL_ID
from feedback_analyzer.generation import GenerationError, PROMPT_VERSION, SYSTEM_PROMPT, load_generation_options
from feedback_analyzer.schemas import GeneratedDetails


PIPELINE_VERSION = "batch-v1"
MODEL_FILES = ("config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt",
               "label_mappings.json", "training_summary.json", "sentiment_source.json")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=64)
def _file_hash(path: str, size: int, modified: int, changed: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_fingerprint(directory: Path) -> dict:
    files = {}
    for name in MODEL_FILES:
        path = directory / name
        try:
            stat = path.stat()
        except FileNotFoundError:
            files[name] = None
        else:
            files[name] = _file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return {"sha256": digest(files), "files": files}


class PipelineIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    metadata: dict

    @property
    def config_hash(self) -> str:
        return digest(self.metadata)


def build_identity(config, device="cpu") -> PipelineIdentity:
    try:
        options = load_generation_options(config.project_root).model_dump(mode="json")
    except GenerationError as error:
        options = {"configuration_error": error.code}
    versions = {}
    for package in ("torch", "transformers", "pydantic", "openai"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    # Source hashes also invalidate reuse after policy/preprocessing changes without a version bump.
    folder = Path(__file__).parent
    sources = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in
               ("filtering.py", "ingestion.py", "intent.py", "sentiment.py", "schemas.py", "batch_contracts.py", "ticket_analysis.py", "generation.py", "pipeline.py")}
    return PipelineIdentity(metadata={
        "pipeline_version": PIPELINE_VERSION, "device": device, "versions": versions,
        "category": {"base_model": CLASSIFIER_MODEL_ID, **model_fingerprint(config.classifier_dir)},
        "sentiment": {"base_model": SENTIMENT_MODEL_ID, **model_fingerprint(config.sentiment_dir)},
        "generation": options, "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(SYSTEM_PROMPT),
        "schema_sha256": digest(GeneratedDetails.model_json_schema()), "source_sha256": sources,
    })
