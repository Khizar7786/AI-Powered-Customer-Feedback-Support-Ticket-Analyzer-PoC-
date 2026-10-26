"""Explicit deterministic batch doubles; never use local credentials or live providers."""

from types import SimpleNamespace

import pytest

from feedback_analyzer.generation import GeneratedResult, GenerationError, GenerationReceipt
from feedback_analyzer.intent import IntentPrediction
from feedback_analyzer.pipeline import BatchRuntime
from feedback_analyzer.pipeline_identity import PipelineIdentity
from feedback_analyzer.schemas import GeneratedDetails
from feedback_analyzer.sentiment import SentimentPrediction


@pytest.fixture
def batch_setup():
    calls = {"category": [], "sentiment": [], "api": []}
    failures = set()

    class Category:
        def predict(self, text):
            calls["category"].append(text)
            if text == "category fails":
                raise RuntimeError("private model exception")
            return IntentPrediction(category="Non-Qualifying" if text == "social greeting" else "Bug Report",
                                    confidence=.4, input_tokens=10, used_tokens=10, max_length=128, truncated=False, device="cpu")

    class Sentiment:
        def predict(self, text):
            calls["sentiment"].append(text)
            if text == "sentiment fails":
                raise RuntimeError("private sentiment exception")
            return SentimentPrediction(sentiment="Negative", confidence=.8, input_tokens=10, used_tokens=10,
                                       max_length=512, truncated=False, device="cpu", model_input=text)

    class Provider:
        def generate(self, context):
            calls["api"].append(context.message)
            if context.message in failures:
                raise GenerationError("refusal", "Provider refused; no complete reply was created.")
            return GeneratedResult(details=GeneratedDetails(priority="Medium", product_feature=None,
                suggested_reply="Thanks for reporting this. Could you share the steps that trigger the issue?"),
                receipt=GenerationReceipt(model="test-mock-not-live", attempts=1, duration_seconds=.01,
                                          completed_at_utc="2026-01-01T00:00:00+00:00"))

    provider = Provider()
    return SimpleNamespace(runtime=BatchRuntime(Category(), Sentiment(), lambda: provider), provider=provider,
                           calls=calls, failures=failures,
                           identity=PipelineIdentity(metadata={"pipeline_version": "unit-batch-v1", "model": "explicit-mock"}))
