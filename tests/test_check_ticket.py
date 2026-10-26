"""Configuration/live-check entry point tests never contact a provider."""

import json

from pydantic import SecretStr

from feedback_analyzer.generation import GenerationError, GenerationSettings
from scripts import check_ticket


def test_missing_credentials_write_honest_pending_status_without_inference(tmp_path, monkeypatch, capsys):
    def missing():
        raise GenerationError("missing_credentials", "Set OPENAI_API_KEY in ignored .env.")

    monkeypatch.setattr(check_ticket, "load_generation_settings", missing)
    monkeypatch.setattr(check_ticket.IntentClassifier, "from_checkpoint", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load")))
    output = tmp_path / "live_check.json"
    assert check_ticket.main(["--live", "--output", str(output)]) == 2
    record = json.loads(output.read_text())
    assert record["status"] == "pending_configuration" and record["live_api_verified"] is False
    assert "result" not in record and "missing_credentials" in capsys.readouterr().out


def test_configuration_check_preserves_prior_live_report_and_never_constructs_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(check_ticket, "load_generation_settings", lambda: GenerationSettings(
        model="gpt-5.4-mini", api_key=SecretStr("unit-secret-only")))
    monkeypatch.setattr(check_ticket, "create_provider", lambda: (_ for _ in ()).throw(AssertionError("must not call")))
    output = tmp_path / "live_check.json"
    output.write_text("existing report fixture", encoding="utf-8")
    assert check_ticket.main(["--output", str(output)]) == 0
    assert output.read_text() == "existing report fixture"
    record = json.loads((tmp_path / "configuration_check.json").read_text())
    assert record["status"] == "configuration_present_unverified"
    assert record["live_api_verified"] is False
    assert "unit-secret-only" not in json.dumps(record)
