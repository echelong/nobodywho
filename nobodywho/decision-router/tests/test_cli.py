from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from decision_router import cli
from decision_router import config as cfg

NOBODYWHO_MODEL = Path(
    os.environ.get(
        "DECISION_ROUTER_TEST_MODEL",
        Path.home()
        / ".cache/nobodywho/models/NobodyWho/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_K_M.gguf",
    )
)

REQUEST = {
    "question_id": "next_step",
    "state": "Two identical retries of a date-parser test failed with a one-hour offset.",
    "question": "What should the agent do next?",
    "choices": {"retry_same": "Retry unchanged.", "inspect_tz": "Inspect timezone handling."},
    "risk": "low",
}


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr()


def test_provider_switching(capsys):
    code, out = run(capsys, "provider")
    assert code == 0 and out.out.startswith("jev")
    for mode in ("off", "local", "shadow", "compare", "jev"):
        assert run(capsys, "provider", mode)[0] == 0
        assert json.loads(cfg.config_path().read_text())["mode"] == mode
        assert run(capsys, "provider")[1].out.startswith(mode)
    assert cfg.config_path().stat().st_mode & 0o077 == 0


def test_provider_rejects_unknown_mode(capsys):
    with pytest.raises(SystemExit):
        cli.main(["provider", "auto"])


def test_env_overrides_mode(capsys, monkeypatch):
    run(capsys, "provider", "jev")
    monkeypatch.setenv("DECISION_ROUTER_MODE", "off")
    assert run(capsys, "provider")[1].out.startswith("off")


def test_ask_in_off_mode_skips(capsys):
    run(capsys, "provider", "off")
    code, out = run(capsys, "ask", "--json", json.dumps(REQUEST))
    data = json.loads(out.out)
    assert code == 0 and data["follow"] is None and data["skipped"] == "mode_off"


def test_ask_bypass_flag_and_directive(capsys):
    run(capsys, "provider", "jev")
    for extra in (["--bypass"], ["--user-directive", "bypass jev"]):
        data = json.loads(run(capsys, "ask", "--json", json.dumps(REQUEST), *extra)[1].out)
        assert data["skipped"] == "bypass"


def test_ask_invalid_request_is_reported_not_raised(capsys):
    bad = dict(REQUEST, choices=["only_one"])
    code, out = run(capsys, "ask", "--json", json.dumps(bad))
    data = json.loads(out.out)
    assert code == 2 and data["error"] == "invalid_request" and data["follow"] is None
    code, out = run(capsys, "ask", "--json", "{not json")
    assert code == 2 and json.loads(out.out)["error"] == "unreadable_request"


def test_ask_jev_without_key_fails_safe(capsys, tmp_path):
    cfg.update({"mode": "jev", "jev": {"key_file": str(tmp_path / "missing.key")}})
    code, out = run(capsys, "ask", "--json", json.dumps(REQUEST))
    data = json.loads(out.out)
    assert code == 0 and data["follow"] is None
    assert data["decision"]["fallback_reason"] == "jev_unavailable"


def test_ask_local_without_model_fails_safe(capsys):
    cfg.update({"mode": "local", "local": {"model_path": "/nonexistent/model.gguf"}})
    data = json.loads(run(capsys, "ask", "--json", json.dumps(REQUEST))[1].out)
    assert data["follow"] is None
    assert data["decision"]["fallback_reason"] == "local_model_unavailable"


def test_install_rule(capsys, tmp_path):
    code, _ = run(capsys, "install-rule", "--dest", str(tmp_path / "rules"))
    text = (tmp_path / "rules" / cli.RULE_NAME).read_text()
    assert code == 0 and "bypass decision router" in text and "bypass jev" in text


@pytest.mark.model
@pytest.mark.skipif(not NOBODYWHO_MODEL.is_file(), reason="local GGUF model not present")
def test_real_local_decision(capsys):
    pytest.importorskip("nobodywho")
    cfg.update({"mode": "local", "local": {"model_path": str(NOBODYWHO_MODEL), "samples": 3}})
    data = json.loads(run(capsys, "ask", "--json", json.dumps(REQUEST))[1].out)
    decision = data["decision"]
    assert decision["error"] is None, decision
    assert decision["provider"] == "nobodywho"
    assert decision["confidence_kind"] == "sample_stability"
    assert sum(decision["votes"].values()) == 3
    assert set(decision["votes"]) <= {"retry_same", "inspect_tz", "ABSTAIN"}
    assert decision["details"]["quantization"] == "Q4_K_M"
