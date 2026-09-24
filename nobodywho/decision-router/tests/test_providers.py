from __future__ import annotations

import json
import struct
import subprocess
import sys
import urllib.error

import pytest
from helpers import FAKE_KEY, FakeTransport, jev_body, jev_provider, make_request

from decision_router.contract import ABSTAIN, CALIBRATED, SAMPLE_STABILITY
from decision_router.gguf_info import describe
from decision_router.local_worker import run as worker_run
from decision_router.providers import nobodywho as local_mod
from decision_router.providers.jev import JevProvider
from decision_router.providers.nobodywho import (
    NobodyWhoProvider,
    aggregate,
    grammar_for,
    plan_samples,
    subprocess_runner,
)

# ---------------------------------------------------------------- JEV


def test_jev_success_preserves_versioned_model_and_distribution(key_file):
    transport = FakeTransport(
        text=jev_body("inspect_tz", 0.97, {"inspect_tz": 0.97, "retry_same": 0.02, "rewrite": 0.01})
    )
    result = jev_provider(key_file, transport).decide(make_request())
    assert result.ok and result.choice == "inspect_tz"
    assert result.model == "jev-1.13.0"
    assert result.details["model_alias"] == "jev-latest"
    assert result.confidence == 0.97
    assert result.confidence_kind == CALIBRATED
    assert result.distribution == {"inspect_tz": 0.97, "retry_same": 0.02, "rewrite": 0.01}


def test_jev_request_shape_and_key_only_in_header(key_file):
    transport = FakeTransport(text=jev_body("inspect_tz"))
    jev_provider(key_file, transport).decide(make_request())
    call = transport.calls[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"]["authorization"] == f"Bearer {FAKE_KEY}"
    body = call["body"]
    assert body["model"] == "jev-latest"
    question = body["questions"]["next_step"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"retry_same", "inspect_tz", "rewrite", ABSTAIN}
    assert FAKE_KEY not in json.dumps(body)


def test_jev_abstention(key_file):
    result = jev_provider(key_file, FakeTransport(text=jev_body(ABSTAIN, 0.6))).decide(
        make_request()
    )
    assert result.ok and result.abstain and result.choice is None


def test_jev_abstain_rejected_when_not_allowed(key_file):
    result = jev_provider(key_file, FakeTransport(text=jev_body(ABSTAIN))).decide(
        make_request(allow_abstain=False)
    )
    assert not result.ok and result.fallback_reason == "jev_malformed_response"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        json.dumps({"model": "jev-1.13.0"}),
        json.dumps({"answers": {"other": {"choice": "retry_same", "confidence": 0.5}}}),
        jev_body("delete_repo"),
        jev_body("inspect_tz", 1.7),
        jev_body("inspect_tz", float("nan")),
        jev_body("inspect_tz", 0.5, {"inspect_tz": 0.5, "surprise": 0.5}),
        json.dumps({"answers": {"next_step": {"choice": "inspect_tz", "confidence": True}}}),
    ],
)
def test_jev_malformed_responses_fail_safe(key_file, text):
    result = jev_provider(key_file, FakeTransport(text=text)).decide(make_request())
    assert not result.ok
    assert result.choice is None
    assert result.fallback_reason == "jev_malformed_response"


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out"))],
)
def test_jev_timeout(key_file, error):
    result = jev_provider(key_file, FakeTransport(raises=error)).decide(make_request())
    assert not result.ok and result.fallback_reason == "jev_timeout"


def test_jev_unavailable_network(key_file):
    error = urllib.error.URLError(ConnectionRefusedError("refused"))
    result = jev_provider(key_file, FakeTransport(raises=error)).decide(make_request())
    assert not result.ok and result.fallback_reason == "jev_unavailable"


def test_jev_http_error_is_redacted(key_file):
    transport = FakeTransport(status=401, text=f'{{"error":"bad key {FAKE_KEY}"}}')
    result = jev_provider(key_file, transport).decide(make_request())
    assert result.fallback_reason == "jev_http_error"
    assert result.error and "401" in result.error and FAKE_KEY not in result.error


def test_jev_missing_key(tmp_path):
    transport = FakeTransport(text=jev_body("inspect_tz"))
    provider = JevProvider(
        "https://api.typesafe.ai/v1/systemone", key_file=str(tmp_path / "missing.key"),
        transport=transport,
    )  # fmt: skip
    result = provider.decide(make_request())
    assert result.fallback_reason == "jev_unavailable"
    assert transport.calls == []


def test_jev_env_key_used_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    transport = FakeTransport(text=jev_body("inspect_tz"))
    provider = JevProvider(
        "https://api.typesafe.ai/v1/systemone", key_file=str(tmp_path / "missing.key"),
        transport=transport,
    )  # fmt: skip
    assert provider.decide(make_request()).ok


def test_jev_refuses_non_https_endpoint(key_file):
    transport = FakeTransport(text=jev_body("inspect_tz"))
    provider = JevProvider("http://example.invalid/v1", key_file=str(key_file), transport=transport)
    assert provider.decide(make_request()).fallback_reason == "jev_misconfigured"
    assert transport.calls == []


# ---------------------------------------------------------------- NobodyWho


def test_grammar_admits_only_allowed_ids():
    assert grammar_for(("a", "b", ABSTAIN)) == 'root ::= "a" | "b" | "ABSTAIN"'
    assert ABSTAIN not in grammar_for(make_request(allow_abstain=False).allowed())


def test_sample_plan_is_deterministic_and_permutes_options():
    request = make_request()
    plan = plan_samples(request, 5, 100)
    assert plan == plan_samples(request, 5, 100)
    assert [s["seed"] for s in plan] == [100, 101, 102, 103, 104]
    orders = {
        tuple(line for line in s["prompt"].splitlines() if line.startswith("- ")) for s in plan
    }
    assert len(orders) > 1
    for sample in plan:
        options = [line for line in sample["prompt"].splitlines() if line.startswith("- ")]
        assert options[-1].startswith(f"- {ABSTAIN}")
        assert request.question in sample["prompt"]


def test_aggregate_votes_and_stability_proxy():
    result = aggregate(make_request(), ["inspect_tz", "inspect_tz", "rewrite"], 12, "m", {})
    assert result.ok and result.choice == "inspect_tz"
    assert result.votes == {"inspect_tz": 2, "rewrite": 1}
    assert result.confidence == pytest.approx(2 / 3)
    assert result.confidence_kind == SAMPLE_STABILITY
    assert result.distribution is None  # never presented as probabilities


def test_aggregate_tie_has_no_majority():
    result = aggregate(make_request(), ["inspect_tz", "rewrite"], 5, "m", {})
    assert not result.ok and result.choice is None
    assert result.fallback_reason == "local_no_majority"


def test_aggregate_abstain_majority():
    result = aggregate(make_request(), [ABSTAIN, ABSTAIN, "rewrite"], 5, "m", {})
    assert result.ok and result.abstain and result.choice is None


@pytest.mark.parametrize("outputs", [[], None, ["inspect_tz", "drop_database"], [ABSTAIN]])
def test_aggregate_rejects_malformed_samples(outputs):
    request = make_request(allow_abstain=False)
    result = aggregate(request, outputs, 5, "m", {})
    assert not result.ok and result.fallback_reason == "local_malformed_response"


def fake_local(tmp_path, runner, **kwargs):
    model = tmp_path / "Qwen_Qwen3-0.6B-Q4_K_M.gguf"
    model.write_bytes(b"GGUF")
    return NobodyWhoProvider(str(model), runner=runner, **kwargs)


def test_local_provider_runs_constrained_samples(tmp_path):
    jobs = []

    def runner(job, timeout):
        jobs.append(job)
        return {"outputs": ["inspect_tz"] * len(job["samples"]), "runtime": "nobodywho test"}

    result = fake_local(tmp_path, runner, samples=3).decide(make_request())
    assert result.ok and result.choice == "inspect_tz" and result.confidence == 1.0
    assert result.model == "Qwen_Qwen3-0.6B-Q4_K_M"
    assert result.details["quantization"] == "Q4_K_M"
    assert jobs[0]["grammar"] == grammar_for(make_request().allowed())
    assert len(jobs[0]["samples"]) == 3


def test_local_provider_timeout(tmp_path):
    def runner(job, timeout):
        raise subprocess.TimeoutExpired("worker", timeout)

    result = fake_local(tmp_path, runner).decide(make_request())
    assert not result.ok and result.fallback_reason == "local_timeout"


def test_local_provider_worker_error(tmp_path):
    result = fake_local(tmp_path, lambda job, t: {"error": "boom", "reason": "local_error"}).decide(
        make_request()
    )
    assert not result.ok and result.fallback_reason == "local_error"


@pytest.mark.parametrize("path", [None, "/nonexistent/model.gguf"])
def test_unavailable_local_model(path):
    called = []
    provider = NobodyWhoProvider(path, runner=lambda job, t: called.append(job) or {})
    result = provider.decide(make_request())
    assert not result.ok and result.fallback_reason == "local_model_unavailable"
    assert called == []


def test_worker_refuses_missing_model_without_downloading(tmp_path):
    output = worker_run({"model_path": str(tmp_path / "absent.gguf")})
    assert output["reason"] == "local_model_unavailable"


def _fake_worker(tmp_path, monkeypatch, body: str):
    script = tmp_path / "worker.py"
    script.write_text(body)
    monkeypatch.setattr(local_mod, "WORKER", script)
    return subprocess_runner(sys.executable)


def test_subprocess_runner_enforces_hard_timeout(tmp_path, monkeypatch):
    runner = _fake_worker(tmp_path, monkeypatch, "import time\ntime.sleep(30)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        runner({"samples": []}, 0.5)


def test_worker_env_is_guarded_and_has_no_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    runner = _fake_worker(
        tmp_path, monkeypatch,
        "import json, os, sys\nsys.stdin.read()\n"
        "print(json.dumps({'active': os.environ.get('DECISION_ROUTER_ACTIVE'),"
        " 'key': 'TYPESAFE_API_KEY' in os.environ}))\n",
    )  # fmt: skip
    assert runner({"samples": []}, 10) == {"active": "1", "key": False}


def test_worker_crash_is_contained(tmp_path, monkeypatch):
    runner = _fake_worker(tmp_path, monkeypatch, "import os\nos.abort()\n")
    output = runner({"samples": []}, 10)
    assert output["reason"] == "local_error"


# ---------------------------------------------------------------- GGUF identity


def _gguf(path, fields):
    def s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields))
    for key, (kind, value) in fields.items():
        body += s(key) + struct.pack("<I", kind)
        body += s(value) if kind == 8 else struct.pack("<I", value)
    path.write_bytes(body)


def test_gguf_identity_from_header(tmp_path):
    path = tmp_path / "model.gguf"
    _gguf(path, {"general.name": (8, "Qwen3 0.6B"), "general.file_type": (4, 15)})
    info = describe(path)
    assert info["name"] == "Qwen3 0.6B" and info["quantization"] == "Q4_K_M"


def test_gguf_identity_falls_back_to_file_name(tmp_path):
    path = tmp_path / "Qwen_Qwen3-0.6B-Q8_0.gguf"
    path.write_bytes(b"not a gguf")
    assert describe(path)["quantization"] == "Q8_0"
