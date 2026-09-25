"""Tev-style specialist coverage: identity, abstention, escalation and contracts.

Coverage map against the shared Local JEV interface checklist:
  1 discovery, 2 unavailable, 3 identity verified, 4 identity mismatch,
  5 thinking disabled, 6 fixed-option grammar, 7 first option, 8 second option,
  9 abstention, 10 malformed output, 11 logits if supported, 12 no fake
  probability, 13 generic escalation, 14 escalation provenance, 15 fallback
  disabled, 16 TypeSafe fallback enabled, 17 TypeSafe provenance, 18 physical
  attempts, 19 timeout, 20 runtime failure, 21 stdin JSON, 22 stdout JSON,
  23 caller=evolve, 24 existing callers, 25 no generic-primary masquerading,
  26 no secret leakage, 27 attempt bounds, 28 cold harness, 29 warm harness,
  30 response schema.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys

from helpers import FAKE_KEY, make_request

from decision_router import cli, specialist_benchmark
from decision_router import config as cfg
from decision_router import specialist as spec
from decision_router.acceptance import Policy
from decision_router.contract import ABSTAIN, ABSTAIN_REASONS, SAMPLE_STABILITY, DecisionResult
from decision_router.providers.nobodywho import NobodyWhoProvider, grammar_for
from decision_router.router import (
    SOURCE_ALL_ABSTAINED,
    SOURCE_ESCALATED_LOCAL,
    SOURCE_FAILED,
    SOURCE_FALLBACK_TYPESAFE,
    SOURCE_IDENTITY_MISMATCH,
    SOURCE_PRIMARY_LOCAL,
    SOURCE_PRIMARY_TEV,
    Router,
)

BENCH_CASES = [
    {
        "kind": "fixture",
        "expected": "inspect_timezone",
        "state": "A date-parser unit test failed twice with the same one-hour offset.",
        "question": "What should the agent do next?",
        "choices": {
            "retry_same": "Run the same test again unchanged.",
            "inspect_timezone": "Inspect how the parser handles timezone offsets.",
            "rewrite_parser": "Rewrite the parser from scratch.",
        },
    },
]


# ------------------------------------------------------------------ fixtures


def make_artifact(tmp_path, *, content=b"GGUF fake specialist artifact"):
    """A model file and the manifest that pins its digest and prompt contract."""
    model = tmp_path / "local-jev-tev-specialist-v1.gguf"
    model.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "classifier_id": spec.SPECIALIST_ID,
        "classifier_family": spec.SPECIALIST_FAMILY,
        "version": "1.0.0",
        "artifact": {"sha256": digest, "file": model.name},
        "training": {"pipeline": "specialist/train_lora.py"},
        "runtime_contract": {
            "system_prompt": spec.SPECIALIST_SYSTEM_PROMPT,
            "system_prompt_sha256": spec.system_prompt_sha256(),
            "renderer": spec.PROMPT_RENDERER,
            "thinking_enabled": False,
            "option_token_grammar": True,
        },
    }
    manifest_path = tmp_path / "specialist.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return model, digest, manifest_path


def classifier_block(manifest, **overrides):
    block = {
        "id": spec.SPECIALIST_ID,
        "family": spec.SPECIALIST_FAMILY,
        "kind": spec.SPECIALIST_KIND,
        "version": "1.0.0",
        "thinking": False,
        "option_token_grammar": True,
        "manifest": str(manifest),
        "on_identity_mismatch": "fail_closed",
    }
    block.update(overrides)
    return block


def tier_settings(model, digest, manifest, **overrides):
    settings = {
        "label": "tier1",
        "model_path": str(model),
        "model_sha256": digest,
        "persistent": False,
        "samples": 3,
        "temperature": 0.0,
        "seed": 1234,
        "n_ctx": 2048,
        "use_gpu": False,
        "timeout_s": 30,
        "classifier": classifier_block(manifest),
    }
    settings.update(overrides)
    return settings


def make_config(tiers, *, jev_enabled=False, retries=0):
    return {
        "mode": "local-first",
        "jev": {"enabled": jev_enabled, "endpoint": "https://example.invalid", "key_file": None},
        "local": {
            "persistent": False,
            "samples": 3,
            "temperature": 0.0,
            "seed": 1234,
            "n_ctx": 2048,
            "use_gpu": False,
            "timeout_s": 30,
        },
        "tiers": tiers,
        "acceptance": {
            "min_stability": 0.66,
            "min_margin": 1,
            "escalate_on_abstain": True,
            "max_local_retries": retries,
        },
    }


class FakeRunner:
    """A worker stand-in: returns queued worker results and records every job."""

    def __init__(self, *results, raises=None):
        self.results = list(results)
        self.raises = raises
        self.jobs: list[dict] = []

    def __call__(self, job, timeout_s):
        self.jobs.append(job)
        if self.raises is not None:
            raise self.raises
        return self.results[min(len(self.jobs) - 1, len(self.results) - 1)]


def worker(outputs, digest, planned=None):
    """A well-formed worker result as the specialist worker reports it."""
    planned = len(outputs) if planned is None else planned
    return {
        "outputs": outputs,
        "sample_ms": [1] * len(outputs),
        "planned": planned,
        "load_ms": 7,
        "model_reused": False,
        "gpu_layers": 0,
        "runtime": "nobodywho 3.0.0",
        "decode_steps": [1] * len(outputs),
        "context_tokens": [42] * len(outputs),
        "model_sha256": digest,
    }


class JevStub:
    name = "jev"

    def __init__(self, choice="inspect_tz"):
        self.choice = choice
        self.calls = 0

    def decide(self, request):
        self.calls += 1
        return DecisionResult(
            provider="jev", choice=self.choice, model="jev-1.13.0",
            confidence=0.9, confidence_kind="calibrated_probability",
        )  # fmt: skip


def build_router(config, ledger, t1, t2=None, jev=None):
    providers = {"nobodywho": t1}
    jev = jev or JevStub()
    if config["jev"].get("enabled"):
        providers["jev"] = jev
    tiers = {1: t1}
    if t2 is not None:
        tiers[2] = t2
    return Router(
        providers, ledger, (FAKE_KEY,), tiers=tiers,
        policy=Policy.from_config(config), jev_enabled=bool(config["jev"].get("enabled")),
    )  # fmt: skip


def specialist_provider(config, runner, tier="1"):
    return NobodyWhoProvider.for_tier(config, tier, runner=runner)


def generic_provider(tmp_path, outputs=None, *, name="generic.gguf", runner=None):
    """A generic tier with a real (empty) model file and a stand-in worker."""
    model = tmp_path / name
    model.write_bytes(b"GGUF generic model")
    return NobodyWhoProvider(
        model_path=str(model),
        runner=runner or FakeRunner(worker(outputs or ["retry_same"] * 3, None)),
    )


# ------------------------------------------------------- 1..8 identity + output


def test_1_discovery_reports_registered_and_unregistered(tmp_path):
    model, digest, manifest = make_artifact(tmp_path)
    unregistered = spec.discovery({"1": {"model_path": str(model)}})
    assert unregistered["classifierId"] == spec.SPECIALIST_ID
    assert unregistered["available"] is False
    assert unregistered["status"] == spec.UNAVAILABLE
    assert unregistered["optionLogits"] is None and unregistered["logitMargin"] is None

    found = spec.discovery({"1": tier_settings(model, digest, manifest)})
    assert found["available"] is True
    assert found["status"] == spec.VERIFIED
    assert found["verifiedBy"] == "file_digest"
    assert found["expectedSha256"] == found["observedSha256"] == digest
    assert found["classifierFamily"] == spec.SPECIALIST_FAMILY
    assert found["tier"] == 1


def test_2_specialist_unavailable_escalates_and_never_masquerades(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    model.unlink()  # the artifact is gone
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker(["retry_same"] * 3, digest))
    tier2 = generic_provider(tmp_path, ["inspect_tz"] * 3)
    router = build_router(config, ledger, specialist_provider(config, runner), tier2)
    outcome = router.route(make_request(), "local-first")

    assert outcome.follow == "inspect_tz" and outcome.tier == 2
    assert outcome.attempts[0]["status"] == "failed"
    assert outcome.attempts[0]["abstain_reason"] == "SPECIALIST_UNAVAILABLE"
    assert outcome.attempts[0]["role"] == "specialist"
    assert outcome.classifier["available"] is False
    assert runner.jobs == []  # an absent artifact never runs the model


def test_3_correct_specialist_identity_is_verified_on_the_loaded_file(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker(["inspect_tz"] * 3, digest))
    router = build_router(config, ledger, specialist_provider(config, runner))
    outcome = router.route(make_request(), "local-first")

    assert outcome.follow == "inspect_tz" and outcome.tier == 1
    assert outcome.decision_source == SOURCE_PRIMARY_TEV
    assert outcome.classifier["available"] is True
    assert outcome.classifier["status"] == spec.VERIFIED
    assert outcome.classifier["verifiedBy"] == "loaded_artifact_digest"
    assert outcome.decision.details["classifier"]["observedSha256"] == digest
    assert runner.jobs[0]["verify_sha256"] is True
    assert runner.jobs[0]["system_prompt"] == spec.SPECIALIST_SYSTEM_PROMPT


def test_4a_identity_mismatch_fails_closed_and_never_substitutes(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    wrong = hashlib.sha256(b"a different file").hexdigest()
    runner = FakeRunner(worker(["retry_same"] * 3, wrong))
    calls = {"2": 0}

    class Tier2:
        name = "nobodywho"
        model = "generic"

        def decide(self, request, attempt=0):
            calls["2"] += 1
            return DecisionResult(provider="nobodywho", choice="rewrite")

    router = build_router(config, ledger, specialist_provider(config, runner), Tier2())
    outcome = router.route(make_request(), "local-first")

    assert outcome.follow is None and outcome.tier is None
    assert outcome.decision_source == SOURCE_IDENTITY_MISMATCH
    assert outcome.decision.fallback_reason == "primary_identity_mismatch"
    assert outcome.decision.abstain_reason == "PRIMARY_IDENTITY_MISMATCH"
    assert outcome.decision.details["fail_closed"] is True
    assert calls["2"] == 0  # fail_closed: no tier substitution
    assert outcome.attempts[0]["role"] == "specialist"
    assert outcome.classifier["status"] == spec.MISMATCH


def test_4b_identity_mismatch_escalates_only_on_explicit_choice(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    settings = tier_settings(
        model, digest, manifest,
        classifier=classifier_block(manifest, on_identity_mismatch="escalate"),
    )  # fmt: skip
    config = make_config({"1": settings})
    wrong = hashlib.sha256(b"a different file").hexdigest()
    runner = FakeRunner(worker(["retry_same"] * 3, wrong))
    tier2 = generic_provider(tmp_path, ["inspect_tz"] * 3)
    router = build_router(config, ledger, specialist_provider(config, runner), tier2)
    outcome = router.route(make_request(), "local-first")

    assert outcome.follow == "inspect_tz" and outcome.tier == 2
    assert outcome.decision_source == SOURCE_ESCALATED_LOCAL
    assert outcome.attempts[0]["abstain_reason"] == "PRIMARY_IDENTITY_MISMATCH"
    assert outcome.attempts[1]["role"] == "generic"


def test_5_thinking_is_off_and_option_token_grammar_is_required(tmp_path):
    model, digest, manifest = make_artifact(tmp_path)
    settings = tier_settings(model, digest, manifest)
    assert spec.registration_for("1", settings) is not None

    thinking = dict(settings, classifier=classifier_block(manifest, thinking=True))
    assert spec.registration_for("1", thinking) is None
    no_grammar = dict(settings, classifier=classifier_block(manifest, option_token_grammar=False))
    assert spec.registration_for("1", no_grammar) is None

    status = spec.discovery({"1": settings})
    assert status["thinkingEnabled"] is False
    assert status["grammarConstrainedOptionTokens"] is True


def test_6_fixed_option_grammar_admits_only_the_allowed_ids(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker(["inspect_tz"] * 3, digest))
    router = build_router(config, ledger, specialist_provider(config, runner))
    request = make_request()
    router.route(request, "local-first")

    allowed = request.allowed()
    assert grammar_for(allowed) == "root ::= " + " | ".join(f'"{o}"' for o in allowed)
    job = runner.jobs[0]
    assert job["choices"] == list(allowed)
    assert job["grammar"] == grammar_for(allowed)


def test_7_valid_first_option_is_selected(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker(["retry_same"] * 3, digest))
    outcome = build_router(config, ledger, specialist_provider(config, runner)).route(
        make_request(), "local-first"
    )
    assert outcome.follow == "retry_same"
    assert outcome.decision.abstain is False and outcome.decision.abstain_reason is None


def test_8_valid_second_option_is_selected(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker(["inspect_tz"] * 3, digest))
    outcome = build_router(config, ledger, specialist_provider(config, runner)).route(
        make_request(), "local-first"
    )
    assert outcome.follow == "inspect_tz"


# --------------------------------------------- 9..14 abstention and escalation


def test_9_deliberate_abstain_is_a_protocol_outcome_never_an_option(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker([ABSTAIN] * 3, digest))
    outcome = build_router(config, ledger, specialist_provider(config, runner)).route(
        make_request(), "local-first"
    )
    assert outcome.follow is None
    assert outcome.decision_source == SOURCE_ALL_ABSTAINED
    assert outcome.decision.ok and outcome.decision.abstain
    first = outcome.attempts[0]
    assert first["abstain"] is True and first["choice"] is None
    assert first["abstain_reason"] == "ABSTAIN"
    assert first["status"] == "abstained"


def test_10_malformed_output_is_invalid_output_and_escalates(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker(["not_an_option"] * 3, digest))
    tier2 = generic_provider(tmp_path, ["inspect_tz"] * 3)
    outcome = build_router(config, ledger, specialist_provider(config, runner), tier2).route(
        make_request(), "local-first"
    )

    assert outcome.attempts[0]["status"] == "failed"
    assert outcome.attempts[0]["abstain_reason"] == "INVALID_OUTPUT"
    assert outcome.attempts[0]["escalation_reason"] == "invalid_output"
    assert outcome.follow == "inspect_tz" and outcome.tier == 2


def test_11_option_logits_are_null_with_a_recorded_reason(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker(["inspect_tz"] * 3, digest))
    outcome = build_router(config, ledger, specialist_provider(config, runner)).route(
        make_request(), "local-first"
    )
    details = outcome.decision.details
    assert details["optionLogits"] is None and details["logitMargin"] is None
    assert "unavailable" in details["logitEvidence"]
    assert outcome.classifier["optionLogits"] is None
    assert outcome.classifier["logitMargin"] is None
    assert "unavailable" in outcome.classifier["logitEvidence"]


def test_12_no_fake_probability_semantics(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker(["inspect_tz", "inspect_tz", ABSTAIN], digest))
    outcome = build_router(config, ledger, specialist_provider(config, runner)).route(
        make_request(), "local-first"
    )
    text = json.dumps(outcome.to_dict())
    for forbidden in ("pSupport", "confidenceProbability", "calibrated_probability"):
        assert forbidden not in text
    decision = outcome.decision
    assert decision.confidence_kind == SAMPLE_STABILITY
    assert decision.confidence == decision.votes["inspect_tz"] / 3
    assert set(decision.votes) <= {"inspect_tz", ABSTAIN, "retry_same", "rewrite"}


def test_13_specialist_abstain_escalates_to_the_generic_tier(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker([ABSTAIN] * 3, digest))
    tier2 = generic_provider(tmp_path, ["inspect_tz"] * 3)
    outcome = build_router(config, ledger, specialist_provider(config, runner), tier2).route(
        make_request(), "local-first"
    )
    assert outcome.follow == "inspect_tz" and outcome.tier == 2
    assert outcome.decision_source == SOURCE_ESCALATED_LOCAL


def test_14_escalation_provenance_is_explicit_per_attempt(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(worker([ABSTAIN] * 3, digest))
    tier2 = generic_provider(tmp_path, ["inspect_tz"] * 3)
    outcome = build_router(config, ledger, specialist_provider(config, runner), tier2).route(
        make_request(), "local-first"
    )
    first, second = outcome.attempts
    assert first["role"] == "specialist" and second["role"] == "generic"
    assert first["status"] == "abstained" and second["status"] == "accepted"
    assert first["escalation_reason"] == "insufficient_evidence"
    assert second["fallback_from"].endswith(first["model"])
    assert second["fallback_reason"] == "insufficient_evidence"
    for attempt in outcome.attempts:
        assert attempt["abstain_reason"] in ABSTAIN_REASONS or attempt["abstain_reason"] is None
        for key in (
            "tier",
            "provider",
            "role",
            "status",
            "latency_ms",
            "started_at",
            "completed_at",
        ):
            assert key in attempt


# --------------------------------------- 15..20 fallbacks, attempts, failures


def test_15_disabled_typesafe_fallback_is_never_called(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)}, jev_enabled=False)
    runner = FakeRunner(worker([ABSTAIN] * 3, digest))
    jev = JevStub()
    router = build_router(config, ledger, specialist_provider(config, runner), jev=jev)
    outcome = router.route(make_request(), "local-first")

    assert jev.calls == 0
    assert outcome.follow is None
    assert outcome.decision.ok and outcome.decision.abstain
    assert outcome.decision_source == SOURCE_ALL_ABSTAINED
    assert all(a["role"] != "typesafe" for a in outcome.attempts)


def test_16_typesafe_fallback_runs_once_after_the_local_tiers_fail(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)}, jev_enabled=True)
    runner = FakeRunner(worker([ABSTAIN] * 3, digest))
    jev = JevStub(choice="inspect_tz")
    router = build_router(config, ledger, specialist_provider(config, runner), jev=jev)
    outcome = router.route(make_request(), "local-first")

    assert jev.calls == 1
    assert outcome.follow == "inspect_tz" and outcome.tier == 3


def test_17_typesafe_attempt_carries_its_real_provenance(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)}, jev_enabled=True)
    runner = FakeRunner(worker([ABSTAIN] * 3, digest))
    router = build_router(config, ledger, specialist_provider(config, runner), jev=JevStub())
    outcome = router.route(make_request(), "local-first")

    last = outcome.attempts[-1]
    assert last["role"] == "typesafe" and last["tier"] == 3
    assert last["provider"] == "jev" and last["accepted"] is True
    assert outcome.decision_source == SOURCE_FALLBACK_TYPESAFE


def test_18_physical_attempts_are_counted_with_timestamps(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)}, jev_enabled=True, retries=1)
    # A tied vote is retryable: the specialist draws two physical attempts.
    runner = FakeRunner(
        worker(["inspect_tz", "retry_same", "rewrite"], digest),
        worker(["inspect_tz", "retry_same", "rewrite"], digest),
    )
    router = build_router(config, ledger, specialist_provider(config, runner), jev=JevStub())
    outcome = router.route(make_request(), "local-first")

    assert len(runner.jobs) == 2  # two physical specialist attempts
    assert [a["tier"] for a in outcome.attempts] == [1, 1, 3]
    assert [a["role"] for a in outcome.attempts] == ["specialist", "specialist", "typesafe"]
    for attempt in outcome.attempts:
        assert attempt["status"] in ("accepted", "abstained", "failed")
        assert attempt["started_at"] and attempt["completed_at"]
        assert isinstance(attempt["latency_ms"], int)


def test_19_timeout_is_an_explicit_timeout_abstention(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner(raises=subprocess.TimeoutExpired("worker", 30))
    tier2 = generic_provider(tmp_path, ["inspect_tz"] * 3)
    outcome = build_router(config, ledger, specialist_provider(config, runner), tier2).route(
        make_request(), "local-first"
    )
    assert outcome.attempts[0]["abstain_reason"] == "TIMEOUT"
    assert outcome.attempts[0]["escalation_reason"] == "timeout"
    assert outcome.follow == "inspect_tz" and outcome.tier == 2


def test_20_runtime_failure_is_an_explicit_runtime_failure(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    runner = FakeRunner({"error": "worker crashed", "reason": "local_error"})
    tier2 = generic_provider(tmp_path, ["inspect_tz"] * 3)
    outcome = build_router(config, ledger, specialist_provider(config, runner), tier2).route(
        make_request(), "local-first"
    )
    assert outcome.attempts[0]["abstain_reason"] == "RUNTIME_FAILURE"
    assert outcome.attempts[0]["escalation_reason"] == "worker_failure"
    assert outcome.follow == "inspect_tz" and outcome.tier == 2


# -------------------------------------------- 21..27 CLI contract and honesty


CLI_REQUEST = {
    "id": "evolve-1",
    "question_id": "next_step",
    "state": "Two identical retries failed with the same timezone offset.",
    "question": "What should the agent do next?",
    "choices": {"support": "Back this change.", "do_not_support": "Do not back it."},
    "allow_abstain": True,
    "risk": "low",
}


def write_cli_config(tmp_path):
    """A specialist tier that is registered but has no artifact yet."""
    model = tmp_path / "local-jev-tev-specialist-v1.gguf"
    manifest = tmp_path / "specialist.manifest.json"
    manifest.write_text(json.dumps({
        "classifier_id": spec.SPECIALIST_ID,
        "classifier_family": spec.SPECIALIST_FAMILY,
        "artifact": {"sha256": hashlib.sha256(b"missing").hexdigest()},
        "runtime_contract": {
            "system_prompt_sha256": spec.system_prompt_sha256(),
            "thinking_enabled": False,
            "option_token_grammar": True,
        },
    }))  # fmt: skip
    cfg.update({
        "mode": "local-first",
        "jev": {"enabled": False},
        "tiers": {"1": {
            "model_path": str(model), "persistent": False,
            "classifier": classifier_block(manifest),
        }},
    })  # fmt: skip


def run_cli(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr()


def test_21_stdin_json_request_contract(capsys, monkeypatch, tmp_path):
    write_cli_config(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(CLI_REQUEST)), raising=False)
    code, out = run_cli(capsys, "ask", "--caller", "evolve", "--mode", "local-first")
    data = json.loads(out.out)
    assert code == 0
    assert data["request_id"] == "evolve-1" and data["mode"] == "local-first"


def test_22_stdout_is_one_json_object_with_the_honest_contract(capsys, tmp_path):
    write_cli_config(tmp_path)
    code, out = run_cli(capsys, "ask", "--caller", "evolve", "--json", json.dumps(CLI_REQUEST))
    assert code == 0
    data = json.loads(out.out)  # exactly one JSON object on stdout
    assert data["follow"] is None
    assert data["decision_source"] == SOURCE_FAILED
    assert data["classifier"]["available"] is False
    assert data["classifier"]["optionLogits"] is None
    assert isinstance(data["attempts"], list) and data["attempts"]
    assert data["decision"]["error"]


def test_23_caller_evolve_is_recorded_and_answered(capsys, tmp_path):
    write_cli_config(tmp_path)
    assert cli.caller_name("evolve") == "evolve"
    code, out = run_cli(capsys, "ask", "--caller", "evolve", "--json", json.dumps(CLI_REQUEST))
    data = json.loads(out.out)
    assert code == 0 and data["request_id"] == CLI_REQUEST["id"]
    receipts = [
        json.loads(line) for line in (tmp_path / "state" / "ledger.jsonl").read_text().splitlines()
    ]
    assert any(r.get("caller") == "evolve" for r in receipts)


def test_24_existing_callers_keep_their_contract(capsys, tmp_path):
    write_cli_config(tmp_path)
    for caller in ("cline", "codex", "csmart", "freebuff", "opencode2"):
        data = json.loads(
            run_cli(capsys, "ask", f"--caller={caller}", "--json", json.dumps(CLI_REQUEST))[1].out
        )
        assert set(data) >= {"request_id", "mode", "follow", "decision", "tier", "attempts"}
        assert data["follow"] is None
        assert data["decision"]["error"]


def test_25_a_generic_primary_is_never_the_tev_specialist(tmp_path, ledger):
    config = make_config({"1": {}})
    runner = FakeRunner(worker(["inspect_tz"] * 3, None))
    tier1 = generic_provider(tmp_path, runner=runner)
    outcome = build_router(config, ledger, tier1).route(make_request(), "local-first")

    assert outcome.follow == "inspect_tz" and outcome.tier == 1
    assert outcome.decision_source == SOURCE_PRIMARY_LOCAL  # never PRIMARY_TEV
    assert outcome.classifier["available"] is False
    assert outcome.attempts[0]["role"] == "generic"


def test_26_no_secret_ever_reaches_the_response_or_ledger(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)}, jev_enabled=True)
    runner = FakeRunner(worker(["inspect_tz"] * 3, digest))
    outcome = build_router(
        config, ledger, specialist_provider(config, runner), jev=JevStub()
    ).route(make_request(), "local-first")
    payload = json.dumps(outcome.to_dict()) + ledger.tail(100).__repr__()
    assert FAKE_KEY not in payload and "tsk_" not in payload


def test_27_physical_attempts_are_bounded_by_the_retry_policy(tmp_path, ledger):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)}, jev_enabled=True, retries=3)
    tied = worker(["inspect_tz", "retry_same", "rewrite"], digest)
    runner1 = FakeRunner(tied)
    runner2 = FakeRunner(tied)
    tier2 = generic_provider(tmp_path, name="g2.gguf", runner=runner2)

    class FailingJev(JevStub):
        def decide(self, request):
            self.calls += 1
            return DecisionResult(provider="jev", error="down", fallback_reason="jev_failed")

    router = build_router(config, ledger, specialist_provider(config, runner1), tier2, FailingJev())
    outcome = router.route(make_request(), "local-first")

    ceiling = (1 + 3) * 2 + 1  # (1 + retries) per local tier, one fallback attempt
    assert len(runner1.jobs) + len(runner2.jobs) == 8
    assert len(outcome.attempts) == 9  # every physical attempt is one trace entry
    assert len(outcome.attempts) <= ceiling


# ------------------------------------------- 28..30 benchmarks and schema


def bench_setup(tmp_path, ledger, outputs):
    model, digest, manifest = make_artifact(tmp_path)
    config = make_config({"1": tier_settings(model, digest, manifest)})
    specialist = specialist_provider(config, FakeRunner(worker(outputs, digest)))
    generic = generic_provider(tmp_path)
    router = build_router(config, ledger, specialist, generic)
    return config, specialist, generic, router


def test_28_cold_benchmark_harness_reports_the_honest_distribution(tmp_path, ledger):
    config, specialist, generic, router = bench_setup(tmp_path, ledger, ["inspect_timezone"] * 3)
    report = specialist_benchmark.run(
        config, 0, cases=BENCH_CASES, specialist=specialist, generic=generic, router=router
    )
    cold = report["cold_specialist"]
    assert report["cases"] == 1
    assert cold["latency"]["n"] == 1
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
        assert isinstance(cold["latency"][key], (int, float))
    assert cold["valid_output_rate"] == 1.0
    assert cold["abstention_rate"] == 0.0
    assert cold["escalation_rate"] == 0.0
    assert cold["correct"] == 1 and cold["objective_cases"] == 1
    assert cold["model"] == "local-jev-tev-specialist-v1"


def test_29_warm_and_escalation_paths_are_measured_separately(tmp_path, ledger):
    config, specialist, generic, router = bench_setup(tmp_path, ledger, [ABSTAIN] * 3)
    report = specialist_benchmark.run(
        config, 0, cases=BENCH_CASES, specialist=specialist, generic=generic, router=router
    )
    warm = report["warm_specialist"]
    assert warm["latency"]["n"] == 1
    assert warm["abstention_rate"] == 1.0
    assert warm["escalation_rate"] == 1.0  # an abstention would escalate
    assert report["generic_baseline"]["escalation_rate"] == 0.0

    escalation = report["escalation_path"]
    assert escalation["latency"]["n"] == 1
    assert escalation["escalation_rate"] == 1.0  # the case routed past tier 1
    row = report["rows"]["escalation_path"][0]
    assert row["decision_source"] == SOURCE_ESCALATED_LOCAL
    assert row["tier"] == 2 and row["attempts"] == 2


def test_30_response_schema_carries_every_honest_field(tmp_path, ledger):
    *_, router = bench_setup(tmp_path, ledger, ["inspect_tz"] * 3)
    outcome = router.route(make_request(), "local-first")
    data = outcome.to_dict()
    assert set(data) == {
        "request_id", "mode", "follow", "decision", "tier",
        "attempts", "decision_source", "classifier",
    }  # fmt: skip
    classifier = data["classifier"]
    assert set(classifier) >= {
        "classifierId", "classifierFamily", "classifierKind", "classifierVersion",
        "thinkingEnabled", "grammarConstrainedOptionTokens", "available", "status",
        "reasons", "expectedSha256", "observedSha256", "verifiedBy", "tier",
        "optionLogits", "logitMargin", "logitEvidence",
    }  # fmt: skip
    assert classifier["classifierId"] == "local-jev-tev-specialist-v1"
    assert classifier["classifierFamily"] == "tev-style-specialist"
    assert classifier["thinkingEnabled"] is False
    attempt = data["attempts"][0]
    assert set(attempt) >= {
        "tier", "provider", "role", "status", "model", "choice", "abstain",
        "abstain_reason", "confidence", "confidence_kind", "latency_ms",
        "accepted", "escalation_reason", "fallback_from", "fallback_reason",
        "runtime", "started_at", "completed_at",
    }  # fmt: skip
    decision = data["decision"]
    assert set(decision) >= {
        "provider", "choice", "abstain", "abstain_reason", "model", "latency_ms",
        "confidence", "confidence_kind", "votes", "samples", "fallback_reason",
        "error", "details",
    }  # fmt: skip
