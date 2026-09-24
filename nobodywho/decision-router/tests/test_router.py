from __future__ import annotations

import json

import pytest
from helpers import (
    FAKE_KEY,
    FakeTransport,
    StubProvider,
    failed,
    jev_body,
    jev_provider,
    make_request,
    make_router,
    ok,
)

from decision_router.contract import SAMPLE_STABILITY, DecisionResult
from decision_router.router import is_bypass_directive, is_meta_routing


def receipts(ledger):
    return ledger.tail(100)


def decision_of(outcome) -> DecisionResult:
    assert outcome.decision is not None
    return outcome.decision


def shadow_of(outcome) -> DecisionResult:
    assert outcome.shadow is not None
    return outcome.shadow


# ---------------------------------------------------------------- modes


def test_off_calls_nothing(ledger):
    jev, local = (
        StubProvider("jev", ok("jev", "a")),
        StubProvider("nobodywho", ok("nobodywho", "a")),
    )
    outcome = make_router(ledger, jev, local).route(make_request(), "off")
    assert outcome.follow is None and outcome.skipped == "mode_off"
    assert jev.requests == [] and local.requests == []
    assert receipts(ledger)[-1]["skipped"] == "mode_off"


def test_unknown_mode_calls_nothing(ledger):
    jev = StubProvider("jev", ok("jev", "inspect_tz"))
    outcome = make_router(ledger, jev).route(make_request(), "auto-escalate")
    assert outcome.skipped == "unknown_mode" and jev.requests == []


@pytest.mark.parametrize("mode, called", [("jev", "jev"), ("local", "nobodywho")])
def test_single_provider_modes(ledger, mode, called):
    jev = StubProvider("jev", ok("jev", "inspect_tz"))
    local = StubProvider("nobodywho", ok("nobodywho", "rewrite", kind=SAMPLE_STABILITY))
    outcome = make_router(ledger, jev, local).route(make_request(), mode)
    used, unused = (jev, local) if called == "jev" else (local, jev)
    assert len(used.requests) == 1 and unused.requests == []
    assert outcome.follow == ("inspect_tz" if called == "jev" else "rewrite")
    [receipt] = receipts(ledger)
    assert receipt["provider"] == called and receipt["role"] == "authoritative"


def test_local_mode_never_escalates_to_jev(ledger):
    jev = StubProvider("jev", ok("jev", "inspect_tz"))
    local = StubProvider("nobodywho", failed("nobodywho", "local_model_unavailable"))
    outcome = make_router(ledger, jev, local).route(make_request(), "local")
    assert outcome.follow is None and jev.requests == []


def test_provider_exception_becomes_data(ledger):
    jev = StubProvider("jev", raises=RuntimeError("bug"))
    outcome = make_router(ledger, jev).route(make_request(), "jev")
    assert outcome.follow is None and decision_of(outcome).fallback_reason == "jev_error"


def test_abstention_never_sets_follow(ledger):
    jev = StubProvider("jev", ok("jev", None, abstain=True))
    outcome = make_router(ledger, jev).route(make_request(), "jev")
    assert outcome.follow is None and decision_of(outcome).abstain
    assert "abstained" in str(outcome.note)


# ---------------------------------------------------------------- shadow


def test_shadow_jev_is_authoritative_and_disagreement_recorded(ledger):
    jev = StubProvider("jev", ok("jev", "inspect_tz"))
    local = StubProvider("nobodywho", ok("nobodywho", "rewrite", kind=SAMPLE_STABILITY))
    outcome = make_router(ledger, jev, local).route(make_request(), "shadow")
    assert outcome.follow == "inspect_tz"
    assert decision_of(outcome).provider == "jev" and shadow_of(outcome).choice == "rewrite"
    assert outcome.agreement is False
    authoritative, shadow = receipts(ledger)
    assert (authoritative["role"], authoritative["provider"]) == ("authoritative", "jev")
    assert (shadow["role"], shadow["provider"]) == ("shadow", "nobodywho")
    assert authoritative["agreement"] is shadow["agreement"] is False
    assert authoritative["request_id"] == shadow["request_id"]


def test_shadow_local_cannot_act_when_jev_fails(ledger):
    jev = StubProvider("jev", failed("jev", "jev_timeout"))
    local = StubProvider("nobodywho", ok("nobodywho", "rewrite", kind=SAMPLE_STABILITY))
    outcome = make_router(ledger, jev, local).route(make_request(), "shadow")
    assert outcome.follow is None
    assert decision_of(outcome).provider == "jev"
    assert shadow_of(outcome).choice == "rewrite"
    assert outcome.agreement is None


def test_shadow_local_cannot_override_jev_abstention(ledger):
    jev = StubProvider("jev", ok("jev", None, abstain=True))
    local = StubProvider("nobodywho", ok("nobodywho", "rewrite", kind=SAMPLE_STABILITY))
    outcome = make_router(ledger, jev, local).route(make_request(), "shadow")
    assert outcome.follow is None and outcome.agreement is False


def test_shadow_agreement(ledger):
    jev = StubProvider("jev", ok("jev", "inspect_tz"))
    local = StubProvider("nobodywho", ok("nobodywho", "inspect_tz", kind=SAMPLE_STABILITY))
    outcome = make_router(ledger, jev, local).route(make_request(), "shadow")
    assert outcome.follow == "inspect_tz" and outcome.agreement is True


def test_shadow_and_compare_send_identical_requests(ledger):
    for mode in ("shadow", "compare"):
        jev = StubProvider("jev", ok("jev", "inspect_tz"))
        local = StubProvider("nobodywho", ok("nobodywho", "inspect_tz", kind=SAMPLE_STABILITY))
        make_router(ledger, jev, local).route(make_request(metadata={"note": "x"}), mode)
        assert jev.requests[0].payload() == local.requests[0].payload()
        assert jev.requests[0].metadata == local.requests[0].metadata == {}
    hashes = {r["payload_sha256"] for r in receipts(ledger)}
    assert len(hashes) == 1


# ---------------------------------------------------------------- compare


def test_compare_disagreement_does_not_pick_one(ledger):
    jev = StubProvider("jev", ok("jev", "inspect_tz"))
    local = StubProvider("nobodywho", ok("nobodywho", "rewrite", kind=SAMPLE_STABILITY))
    outcome = make_router(ledger, jev, local).route(make_request(), "compare")
    assert outcome.follow is None and outcome.decision is None
    assert outcome.agreement is False
    assert "jev=inspect_tz" in str(outcome.note) and "nobodywho=rewrite" in str(outcome.note)
    data = outcome.to_dict()
    assert set(data["results"]) == {"jev", "nobodywho"}
    assert data["results"]["jev"]["confidence_kind"] == "calibrated_probability"
    assert data["results"]["nobodywho"]["confidence_kind"] == "sample_stability"
    assert [r["role"] for r in receipts(ledger)] == ["compare", "compare"]


def test_compare_agreement_and_failure(ledger):
    jev = StubProvider("jev", ok("jev", "inspect_tz"))
    local = StubProvider("nobodywho", ok("nobodywho", "inspect_tz", kind=SAMPLE_STABILITY))
    assert make_router(ledger, jev, local).route(make_request(), "compare").follow == "inspect_tz"
    local.result = failed("nobodywho", "local_timeout")
    outcome = make_router(ledger, jev, local).route(make_request(), "compare")
    assert outcome.follow is None and outcome.agreement is None
    assert "nobodywho: local_timeout" in str(outcome.note)


# ---------------------------------------------------------------- bypass / recursion


@pytest.mark.parametrize(
    "text, expected",
    [
        ("bypass decision router", True),
        ("Please BYPASS JEV for this one", True),
        ("bypass the decision-router and just do it", True),
        ("do not bypass jev", False),
        ("don't bypass decision router", False),
        ("fix the router bypass bug", False),
        (None, False),
    ],
)
def test_bypass_phrases(text, expected):
    assert is_bypass_directive(text) is expected


@pytest.mark.parametrize(
    "how",
    [
        {"bypass": True},
        {"user_directive": "bypass jev"},
        {"user_directive": "bypass decision router"},
    ],
)
def test_bypass_calls_nothing(ledger, how):
    jev, local = (
        StubProvider("jev", ok("jev", "a")),
        StubProvider("nobodywho", ok("nobodywho", "a")),
    )
    outcome = make_router(ledger, jev, local).route(make_request(), "shadow", **how)
    assert outcome.skipped == "bypass" and outcome.follow is None
    assert jev.requests == [] and local.requests == []


def test_bypass_via_env_and_metadata(ledger, monkeypatch):
    jev = StubProvider("jev", ok("jev", "a"))
    router = make_router(ledger, jev)
    request = make_request(metadata={"user_directive": "bypass JEV"})
    assert router.route(request, "jev").skipped == "bypass"
    monkeypatch.setenv("DECISION_ROUTER_BYPASS", "1")
    assert router.route(make_request(), "jev").skipped == "bypass"
    assert jev.requests == []


def test_nested_call_is_refused(ledger, monkeypatch):
    monkeypatch.setenv("DECISION_ROUTER_ACTIVE", "1")
    jev = StubProvider("jev", ok("jev", "a"))
    outcome = make_router(ledger, jev).route(make_request(), "jev")
    assert outcome.skipped == "recursive_call" and jev.requests == []


def test_provider_cannot_route_recursively(ledger):
    inner_outcomes = []

    class Recursing:
        name = "jev"

        def decide(self, request):
            inner_outcomes.append(router.route(request, "jev"))
            return ok("jev", "inspect_tz")

    router = make_router(ledger, Recursing())
    outcome = router.route(make_request(), "jev")
    assert outcome.follow == "inspect_tz"
    assert inner_outcomes[0].skipped == "recursive_call"


@pytest.mark.parametrize(
    "overrides",
    [
        {"question": "Should I use the decision router for this?"},
        {"question": "Is it worth it to call JEV here?"},
        {"choices": ["call_jev", "skip_router"]},
        {"choices": ["use_decision_router", "proceed"]},
    ],
)
def test_meta_routing_questions_are_refused(ledger, overrides):
    request = make_request(**overrides, criteria={})
    assert is_meta_routing(request)
    jev = StubProvider("jev", ok("jev", "a"))
    outcome = make_router(ledger, jev).route(request, "jev")
    assert outcome.skipped == "recursive_routing_question" and jev.requests == []


def test_ordinary_question_is_not_meta():
    assert not is_meta_routing(make_request(question="Which HTTP router library fits best?"))


# ---------------------------------------------------------------- secrets


def test_secrets_never_reach_logs_output_or_provider_state(ledger, key_file):
    leaky = (
        f"Tried with TYPESAFE_API_KEY={FAKE_KEY} and header Bearer abcdefghijklmnop123; "
        "token ghp_abcdefghijklmnopqrstuvwxyz0123 failed"
    )
    transport = FakeTransport(status=401, text=f"invalid key {FAKE_KEY}")
    router = make_router(ledger, jev_provider(key_file, transport))
    request = make_request(state=leaky, question=f"Retry with key {FAKE_KEY}?")
    outcome = router.route(request, "jev")

    sent = json.dumps(transport.calls[0]["body"])
    assert FAKE_KEY not in sent and "ghp_" not in sent and "abcdefghijklmnop123" not in sent
    assert transport.calls[0]["headers"]["authorization"].endswith(FAKE_KEY)

    log = ledger.path.read_text()
    assert FAKE_KEY not in log and "ghp_abcdef" not in log and leaky not in log
    assert FAKE_KEY not in json.dumps(outcome.to_dict())
    assert ledger.path.stat().st_mode & 0o077 == 0


def test_ledger_holds_hashes_not_state(ledger, key_file):
    transport = FakeTransport(text=jev_body("inspect_tz", 0.9, {"inspect_tz": 0.9, "rewrite": 0.1}))
    state = "def secret_algorithm():\n    return 42  # full source should never be logged"
    make_router(ledger, jev_provider(key_file, transport)).route(make_request(state=state), "jev")
    [receipt] = receipts(ledger)
    assert "secret_algorithm" not in json.dumps(receipt)
    assert len(receipt["state_sha256"]) == 64
    for field in (
        "ts", "request_id", "cwd", "question_id", "question", "choices", "mode",
        "provider", "model", "choice", "abstain", "confidence", "confidence_kind",
        "distribution", "latency_ms", "agreement", "error", "fallback_reason",
    ):  # fmt: skip
        assert field in receipt
    assert receipt["model"] == "jev-1.13.0"
    assert receipt["distribution"] == {"inspect_tz": 0.9, "rewrite": 0.1}
