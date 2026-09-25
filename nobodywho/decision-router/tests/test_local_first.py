from __future__ import annotations

import json

import pytest
from helpers import FAKE_KEY, FakeTransport, jev_body, jev_provider, make_request

from decision_router import cli
from decision_router import config as cfg
from decision_router.acceptance import ESCALATION_REASONS, Policy, judge
from decision_router.contract import ABSTAIN, SAMPLE_STABILITY, DecisionResult
from decision_router.providers.nobodywho import NobodyWhoProvider, PersistentWorker
from decision_router.router import SOURCE_ALL_ABSTAINED, SOURCE_FAILED, Router


class Tier:
    """A local tier stub: returns queued results and records every call."""

    def __init__(self, name: str, *results: DecisionResult, raises=None):
        self.name = "nobodywho"
        self.model = name
        self.results = list(results)
        self.raises = raises
        self.calls: list[int] = []

    def decide(self, request, attempt: int = 0):
        self.calls.append(attempt)
        if self.raises:
            raise self.raises
        result = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        result.model = self.model
        return result


def voted(votes: dict[str, int]) -> DecisionResult:
    total = sum(votes.values())
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    winner, count = ranked[0]
    tie = len(ranked) > 1 and ranked[1][1] == count
    return DecisionResult(
        provider="nobodywho",
        choice=None if tie or winner == ABSTAIN else winner,
        abstain=not tie and winner == ABSTAIN,
        confidence=count / total,
        confidence_kind=SAMPLE_STABILITY,
        votes=votes,
        samples=total,
        fallback_reason="local_no_majority" if tie else None,
    )


def broken(reason: str) -> DecisionResult:
    return DecisionResult.failure("nobodywho", "broken", reason)


class Jev:
    def __init__(self, result: DecisionResult | None = None):
        self.name = "jev"
        self.result = result or DecisionResult(
            provider="jev", choice="inspect_tz", model="jev-1.13.0",
            confidence=0.9, confidence_kind="calibrated_probability",
        )  # fmt: skip
        self.calls = 0

    def decide(self, request):
        self.calls += 1
        return self.result


def router(ledger, t1, t2, jev=None, *, jev_enabled=True, policy=None) -> Router:
    return Router(
        {"jev": jev or Jev()}, ledger, (FAKE_KEY,),
        tiers={1: t1, 2: t2}, policy=policy or Policy(), jev_enabled=jev_enabled,
    )  # fmt: skip


def receipts(ledger):
    return ledger.tail(100)


# ---------------------------------------------------------------- tiers


def test_tier1_success_keeps_everything_else_dormant(ledger):
    t1, t2, jev = Tier("q9b", voted({"inspect_tz": 3})), Tier("q27b"), Jev()
    r = router(ledger, t1, t2, jev)
    out = r.route(make_request(), "local-first", caller="cline")
    assert out.follow == "inspect_tz" and out.tier == 1
    assert t1.calls == [0] and t2.calls == [] and jev.calls == 0 and r.jev_calls == 0
    [receipt] = receipts(ledger)
    assert receipt["tier"] == 1 and receipt["accepted"] is True and receipt["success"] is True
    assert receipt["caller"] == "cline" and receipt["confidence_kind"] == "sample_stability"


@pytest.mark.parametrize(
    "tier1, reason",
    [
        (voted({ABSTAIN: 2, "rewrite": 1}), "abstain"),
        (voted({ABSTAIN: 3}), "insufficient_evidence"),
        (voted({"inspect_tz": 1, "rewrite": 1, "retry_same": 1}), "no_valid_choice"),
        (broken("local_timeout"), "timeout"),
        (broken("local_model_unavailable"), "model_unavailable"),
        (broken("local_malformed_response"), "invalid_output"),
        (broken("local_error"), "worker_failure"),
    ],
)
def test_tier2_runs_only_after_tier1_fails_and_jev_stays_dormant(ledger, tier1, reason):
    t1, t2, jev = Tier("q9b", tier1), Tier("q27b", voted({"rewrite": 3})), Jev()
    out = router(ledger, t1, t2, jev).route(make_request(), "local-first")
    assert out.follow == "rewrite" and out.tier == 2 and jev.calls == 0
    first, second = receipts(ledger)
    assert first["escalation_reason"] == reason and first["accepted"] is False
    assert second["fallback_from"] == "nobodywho/q9b" and second["fallback_reason"] == reason


def test_jev_runs_exactly_once_after_both_locals_fail(ledger):
    t1 = Tier("q9b", broken("local_timeout"))
    t2 = Tier("q27b", voted({ABSTAIN: 2, "rewrite": 1}))
    jev = Jev()
    r = router(ledger, t1, t2, jev)
    out = r.route(make_request(), "local-first")
    assert out.follow == "inspect_tz" and out.tier == 3
    assert jev.calls == 1 and r.jev_calls == 1
    assert out.decision and out.decision.confidence_kind == "calibrated_probability"
    tier3 = receipts(ledger)[-1]
    assert tier3["provider"] == "jev" and tier3["tier"] == 3
    assert tier3["fallback_from"] == "nobodywho/q27b" and tier3["fallback_reason"] == "abstain"
    assert [a["tier"] for a in out.to_dict()["attempts"]] == [1, 2, 3]


def test_jev_disabled_is_a_hard_prohibition(ledger):
    t1, t2, jev = Tier("q9b", broken("local_error")), Tier("q27b", broken("local_timeout")), Jev()
    r = router(ledger, t1, t2, jev, jev_enabled=False)
    out = r.route(make_request(), "local-first")
    assert out.follow is None and out.tier is None
    assert jev.calls == 0 and r.jev_calls == 0
    assert "JEV is disabled" in str(out.note)
    assert [r["provider_reason"] for r in receipts(ledger)] == ["local_error", "local_timeout"]
    decision = out.decision
    assert decision is not None
    assert out.decision_source == SOURCE_FAILED and decision.error == "broken"


def test_all_valid_local_abstentions_are_a_final_abstention(ledger):
    t1 = Tier("specialist", voted({ABSTAIN: 3}))
    t2 = Tier("9b", voted({ABSTAIN: 3}))
    jev = Jev()
    r = router(ledger, t1, t2, jev, jev_enabled=False)
    out = r.route(make_request(), "local-first", caller="codex")
    assert out.follow is None and out.tier is None
    decision = out.decision
    assert decision is not None
    assert decision.ok and decision.abstain and decision.error is None
    assert out.decision_source == SOURCE_ALL_ABSTAINED
    assert [a["status"] for a in out.attempts] == ["abstained", "abstained"]
    assert [r["model"] for r in receipts(ledger)] == ["specialist", "9b"]
    assert jev.calls == r.jev_calls == 0


@pytest.mark.parametrize("first_fails", [True, False])
def test_mixed_abstention_and_failure_is_not_clean_abstention(ledger, first_fails):
    abstain = voted({ABSTAIN: 3})
    failure = broken("local_timeout")
    t1, t2 = (failure, abstain) if first_fails else (abstain, failure)
    out = router(ledger, Tier("specialist", t1), Tier("9b", t2), jev_enabled=False).route(
        make_request(), "local-first"
    )
    assert out.decision_source == SOURCE_FAILED
    decision = out.decision
    assert decision is not None
    assert decision.error == "broken" and not decision.abstain


@pytest.mark.parametrize("mode", ["jev", "shadow", "compare"])
def test_jev_disabled_applies_to_every_mode(ledger, mode):
    jev = Jev()
    r = Router(
        {"jev": jev, "nobodywho": Tier("q4b", voted({"inspect_tz": 3}))}, ledger,
        jev_enabled=False,
    )  # fmt: skip
    r.route(make_request(), mode)
    assert jev.calls == 0 and r.jev_calls == 0


def test_jev_abstention_at_tier3_is_final(ledger):
    jev = Jev(DecisionResult(provider="jev", abstain=True, confidence=0.7,
                             confidence_kind="calibrated_probability"))  # fmt: skip
    out = router(ledger, Tier("a", broken("local_error")), Tier("b", broken("local_error")), jev
                 ).route(make_request(), "local-first")  # fmt: skip
    assert out.follow is None and out.tier == 3 and "abstained" in str(out.note)


def test_malformed_jev_output_at_tier3_fails_safe(ledger, key_file):
    jev = jev_provider(key_file, FakeTransport(text=jev_body("drop_table")))
    out = router(ledger, Tier("a", broken("local_error")), Tier("b", broken("local_error")), jev
                 ).route(make_request(), "local-first")  # fmt: skip
    assert out.follow is None and out.tier is None
    assert receipts(ledger)[-1]["provider_reason"] == "jev_malformed_response"


def test_real_jev_provider_is_called_once_with_key_only_in_header(ledger, key_file):
    transport = FakeTransport(text=jev_body("inspect_tz"))
    jev = jev_provider(key_file, transport)
    router(ledger, Tier("a", broken("local_error")), Tier("b", broken("local_error")), jev
           ).route(make_request(state=f"key {FAKE_KEY}"), "local-first")  # fmt: skip
    assert len(transport.calls) == 1
    assert FAKE_KEY not in json.dumps(transport.calls[0]["body"])
    assert FAKE_KEY not in ledger.path.read_text()


def test_provider_exception_is_provider_error(ledger):
    t1 = Tier("q9b", raises=RuntimeError(f"boom {FAKE_KEY}"))
    t2 = Tier("q27b", voted({"rewrite": 3}))
    out = router(ledger, t1, t2).route(make_request(), "local-first")
    assert out.tier == 2 and receipts(ledger)[0]["escalation_reason"] == "provider_error"
    assert FAKE_KEY not in ledger.path.read_text()


def test_no_tiers_never_turns_into_jev_first(ledger):
    jev = Jev()
    r = Router({"jev": jev}, ledger, tiers={})
    out = r.route(make_request(), "local-first")
    assert out.follow is None and jev.calls == 0


def test_each_provider_called_at_most_once_without_retries(ledger):
    t1, t2, jev = Tier("a", broken("local_error")), Tier("b", broken("local_error")), Jev()
    router(ledger, t1, t2, jev).route(make_request(), "local-first")
    assert t1.calls == [0] and t2.calls == [0] and jev.calls == 1


# ---------------------------------------------------------------- acceptance policy


def test_default_policy_accepts_two_of_three():
    assert judge(voted({"inspect_tz": 2, "rewrite": 1}), Policy()) == (True, None)


def test_stricter_policy_rejects_two_of_three():
    policy = Policy(min_stability=0.9)
    assert judge(voted({"inspect_tz": 2, "rewrite": 1}), policy) == (
        False,
        "stability_below_threshold",
    )


def test_margin_gate():
    policy = Policy(min_stability=0.0, min_margin=2)
    assert judge(voted({"inspect_tz": 3, "rewrite": 2}), policy)[1] == "stability_below_threshold"
    assert judge(voted({"inspect_tz": 4, "rewrite": 1}), policy) == (True, None)


def test_abstain_can_be_accepted_without_escalation(ledger):
    t1, t2, jev = Tier("a", voted({ABSTAIN: 3})), Tier("b"), Jev()
    policy = Policy(escalate_on_abstain=False)
    out = router(ledger, t1, t2, jev, policy=policy).route(make_request(), "local-first")
    assert out.tier == 1 and out.follow is None and t2.calls == [] and jev.calls == 0


def test_retries_draw_fresh_seeds_but_not_for_broken_models(ledger):
    shaky = Tier("a", voted({"x_a": 1, "x_b": 1, "x_c": 1}), voted({"inspect_tz": 3}))
    policy = Policy(max_local_retries=1)
    out = router(ledger, shaky, Tier("b"), policy=policy).route(make_request(), "local-first")
    assert shaky.calls == [0, 1] and out.tier == 1
    missing = Tier("a", broken("local_model_unavailable"))
    router(ledger, missing, Tier("b", voted({"rewrite": 3})), policy=policy).route(
        make_request(), "local-first"
    )
    assert missing.calls == [0]


def test_policy_validation():
    with pytest.raises(ValueError):
        Policy.from_config({"acceptance": {"min_stability": 1.5}})
    with pytest.raises(ValueError):
        Policy.from_config({"acceptance": {"max_local_retries": 9}})


def test_escalation_vocabulary_is_closed():
    assert set(ESCALATION_REASONS) == {
        "provider_error", "timeout", "invalid_output", "abstain", "no_valid_choice",
        "worker_failure", "model_unavailable", "stability_below_threshold",
        "insufficient_evidence",
    }  # fmt: skip


# ---------------------------------------------------------------- bypass / config / cli


@pytest.mark.parametrize("directive", ["bypass jev", "bypass decision router"])
def test_bypass_skips_every_tier(ledger, directive):
    t1, t2, jev = Tier("a", voted({"x_a": 3})), Tier("b"), Jev()
    out = router(ledger, t1, t2, jev).route(make_request(), "local-first", user_directive=directive)
    assert out.skipped == "bypass" and t1.calls == [] and t2.calls == [] and jev.calls == 0


def test_tiers_never_inherit_the_local_model():
    config = cfg.load()
    config["local"]["model_path"] = "/models/q4b.gguf"
    assert cfg.tier_settings(config, "1").get("model_path") is None
    config["tiers"]["1"]["model_path"] = "/models/q9b.gguf"
    assert cfg.tier_settings(config, "1")["model_path"] == "/models/q9b.gguf"
    assert cfg.tier_settings(config, "2")["idle_timeout_s"] == 120


def test_each_tier_has_its_own_worker():
    config = cfg.load()
    names = set()
    for tier in cfg.TIER_NAMES:
        runner = NobodyWhoProvider.for_tier(config, tier).runner
        assert isinstance(runner, PersistentWorker)
        names.add(runner.name)
    assert names == {"tier1-worker", "tier2-worker"}


def test_cli_jev_switch_and_local_first_without_models_makes_no_remote_call(capsys, key_file):
    cfg.update({"jev": {"key_file": str(key_file)}})
    assert cli.main(["jev", "disable"]) == 0
    assert "disabled" in capsys.readouterr().out
    cli.main(["provider", "local-first"])
    shown = capsys.readouterr().out
    assert "mode: local-first" in shown and "D3:" in shown and "P3:" in shown
    assert "disabled" in shown
    request = {"question": "Which?", "choices": ["x_a", "x_b"], "state": "evidence"}
    cli.main(["ask", "--caller", "codex", "--json", json.dumps(request)])
    data = json.loads(capsys.readouterr().out)
    assert data["follow"] is None and data["tier"] is None
    assert [a["tier"] for a in data["attempts"]] == [1, 2]
    assert data["attempts"][-1]["escalation_reason"] == "model_unavailable"
    cli.main(["stats"])
    stats = json.loads(capsys.readouterr().out)
    assert stats["ask"]["jev_calls"] == 0 and stats["ask"]["by_caller"]["codex"]["decisions"] == 1
    assert stats["jev_calls_total"] == 0
    cli.main(["jev", "enable"])
    assert "dormant" in capsys.readouterr().out


def test_caller_name_is_sanitized(monkeypatch):
    assert cli.caller_name("Cline") == "cline"
    assert cli.caller_name("bad name; rm -rf") == "unknown"
    monkeypatch.setenv("DECISION_ROUTER_CALLER", "claude")
    assert cli.caller_name(None) == "claude"
