"""Provider-neutral routing: one request, one mode, a normalized outcome.

Modes
  off      no provider is called
  jev      TypeSafe JEV only
  local    NobodyWho only (no inference API cost)
  shadow   JEV is authoritative; NobodyWho sees the identical request, its
           answer is recorded but can never become the decision
  compare  both run; the outcome is a comparison, and a disagreement never
           resolves to either provider
  local-first
           production mode: tier 1 (local), then tier 2 (larger local), then
           tier 3 (JEV). A tier runs only when every earlier tier failed the
           acceptance policy for a reason in acceptance.ESCALATION_REASONS,
           so JEV stays dormant whenever a local tier answers acceptably.

`jev.enabled = false` is a hard switch: JEV is never called, in any mode.

The router only advises. It never executes anything, and its failures are
returned as data (`follow: null`) instead of raised.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from .acceptance import Policy, judge
from .config import MODES
from .contract import DecisionRequest, DecisionResult
from .ledger import Ledger
from .sanitize import redact, redact_value


class Provider(Protocol):
    name: str

    def decide(self, request: DecisionRequest) -> DecisionResult: ...


class TierProvider(Protocol):
    """A local-first tier: `attempt` > 0 asks for fresh seeds on a retry."""

    name: str

    def decide(self, request: DecisionRequest, attempt: int = 0) -> DecisionResult: ...


BYPASS_PHRASES = re.compile(r"(?i)\bbypass\s+(?:the\s+)?(?:decision[\s-]*router|jev)\b")
_META_QUESTION = re.compile(
    r"(?i)\b(?:(?:use|call|ask|consult|query|invoke|skip|bypass)\s+(?:the\s+)?"
    r"(?:decision[\s-]*router|router|jev|decision\s+(?:model|provider))"
    r"|decision[\s-]*router)\b"
)
_META_CHOICE = re.compile(
    r"^(?:use|call|ask|consult|query|invoke|skip|bypass)_(?:the_)?(?:decision_)?(?:router|jev)$"
)


_NEGATED = re.compile(r"(?i)(?:\bnot|\bnever|n't|\bno)\s*$")


def is_bypass_directive(text: str | None) -> bool:
    """True for the user instructions "bypass decision router" / "bypass jev"."""
    text = text or ""
    for match in BYPASS_PHRASES.finditer(text):
        if not _NEGATED.search(text[max(0, match.start() - 12) : match.start()]):
            return True
    return False


def is_meta_routing(request: DecisionRequest) -> bool:
    """True when the question asks whether/how to use the router itself."""
    return bool(_META_QUESTION.search(request.question)) or any(
        _META_CHOICE.match(c) for c in request.choices
    )


def sanitize_request(request: DecisionRequest, literals: tuple[str, ...]) -> DecisionRequest:
    """The same request with secrets removed from everything a provider sees."""
    return DecisionRequest(
        id=request.id,
        question_id=request.question_id,
        question=redact(request.question, literals),
        choices=request.choices,
        state=redact_value(request.state, literals),
        criteria={k: redact(v, literals) for k, v in request.criteria.items()},
        allow_abstain=request.allow_abstain,
        risk=request.risk,
        metadata={},  # metadata is router-only; providers never see it
    )


def agree(a: DecisionResult | None, b: DecisionResult | None) -> bool | None:
    if a is None or b is None or not a.ok or not b.ok:
        return None
    return (a.choice, a.abstain) == (b.choice, b.abstain)


@dataclass
class Outcome:
    request_id: str
    mode: str
    follow: str | None = None
    decision: DecisionResult | None = None
    shadow: DecisionResult | None = None
    results: dict[str, DecisionResult] = field(default_factory=dict)
    agreement: bool | None = None
    skipped: str | None = None
    note: str | None = None
    tier: int | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "request_id": self.request_id,
            "mode": self.mode,
            "follow": self.follow,
            "decision": self.decision.to_dict() if self.decision else None,
        }
        if self.shadow is not None:
            data["shadow"] = self.shadow.to_dict()
        if self.results:
            data["results"] = {k: v.to_dict() for k, v in self.results.items()}
        if self.mode in ("shadow", "compare"):
            data["agreement"] = self.agreement
        if self.mode == "local-first":
            data["tier"] = self.tier
            data["attempts"] = self.attempts
        if self.skipped:
            data["skipped"] = self.skipped
        if self.note:
            data["note"] = self.note
        return data


class Router:
    def __init__(
        self,
        providers: dict[str, Provider],
        ledger: Ledger,
        secret_literals: tuple[str, ...] = (),
        *,
        tiers: dict[int, TierProvider] | None = None,
        policy: Policy | None = None,
        jev_enabled: bool = True,
    ) -> None:
        self.providers = providers
        self.ledger = ledger
        self.secret_literals = secret_literals
        self.tiers = tiers or {}
        self.policy = policy or Policy()
        self.jev_enabled = jev_enabled
        self.jev_calls = 0

    def route(
        self,
        request: DecisionRequest,
        mode: str,
        *,
        bypass: bool = False,
        user_directive: str | None = None,
        caller: str | None = None,
    ) -> Outcome:
        mode = (mode or "").lower()
        self.ledger.caller = caller or self.ledger.caller
        skip = self._skip_reason(request, mode, bypass, user_directive)
        request = sanitize_request(request, self.secret_literals)
        if skip:
            self.ledger.write([self.ledger.receipt(request, mode, "none", None, skipped=skip)])
            return Outcome(request.id, mode, skipped=skip, note=_SKIP_NOTES.get(skip))

        # The router is the providers' only caller: nested routing is refused.
        previous = os.environ.get("DECISION_ROUTER_ACTIVE")
        os.environ["DECISION_ROUTER_ACTIVE"] = "1"
        try:
            if mode == "jev":
                return self._single(request, mode, "jev")
            if mode == "local":
                return self._single(request, mode, "nobodywho")
            if mode == "shadow":
                return self._shadow(request)
            if mode == "local-first":
                return self._local_first(request)
            return self._compare(request)
        finally:
            if previous is None:
                os.environ.pop("DECISION_ROUTER_ACTIVE", None)
            else:
                os.environ["DECISION_ROUTER_ACTIVE"] = previous

    def _skip_reason(
        self, request: DecisionRequest, mode: str, bypass: bool, directive: str | None
    ) -> str | None:
        if os.environ.get("DECISION_ROUTER_ACTIVE") == "1":
            return "recursive_call"
        if is_meta_routing(request):
            return "recursive_routing_question"
        if (
            bypass
            or os.environ.get("DECISION_ROUTER_BYPASS", "").strip() in ("1", "true", "yes")
            or is_bypass_directive(directive)
            or is_bypass_directive(str(request.metadata.get("user_directive", "")))
        ):
            return "bypass"
        if mode == "off":
            return "mode_off"
        if mode not in MODES:
            return "unknown_mode"
        return None

    def _call(self, name: str, request: DecisionRequest) -> DecisionResult:
        if name == "jev":
            if not self.jev_enabled:
                return DecisionResult.failure("jev", "JEV is disabled", "jev_disabled")
            self.jev_calls += 1
        provider = self.providers.get(name)
        if provider is None:
            return DecisionResult.failure(name, "provider not configured", f"{name}_unavailable")
        try:
            return provider.decide(request)
        except Exception as error:  # noqa: BLE001 - a provider bug must not reach the agent
            return DecisionResult.failure(
                name, redact(f"{type(error).__name__}: {error}", self.secret_literals)[:200],
                f"{name}_error",
            )  # fmt: skip

    def _both(self, request: DecisionRequest) -> tuple[DecisionResult, DecisionResult]:
        """JEV and NobodyWho on the identical sanitized request, concurrently."""
        with ThreadPoolExecutor(max_workers=1) as pool:
            local = pool.submit(self._call, "nobodywho", request)
            jev = self._call("jev", request)
            return jev, local.result()

    def _single(self, request: DecisionRequest, mode: str, name: str) -> Outcome:
        result = self._call(name, request)
        self.ledger.write([self.ledger.receipt(request, mode, "authoritative", result)])
        return Outcome(
            request.id,
            mode,
            follow=result.choice if result.ok else None,
            decision=result,
            note=_result_note(result),
        )

    def _shadow(self, request: DecisionRequest) -> Outcome:
        jev, local = self._both(request)
        agreement = agree(jev, local)
        self.ledger.write(
            [
                self.ledger.receipt(request, "shadow", "authoritative", jev, agreement),
                self.ledger.receipt(request, "shadow", "shadow", local, agreement),
            ]
        )
        # Only JEV can set `follow`; the local answer is informational, even when JEV fails.
        return Outcome(
            request.id,
            "shadow",
            follow=jev.choice if jev.ok else None,
            decision=jev,
            shadow=local,
            agreement=agreement,
            note=_result_note(jev),
        )

    def _compare(self, request: DecisionRequest) -> Outcome:
        jev, local = self._both(request)
        agreement = agree(jev, local)
        self.ledger.write(
            [
                self.ledger.receipt(request, "compare", "compare", jev, agreement),
                self.ledger.receipt(request, "compare", "compare", local, agreement),
            ]
        )
        if agreement:
            follow, note = jev.choice, "providers agree"
        elif agreement is False:
            follow = None
            note = (
                f"providers disagree (jev={_label(jev)}, nobodywho={_label(local)}); "
                "compare mode never picks one; use your own judgement"
            )
        else:
            missing = ", ".join(
                f"{r.provider}: {r.fallback_reason}" for r in (jev, local) if not r.ok
            )
            follow, note = None, f"comparison incomplete ({missing}); use your own judgement"
        return Outcome(
            request.id,
            "compare",
            follow=follow,
            results={"jev": jev, "nobodywho": local},
            agreement=agreement,
            note=note,
        )

    def _local_first(self, request: DecisionRequest) -> Outcome:
        """Tier 1, then tier 2, then JEV; stop at the first acceptable answer."""
        attempts: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        previous: tuple[str, str] | None = None  # (label, escalation reason)

        def record(tier: int, result: DecisionResult, accepted: bool, why: str | None) -> None:
            attempt = {
                "tier": tier,
                "provider": result.provider,
                "model": result.model,
                "choice": result.choice,
                "abstain": result.abstain,
                "confidence": result.confidence,
                "confidence_kind": result.confidence_kind,
                "latency_ms": result.latency_ms,
                "accepted": accepted,
                "escalation_reason": why,
                "fallback_from": previous[0] if previous else None,
                "fallback_reason": previous[1] if previous else None,
            }
            attempts.append(attempt)
            receipts.append(
                self.ledger.receipt(
                    request, "local-first", f"tier{tier}", result,
                    tier=tier, accepted=accepted, escalation_reason=why,
                    fallback_from=attempt["fallback_from"],
                    fallback_reason=attempt["fallback_reason"],
                )
            )  # fmt: skip

        final: DecisionResult | None = None
        final_tier: int | None = None
        if not self.tiers:  # never let a misconfiguration turn local-first into JEV-first
            result = DecisionResult.failure("router", "no local tiers configured", "no_local_tiers")
            self.ledger.write([self.ledger.receipt(request, "local-first", "none", result)])
            return Outcome(
                request.id, "local-first", decision=result, note="no local tiers configured"
            )
        for tier in sorted(self.tiers):
            provider = self.tiers[tier]
            for attempt in range(1 + self.policy.max_local_retries):
                result = self._call_tier(provider, request, attempt)
                accepted, why = judge(result, self.policy)
                record(tier, result, accepted, why)
                if accepted:
                    final, final_tier = result, tier
                    break
                if why not in ("stability_below_threshold", "abstain", "no_valid_choice"):
                    break  # retrying cannot fix a broken or missing model
            if final is not None:
                break
            label = result.model or getattr(provider, "model_name", None) or "unknown"
            previous = (f"{result.provider}/{label}", why or "provider_error")

        if final is None:
            result = self._call("jev", request)
            accepted = result.ok
            record(
                3, result, accepted, None if accepted else (result.fallback_reason or "jev_failed")
            )
            if accepted:
                final, final_tier = result, 3
            else:
                final = result

        self.ledger.write(receipts)
        follow = final.choice if final_tier is not None and final.ok else None
        if final_tier is None:
            note = (
                "no acceptable decision: local tiers escalated and JEV is disabled; "
                "use your own judgement"
                if final.fallback_reason == "jev_disabled"
                else f"no acceptable decision ({final.fallback_reason}); use your own judgement"
            )
        elif final.abstain:
            note = f"tier {final_tier} abstained; use your own judgement"
        else:
            note = None
        return Outcome(
            request.id, "local-first", follow=follow, decision=final,
            tier=final_tier, attempts=attempts, note=note,
        )  # fmt: skip

    def _call_tier(
        self, provider: TierProvider, request: DecisionRequest, attempt: int = 0
    ) -> DecisionResult:
        try:
            if attempt:  # a retry draws fresh seeds
                return provider.decide(request, attempt=attempt)
            return provider.decide(request)
        except Exception as error:  # noqa: BLE001 - a provider bug must not reach the agent
            return DecisionResult.failure(
                provider.name, redact(f"{type(error).__name__}: {error}", self.secret_literals)[:200],
                "nobodywho_error",
            )  # fmt: skip


def _label(result: DecisionResult) -> str:
    return "ABSTAIN" if result.abstain else str(result.choice)


def _result_note(result: DecisionResult) -> str | None:
    if result.abstain:
        return "provider abstained; use your own judgement"
    if not result.ok:
        return f"no decision ({result.fallback_reason}); use your own judgement"
    return None


_SKIP_NOTES = {
    "recursive_call": "nested decision call refused",
    "recursive_routing_question": "the router does not decide whether to use itself",
    "bypass": "bypassed by user instruction",
    "mode_off": "decision provider is off",
    "unknown_mode": "unknown mode; no provider called",
}
