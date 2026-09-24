"""Provider-neutral routing: one request, one mode, a normalized outcome.

Modes
  off      no provider is called
  jev      TypeSafe JEV only
  local    NobodyWho only (no inference API cost)
  shadow   JEV is authoritative; NobodyWho sees the identical request, its
           answer is recorded but can never become the decision
  compare  both run; the outcome is a comparison, and a disagreement never
           resolves to either provider

There is deliberately no automatic local -> JEV escalation.

The router only advises. It never executes anything, and its failures are
returned as data (`follow: null`) instead of raised.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import MODES
from .contract import DecisionRequest, DecisionResult
from .ledger import Ledger
from .sanitize import redact, redact_value


class Provider(Protocol):
    name: str

    def decide(self, request: DecisionRequest) -> DecisionResult: ...


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
    ) -> None:
        self.providers = providers
        self.ledger = ledger
        self.secret_literals = secret_literals

    def route(
        self,
        request: DecisionRequest,
        mode: str,
        *,
        bypass: bool = False,
        user_directive: str | None = None,
    ) -> Outcome:
        mode = (mode or "").lower()
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
