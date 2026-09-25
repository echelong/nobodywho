"""When a local-first tier's answer is accepted, and why it escalates otherwise.

Only the reasons in ESCALATION_REASONS may move a decision to the next tier.
A local answer that is valid and passes the policy is final: nothing here
looks at whether a caller would have preferred a different choice, at
latency, or at earlier decisions.

The thresholds work on `sample_stability` (vote share across seeded,
grammar-constrained samples). That is a stability proxy, not a calibrated
probability, and it is never converted into one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import (
    ABSTAIN,
    ABSTAIN_AMBIGUOUS_OUTPUT,
    ABSTAIN_GRAMMAR_FAILURE,
    ABSTAIN_INVALID_OUTPUT,
    ABSTAIN_LOW_MARGIN,
    ABSTAIN_MODEL,
    ABSTAIN_PRIMARY_IDENTITY_MISMATCH,
    ABSTAIN_RUNTIME_FAILURE,
    ABSTAIN_SPECIALIST_UNAVAILABLE,
    ABSTAIN_TIMEOUT,
    DecisionResult,
)

ESCALATION_REASONS = (
    "provider_error",
    "timeout",
    "invalid_output",
    "abstain",
    "no_valid_choice",
    "worker_failure",
    "model_unavailable",
    "stability_below_threshold",
    "insufficient_evidence",
)

# Provider-level fallback reasons mapped onto the escalation vocabulary.
_PROVIDER_REASONS = {
    "local_timeout": "timeout",
    "jev_timeout": "timeout",
    "local_model_unavailable": "model_unavailable",
    "local_runtime_unavailable": "model_unavailable",
    "local_gpu_unavailable": "model_unavailable",
    "local_malformed_response": "invalid_output",
    "local_grammar_failure": "invalid_output",
    "jev_malformed_response": "invalid_output",
    "local_no_majority": "no_valid_choice",
    "local_error": "worker_failure",
    "nobodywho_error": "provider_error",
    "nobodywho_unavailable": "model_unavailable",
    "specialist_unavailable": "model_unavailable",
    "primary_identity_mismatch": "model_unavailable",
}

# Provider-level failure reasons mapped onto the closed abstention vocabulary
# (contract.ABSTAIN_REASONS). A specialist tier that cannot serve is
# SPECIALIST_UNAVAILABLE; any other tier whose runtime cannot serve has a
# runtime failure. Nothing here ever turns into a candidate option.
_ABSTAIN_BY_FALLBACK_SPECIALIST = {
    "specialist_unavailable": ABSTAIN_SPECIALIST_UNAVAILABLE,
    "primary_identity_mismatch": ABSTAIN_PRIMARY_IDENTITY_MISMATCH,
    "local_malformed_response": ABSTAIN_INVALID_OUTPUT,
    "local_grammar_failure": ABSTAIN_GRAMMAR_FAILURE,
    "jev_malformed_response": ABSTAIN_INVALID_OUTPUT,
    "local_no_majority": ABSTAIN_AMBIGUOUS_OUTPUT,
    "local_timeout": ABSTAIN_TIMEOUT,
    "jev_timeout": ABSTAIN_TIMEOUT,
    "local_error": ABSTAIN_RUNTIME_FAILURE,
    "local_model_unavailable": ABSTAIN_SPECIALIST_UNAVAILABLE,
    "local_runtime_unavailable": ABSTAIN_SPECIALIST_UNAVAILABLE,
    "local_gpu_unavailable": ABSTAIN_SPECIALIST_UNAVAILABLE,
    "jev_disabled": ABSTAIN_RUNTIME_FAILURE,
    "jev_failed": ABSTAIN_RUNTIME_FAILURE,
}
_ABSTAIN_BY_FALLBACK_GENERIC = dict(
    _ABSTAIN_BY_FALLBACK_SPECIALIST,
    local_model_unavailable=ABSTAIN_RUNTIME_FAILURE,
    local_runtime_unavailable=ABSTAIN_RUNTIME_FAILURE,
    local_gpu_unavailable=ABSTAIN_RUNTIME_FAILURE,
)


@dataclass(frozen=True)
class Policy:
    min_stability: float = 0.66
    min_margin: int = 1
    escalate_on_abstain: bool = True
    max_local_retries: int = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Policy:
        a = config.get("acceptance", {})
        policy = cls(
            min_stability=float(a.get("min_stability", 0.66)),
            min_margin=int(a.get("min_margin", 1)),
            escalate_on_abstain=bool(a.get("escalate_on_abstain", True)),
            max_local_retries=int(a.get("max_local_retries", 0)),
        )
        if not 0.0 <= policy.min_stability <= 1.0:
            raise ValueError("acceptance.min_stability must be within [0, 1]")
        if policy.min_margin < 0 or not 0 <= policy.max_local_retries <= 3:
            raise ValueError("acceptance.min_margin >= 0 and max_local_retries in 0..3")
        return policy


def vote_margin(votes: dict[str, int] | None) -> int:
    counts = sorted((votes or {}).values(), reverse=True)
    if not counts:
        return 0
    return counts[0] - (counts[1] if len(counts) > 1 else 0)


def judge(result: DecisionResult, policy: Policy) -> tuple[bool, str | None]:
    """(accepted, escalation_reason) for one local tier's result."""
    if result.error is not None or (result.choice is None and not result.abstain):
        reason = _PROVIDER_REASONS.get(result.fallback_reason or "", "provider_error")
        return False, reason
    if result.abstain:
        if not policy.escalate_on_abstain:
            return True, None
        # Every sample judged the evidence insufficient, versus a mixed vote that landed on it.
        unanimous = bool(result.votes) and set(result.votes) == {ABSTAIN}
        return False, "insufficient_evidence" if unanimous else "abstain"
    if result.confidence is None or result.confidence < policy.min_stability:
        return False, "stability_below_threshold"
    if vote_margin(result.votes) < policy.min_margin:
        return False, "stability_below_threshold"
    return True, None


def abstain_reason_for(result: DecisionResult, *, specialist: bool) -> str | None:
    """The explicit abstention reason implied by a result that selected no option.

    None when the result carries a selection or an explicit reason already. The
    returned value is always one of contract.ABSTAIN_REASONS.
    """
    if result.abstain_reason is not None:
        return result.abstain_reason
    if result.abstain:
        return ABSTAIN_MODEL
    if result.choice is not None:
        return None
    table = _ABSTAIN_BY_FALLBACK_SPECIALIST if specialist else _ABSTAIN_BY_FALLBACK_GENERIC
    if result.fallback_reason in table:
        return table[result.fallback_reason]
    return ABSTAIN_RUNTIME_FAILURE if result.error else ABSTAIN_AMBIGUOUS_OUTPUT


def rejection_abstain_reason(escalation_reason: str | None) -> str | None:
    """The abstention reason for a valid result the acceptance policy rejected."""
    if escalation_reason == "stability_below_threshold":
        return ABSTAIN_LOW_MARGIN
    if escalation_reason in ("abstain", "insufficient_evidence"):
        return ABSTAIN_MODEL
    return None
