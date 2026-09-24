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

from .contract import ABSTAIN, DecisionResult

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
    "local_malformed_response": "invalid_output",
    "jev_malformed_response": "invalid_output",
    "local_no_majority": "no_valid_choice",
    "local_error": "worker_failure",
    "nobodywho_error": "provider_error",
    "nobodywho_unavailable": "model_unavailable",
}


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
