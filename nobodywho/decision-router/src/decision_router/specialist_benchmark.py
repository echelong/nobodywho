"""Cold/warm benchmarks for the Tev-style specialist and its escalation path.

Runs only when explicitly asked (`decision benchmark specialist`); it never runs
during real work and never contacts TypeSafe. It reports measurements only:
latency distributions, valid-output, abstention and escalation rates, and
accuracy on the objective benchmark cases. Vote shares are stability proxies,
never probabilities; no latency target is an acceptance criterion here.

Sections
  cold_specialist    specialist tier, a fresh worker per call (model reload)
  warm_specialist    specialist tier, the persistent worker kept loaded
  generic_baseline   the generic escalation tier answering the same cases
  escalation_path    the real local-first route (specialist then generic)

Rates per section: `valid_output_rate` (a valid option token or the explicit
ABSTAIN outcome, no error), `abstention_rate`, and `escalation_rate` (the result
would not pass the acceptance policy; in `escalation_path` the case actually
routed past tier 1).
"""

from __future__ import annotations

import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from .acceptance import Policy, judge
from .contract import DecisionRequest
from .ledger import Ledger
from .providers.nobodywho import NobodyWhoProvider, subprocess_runner
from .router import Router


def stats(latencies: list[int]) -> dict[str, Any]:
    """n, mean, p50, p95, p99, min, max (nearest-rank percentiles)."""
    if not latencies:
        return {
            "n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None,
            "p99_ms": None, "min_ms": None, "max_ms": None,
        }  # fmt: skip
    ordered = sorted(latencies)

    def percentile(p: float) -> int:
        return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]

    return {
        "n": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 1),
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return round(sum(1 for row in rows if row.get(key)) / len(rows), 4) if rows else None


def summarize(rows: list[dict[str, Any]], escalated_key: str = "would_escalate") -> dict[str, Any]:
    """The honest per-section report; every rate is a plain fraction, not a score."""
    objective = [row for row in rows if row.get("expected") is not None]
    return {
        "latency": stats(
            [row["latency_ms"] for row in rows if isinstance(row.get("latency_ms"), int)]
        ),
        "valid_output_rate": _rate(rows, "valid_output"),
        "abstention_rate": _rate(rows, "abstained"),
        "escalation_rate": _rate(rows, escalated_key),
        "objective_cases": len(objective),
        "correct": sum(row.get("correct") is True for row in objective),
    }


def _requests(cases: list[dict[str, Any]]) -> list[DecisionRequest]:
    return [
        DecisionRequest.from_dict(
            {
                "id": f"bench-specialist-{index:02d}",
                "question_id": "benchmark",
                "state": case["state"],
                "question": case["question"],
                "choices": case["choices"],
                "allow_abstain": True,
                "risk": "low",
            }
        )
        for index, case in enumerate(cases)
    ]


def _decide_section(
    provider: Any,
    requests: list[DecisionRequest],
    cases: list[dict[str, Any]],
    policy: Policy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for request, case in zip(requests, cases):
        result = provider.decide(request)
        chosen = "ABSTAIN" if result.abstain else result.choice
        accepted, _why = judge(result, policy)
        expected = case.get("expected")
        rows.append(
            {
                "case": case.get("kind"),
                "choice": chosen,
                "expected": expected,
                "correct": None if expected is None else chosen == expected,
                "valid_output": result.ok,
                "abstained": result.abstain,
                "would_escalate": not accepted,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }
        )
    return summarize(rows), rows


def _route_section(
    router: Any, requests: list[DecisionRequest], cases: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for request, case in zip(requests, cases):
        outcome = router.route(request, "local-first")
        decision = outcome.decision
        chosen = (
            "ABSTAIN" if decision and decision.abstain else (decision.choice if decision else None)
        )
        accepted = outcome.tier is not None and decision is not None and decision.ok
        expected = case.get("expected")
        rows.append(
            {
                "case": case.get("kind"),
                "choice": chosen,
                "expected": expected,
                "correct": None if expected is None else chosen == expected,
                "valid_output": bool(decision is not None and decision.ok),
                "abstained": bool(decision and decision.abstain),
                "escalated_past_tier1": not accepted or outcome.tier != 1,
                "decision_source": outcome.decision_source,
                "tier": outcome.tier,
                "attempts": len(outcome.attempts),
                "error": decision.error if decision else None,
                "latency_ms": sum(a.get("latency_ms") or 0 for a in outcome.attempts),
            }
        )
    return summarize(rows, "escalated_past_tier1"), rows


def run(
    config: dict[str, Any],
    limit: int = 0,
    *,
    cases: list[dict[str, Any]] | None = None,
    specialist: Any | None = None,
    generic: Any | None = None,
    router: Any | None = None,
) -> dict[str, Any]:
    """The four benchmark sections. Injected providers make the harness testable."""
    from .benchmark import CASES

    chosen_cases = CASES if cases is None else cases
    if limit:
        chosen_cases = chosen_cases[:limit]
    requests = _requests(chosen_cases)
    policy = Policy.from_config(config)

    if specialist is None:
        # Cold: a fresh worker per call, so every row includes the model load.
        specialist = NobodyWhoProvider.for_tier(
            config, "1", runner=subprocess_runner(sys.executable)
        )
    if generic is None:
        generic = NobodyWhoProvider.for_tier(config, "2")
    if router is None:
        warm_specialist = NobodyWhoProvider.for_tier(config, "1")
        router = Router(
            {"nobodywho": generic},
            Ledger(Path(tempfile.mkdtemp(prefix="decision-bench-")) / "ledger.jsonl", ()),
            (),
            tiers={1: warm_specialist, 2: generic},
            policy=policy,
            jev_enabled=False,  # a benchmark never contacts TypeSafe
        )

    cold_summary, cold_rows = _decide_section(specialist, requests, chosen_cases, policy)
    warm_provider = getattr(router, "tiers", {}).get(1, specialist)
    warm_summary, warm_rows = _decide_section(warm_provider, requests, chosen_cases, policy)
    generic_summary, generic_rows = _decide_section(generic, requests, chosen_cases, policy)
    escalation_summary, escalation_rows = _route_section(router, requests, chosen_cases)

    return {
        "benchmark": "specialist",
        "cases": len(chosen_cases),
        "cold_specialist": dict(cold_summary, model=getattr(specialist, "model_name", None)),
        "warm_specialist": dict(warm_summary, model=getattr(warm_provider, "model_name", None)),
        "generic_baseline": dict(generic_summary, model=getattr(generic, "model_name", None)),
        "escalation_path": escalation_summary,
        "notes": [
            (
                "latencies are wall-clock milliseconds of real local calls; no latency "
                "target is an acceptance criterion"
            ),
            "confidence is sample_stability (a vote-share proxy), never a probability",
            (
                "escalation_rate in the provider sections is the share of results the "
                "acceptance policy would not accept (they would escalate); in "
                "escalation_path it is the share of cases that routed past tier 1"
            ),
        ],
        "rows": {
            "cold_specialist": cold_rows,
            "warm_specialist": warm_rows,
            "generic_baseline": generic_rows,
            "escalation_path": escalation_rows,
        },
    }
