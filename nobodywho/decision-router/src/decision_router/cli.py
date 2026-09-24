"""`decision` command line.

decision provider [off|jev|local|shadow|compare|local-first]   show or switch the mode
decision ask [--json JSON | --file F] [--mode M] [--caller NAME] [--bypass]
             [--user-directive TEXT]
decision jev status | enable | disable              JEV fallback switch (disable is absolute)
decision stats [--caller NAME] [--since ISO]        usage, tiers, JEV calls, fallbacks
decision doctor [--live]                           check every provider
decision benchmark [quality|latency] [...]           explicit local benchmarks (latency: no JEV)
decision test [--mode M]                           run a canned live decision
decision ledger [-n N]                             recent receipts
decision local pull [--tier local|1|2] [--source S]  download + pin a local model
decision prune [--caller NAME] [--budget-chars N] [--goal T] [--command C] [--json]
               [-- COMMAND ARGS...]              shorten long output (stdin, or run COMMAND)
decision prune status | test                     pruning configuration / corpus self-test
decision adapter shim CALLER                     write the bash shim a tool uses for pruning
decision install-rule [--dest DIR]                 install the Cline rule
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from . import config as cfg
from .acceptance import Policy
from .contract import DecisionRequest, RequestError
from .ledger import Ledger
from .providers import JevProvider, NobodyWhoProvider
from .providers.nobodywho import PersistentWorker
from .prune import (
    MAX_OUTPUT_CHARS,
    MIN_BUDGET_CHARS,
    Pruner,
    PruneRequest,
    PruneResult,
    Tier,
    estimate_tokens,
    fit,
    hard_cap,
    plan,
)
from .prune_providers import jev_judge, local_judge
from .router import Router

RULE_NAME = "decision-router.md"
STREAM_AFTER_BYTES = 8 << 20
DEFAULT_RULE_DIR = Path.home() / ".cline" / "rules"


def caller_name(explicit: str | None) -> str:
    """Which environment is asking; adapters pass --caller (or DECISION_ROUTER_CALLER)."""
    name = (explicit or os.environ.get("DECISION_ROUTER_CALLER", "")).strip().lower()
    return name[:32] if name and name.replace("-", "").replace("_", "").isalnum() else "unknown"


def build_router(config: dict[str, Any], caller: str = "unknown") -> Router:
    jev = JevProvider.from_config(config)
    local = NobodyWhoProvider.from_config(config)
    literals = jev.secret_literals()
    return Router(
        {"jev": jev, "nobodywho": local},
        Ledger(cfg.ledger_path(), literals, caller),
        literals,
        tiers={int(t): NobodyWhoProvider.for_tier(config, t) for t in cfg.TIER_NAMES},
        policy=Policy.from_config(config),
        jev_enabled=jev_enabled(config),
    )


def jev_enabled(config: dict[str, Any]) -> bool:
    return config["jev"].get("enabled", True) is not False


def _model_label(settings: dict[str, Any]) -> str:
    info = settings.get("model_info") or {}
    name = info.get("name") or (
        Path(settings["model_path"]).stem if settings.get("model_path") else None
    )
    if not name:
        return "not configured (decision local pull --tier ...)"
    quant = info.get("quantization")
    return f"{name}{' ' + quant if quant else ''}"


def describe_mode(config: dict[str, Any]) -> str:
    mode, source = cfg.mode(config)
    lines = [f"mode: {mode}  (from {source})"]
    jev_state = "disabled (hard: never called)" if not jev_enabled(config) else "enabled"
    if mode == "local-first":
        tier_labels = {}
        lines.append("decision:")
        for tier in cfg.TIER_NAMES:
            t = cfg.tier_settings(config, tier)
            tier_labels[tier] = f"nobodywho / {_model_label(t)}"
            lifecycle = (
                f"persistent worker, idle exit {t.get('idle_timeout_s'):g}s"
                if t.get("persistent")
                else "loaded per decision"
            )
            lines.append(
                f"  tier {tier}: {tier_labels[tier]} ({lifecycle}, gpu={bool(t.get('use_gpu'))})"
            )
        lines += [
            "  tier 3: typesafe / " + config["jev"]["model"],
            "pruning:",
            f"  tier 1: {tier_labels['1']}",
            f"  tier 2: {tier_labels['2']}",
            "  tier 3: typesafe / " + config["jev"]["model"],
            "  final fallback: native truncation",
            "JEV:",
            f"  {'dormant unless both local tiers fail' if jev_enabled(config) else jev_state + '; tier 3 dormant unless enabled after both local tiers fail'}",
        ]
    elif mode in ("jev", "shadow", "compare"):
        lines += [f"jev: {jev_state}"]
    return "\n".join(lines)


def _print(data: Any) -> None:
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")


def cmd_provider(args: argparse.Namespace) -> int:
    config = cfg.load()
    if args.mode is not None:
        cfg.update({"mode": args.mode})
        config = cfg.load()
        if args.mode in ("local", "shadow", "compare"):
            ok, why = NobodyWhoProvider.from_config(config).available()
            if not ok:
                print(f"warning: local provider not ready: {why}", file=sys.stderr)
        if args.mode == "local-first":
            for tier in cfg.TIER_NAMES:
                ok, why = NobodyWhoProvider.for_tier(config, tier).available()
                if not ok:
                    print(f"warning: tier {tier} not ready: {why}", file=sys.stderr)
    print(describe_mode(config))
    if os.environ.get("DECISION_ROUTER_MODE"):
        print("warning: DECISION_ROUTER_MODE is set and overrides this", file=sys.stderr)
    return 0


def _read_request(args: argparse.Namespace) -> Any:
    if args.json is not None:
        return json.loads(args.json)
    if args.file is not None:
        return json.loads(Path(args.file).read_text())
    return json.loads(sys.stdin.read())


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        request = DecisionRequest.from_dict(_read_request(args))
    except (ValueError, OSError) as error:
        kind = "invalid_request" if isinstance(error, RequestError) else "unreadable_request"
        _print({"follow": None, "error": kind, "detail": str(error)[:300]})
        return 2
    try:
        config = cfg.load()
        mode = args.mode or cfg.mode(config)[0]
        caller = caller_name(args.caller)
        outcome = build_router(config, caller).route(
            request, mode, bypass=args.bypass, user_directive=args.user_directive, caller=caller
        )
    except Exception as error:  # noqa: BLE001 - the agent gets data, never a traceback
        _print({"follow": None, "error": "router_error", "detail": type(error).__name__})
        return 0
    _print(outcome.to_dict())
    return 0


SAMPLE_REQUEST = {
    "question_id": "next_step",
    "state": (
        "Goal: fix a failing unit test for an ISO-8601 date parser. The test "
        "failed twice with the same one-hour offset after identical retries. "
        "The parser ignores the timezone suffix; the fixtures contain '+01:00'."
    ),
    "question": "What should the agent do next?",
    "choices": {
        "retry_same": "Run the same test again without changing anything.",
        "inspect_timezone": "Inspect how the parser handles timezone offsets before editing.",
        "rewrite_parser": "Rewrite the whole parser from scratch.",
    },
    "allow_abstain": True,
    "risk": "low",
}


def cmd_test(args: argparse.Namespace) -> int:
    config = cfg.load()
    mode = args.mode or cfg.mode(config)[0]
    request = DecisionRequest.from_dict(dict(SAMPLE_REQUEST, id=f"selftest-{os.getpid()}"))
    outcome = build_router(config).route(request, mode)
    print(json.dumps(outcome.to_dict(), indent=2))
    return 0


def _median(values: list[int]) -> int | None:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else None


def _tier_agreement(per_model: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    pairs = [
        (tier1.get("choice"), tier2.get("choice"))
        for tier1, tier2 in zip(per_model.get("1", []), per_model.get("2", []))
        if tier1.get("choice") is not None
        and tier2.get("choice") is not None
        and not tier1.get("error")
        and not tier2.get("error")
    ]
    disagreements = sum(left != right for left, right in pairs)
    return {
        "tier_comparable_cases": len(pairs),
        "tier_agreement": len(pairs) - disagreements,
        "tier_disagreement": disagreements,
    }


def _worker_metrics(provider: NobodyWhoProvider) -> dict[str, Any]:
    runner = provider.runner
    if not isinstance(runner, PersistentWorker):
        return {"worker": "oneshot", "ram_rss_mib": None, "gpu_offload": "not observable"}
    pid = runner.pid()
    rss_mib = None
    if pid:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_mib = round(int(line.split()[1]) / 1024, 1)
                    break
        except (OSError, ValueError):
            pass
    offload: str | None = None
    gpu_load_error: str | None = None
    try:
        if runner.log_path.is_file():
            log = runner.log_path.read_text(errors="replace")
            match = re.search(r"offload(?:ed|ing)?\s+(\d+)/(\d+)\s+layers", log, re.IGNORECASE)
            if match:
                offload = f"{match.group(1)}/{match.group(2)} layers"
            if re.search(r"ErrorOutOfDeviceMemory|out of device memory", log, re.IGNORECASE):
                gpu_load_error = "out of device memory"
    except OSError:
        pass
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        gpu_memory = (
            gpu.stdout.strip().splitlines()[0]
            if gpu.returncode == 0 and gpu.stdout.strip()
            else None
        )
    except (OSError, subprocess.TimeoutExpired):
        gpu_memory = None
    if offload is None:
        offload = "not observed in worker log" if provider.use_gpu else "CPU configured"
    return {
        "worker": "persistent",
        "worker_pid": pid,
        "ram_rss_mib": rss_mib,
        "gpu_memory_used_total_mib": gpu_memory,
        "gpu_offload": offload,
        "gpu_load_error": gpu_load_error,
        "worker_log": str(runner.log_path) if runner.log_path.is_file() else None,
    }


def _benchmark_decisions(config: dict[str, Any], limit: int, include_jev: bool) -> dict[str, Any]:
    from .benchmark import CASES

    cases = CASES[:limit] if limit else CASES
    results_by_tier: dict[str, dict[str, Any]] = {}
    per_model: dict[str, list[dict[str, Any]]] = {}
    # Tier 2 runs first; tier 1 is left resident when the benchmark completes.
    old_capture = os.environ.get("DECISION_ROUTER_CAPTURE_WORKER_LOG")
    os.environ["DECISION_ROUTER_CAPTURE_WORKER_LOG"] = "1"
    try:
        for tier in reversed(cfg.TIER_NAMES):
            provider = NobodyWhoProvider.for_tier(config, tier)
            ready, reason = provider.available()
            if not ready:
                results_by_tier[tier] = {"status": "unavailable", "reason": reason}
                continue
            rows: list[dict[str, Any]] = []
            for index, case in enumerate(cases):
                request = DecisionRequest.from_dict(
                    {
                        "id": f"bench-{tier}-{index:02d}",
                        "question_id": "benchmark",
                        "state": case["state"],
                        "question": case["question"],
                        "choices": case["choices"],
                        "allow_abstain": True,
                        "risk": "low",
                    }
                )
                result = provider.decide(request)
                chosen = "ABSTAIN" if result.abstain else result.choice
                expected = case.get("expected")
                correct = None if expected is None else chosen == expected
                row = {
                    "case": case["kind"],
                    "choice": chosen,
                    "expected": expected,
                    "correct": correct,
                    "error": result.error,
                    "fallback_reason": result.fallback_reason,
                    "confidence": result.confidence,
                    "confidence_kind": result.confidence_kind,
                    "latency_ms": result.latency_ms,
                    "sample_ms": result.details.get("sample_ms"),
                    "cold_load_ms": result.details.get("load_ms")
                    if result.details.get("model_reused") is False
                    else None,
                    "model_reused": result.details.get("model_reused"),
                }
                rows.append(row)
            per_model[tier] = rows
            latencies = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), int)]
            sample_latencies = [
                value
                for r in rows
                for value in (r.get("sample_ms") or [])
                if isinstance(value, int)
            ]
            stability = [
                r["confidence"]
                for r in rows
                if r.get("confidence_kind") == "sample_stability"
                and isinstance(r.get("confidence"), (int, float))
            ]
            objective = [r["correct"] for r in rows if r.get("correct") is not None]
            results_by_tier[tier] = {
                "status": "complete"
                if rows and not all(r.get("error") for r in rows)
                else "failed",
                "model": provider.model_name,
                "correct": sum(value is True for value in objective),
                "objective_cases": len(objective),
                "incorrect": sum(value is False for value in objective),
                "abstentions": sum(r.get("choice") == "ABSTAIN" for r in rows),
                "mean_sample_stability": round(sum(stability) / len(stability), 4)
                if stability
                else None,
                "median_total_latency_ms": _median(latencies),
                "median_sample_latency_ms": _median(sample_latencies),
                "cold_load_ms": next(
                    (r["cold_load_ms"] for r in rows if r.get("cold_load_ms") is not None), None
                ),
                "warm_total_latency_ms": _median(
                    [
                        r["latency_ms"]
                        for r in rows
                        if r.get("model_reused") is True and isinstance(r.get("latency_ms"), int)
                    ]
                ),
                "hardware": _worker_metrics(provider),
                "cases": rows,
            }
    finally:
        if old_capture is None:
            os.environ.pop("DECISION_ROUTER_CAPTURE_WORKER_LOG", None)
        else:
            os.environ["DECISION_ROUTER_CAPTURE_WORKER_LOG"] = old_capture

    agreement = _tier_agreement(per_model)
    jev_report: dict[str, Any] | str = "not requested"
    if include_jev:
        if not jev_enabled(config):
            jev_report = "skipped: global JEV switch is disabled"
        else:
            jev = JevProvider.from_config(config)
            available, why = jev.available()
            if not available:
                jev_report = {"status": "unavailable", "reason": why, "calls": 0}
            else:
                rows = []
                for index, case in enumerate(cases):
                    request = DecisionRequest.from_dict(
                        {
                            "id": f"bench-jev-{index:02d}",
                            "question_id": "benchmark",
                            "state": case["state"],
                            "question": case["question"],
                            "choices": case["choices"],
                            "allow_abstain": True,
                            "risk": "low",
                        }
                    )
                    result = jev.decide(request)
                    chosen = "ABSTAIN" if result.abstain else result.choice
                    expected = case.get("expected")
                    rows.append(
                        {
                            "case": case["kind"],
                            "choice": chosen,
                            "expected": expected,
                            "correct": None if expected is None else chosen == expected,
                            "error": result.error,
                            "confidence": result.confidence,
                            "confidence_kind": result.confidence_kind,
                            "latency_ms": result.latency_ms,
                        }
                    )
                outcomes = [row["correct"] for row in rows if row["correct"] is not None]
                jev_report = {
                    "status": "complete",
                    "calls": len(rows),
                    "correct": sum(value is True for value in outcomes),
                    "objective_cases": len(outcomes),
                    "incorrect": sum(value is False for value in outcomes),
                    "abstentions": sum(row["choice"] == "ABSTAIN" for row in rows),
                    "median_latency_ms": _median(
                        [
                            row["latency_ms"]
                            for row in rows
                            if isinstance(row.get("latency_ms"), int)
                        ]
                    ),
                    "cases": rows,
                }

    outcome: dict[str, Any] = {
        "cases_run": len(cases),
        "tiers": results_by_tier,
        **agreement,
        "jev": jev_report,
    }
    return outcome


def _benchmark_pruning(config: dict[str, Any], limit: int, include_jev: bool) -> dict[str, Any]:
    from .prune_corpus import cases as fixture_cases

    corpus = fixture_cases()
    if limit:
        corpus = corpus[:limit]
    pc = config["prune"]
    by_tier: dict[str, Any] = {}
    old_capture = os.environ.get("DECISION_ROUTER_CAPTURE_WORKER_LOG")
    os.environ["DECISION_ROUTER_CAPTURE_WORKER_LOG"] = "1"
    try:
        for tier_name in reversed(cfg.TIER_NAMES):
            settings = pc.get(f"tier{tier_name}", {})
            provider = NobodyWhoProvider.for_tier(config, tier_name)
            ready, reason = provider.available()
            if not ready or settings.get("enabled", True) is False:
                by_tier[tier_name] = {
                    "status": "unavailable",
                    "reason": reason if not ready else "disabled",
                }
                continue
            model_tier = Tier(
                int(tier_name),
                "nobodywho",
                provider.model_name,
                local_judge(provider, timeout_s=settings.get("timeout_s")),
                max_blocks=int(settings.get("max_blocks", 8)),
            )
            pruner = Pruner(
                [model_tier],
                jev_enabled=False,
                block_lines=int(pc.get("block_lines", 30)),
                hard_cap_factor=float(pc.get("hard_cap_factor", 2.0)),
                archive_dir=None,
            )
            rows: list[dict[str, Any]] = []
            for name, command, output, required in corpus:
                request = PruneRequest(
                    output=output,
                    caller="benchmark",
                    command=command,
                    budget_chars=int(pc.get("budget_chars", 12_000)),
                )
                result = pruner.prune(request)
                missing = [fact for fact in required if fact not in result.text]
                source_lines = output.splitlines()
                hallucinated = [
                    line
                    for line in result.text.splitlines()
                    if not line.startswith("[... ")
                    and not line.startswith("[decision prune:")
                    and line not in source_lines
                    and line not in output
                ]
                rows.append(
                    {
                        "fixture": name,
                        "provider": result.provider,
                        "latency_ms": result.latency_ms,
                        "compression_ratio": result.compression_ratio,
                        "required_facts": len(required),
                        "retained_facts": len(required) - len(missing),
                        "missing_facts": missing,
                        "hallucinated_lines": len(hallucinated),
                        "native_fallback": result.native_fallback,
                    }
                )
            successful = [r for r in rows if r["provider"] == "nobodywho"]
            ratios = [float(r["compression_ratio"]) for r in rows]
            by_tier[tier_name] = {
                "status": "complete",
                "model": provider.model_name,
                "fixtures_run": len(rows),
                "provider_successes": len(successful),
                "required_facts_retained": sum(r["retained_facts"] for r in rows),
                "required_facts_total": sum(r["required_facts"] for r in rows),
                "hallucinated_lines": sum(r["hallucinated_lines"] for r in rows),
                "mean_compression_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
                "median_latency_ms": _median([int(r["latency_ms"]) for r in rows]),
                "hardware": _worker_metrics(provider),
                "fixtures": rows,
                "jev_calls": pruner.jev_calls,
            }
    finally:
        if old_capture is None:
            os.environ.pop("DECISION_ROUTER_CAPTURE_WORKER_LOG", None)
        else:
            os.environ["DECISION_ROUTER_CAPTURE_WORKER_LOG"] = old_capture

    jev_report: dict[str, Any] | str = "not requested"
    if include_jev:
        if not jev_enabled(config):
            jev_report = "skipped: global JEV switch is disabled"
        else:
            chained = build_pruner(config)
            local_first_rows = []
            for name, command, output, required in corpus:
                request = PruneRequest(
                    output=output,
                    caller="benchmark",
                    command=command,
                    budget_chars=int(pc.get("budget_chars", 12_000)),
                )
                result = chained.prune(request)
                local_first_rows.append(
                    {
                        "fixture": name,
                        "provider": result.provider,
                        "tier": result.tier,
                        "required_facts": len(required),
                        "retained_facts": sum(fact in result.text for fact in required),
                        "native_fallback": result.native_fallback,
                        "fallback_reason": result.fallback_reason,
                        "jev_used": result.jev_used,
                    }
                )
            jev_report = {
                "status": "complete",
                "calls": chained.jev_calls,
                "local_first": local_first_rows,
            }
    result: dict[str, Any] = {"tiers": by_tier, "jev": jev_report}
    return result


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = cfg.load()
    if args.kind == "latency":
        from . import latency

        latency_report = latency.run(config, args)
        print(json.dumps(latency_report, indent=2) if args.json else latency.table(latency_report))
        return 0 if all(not m.get("error") for m in latency_report["models"]) else 1
    report: dict[str, Any] = {"operation": args.operation, "limit": args.limit or "all"}
    if args.operation in ("all", "decision"):
        report["decision"] = _benchmark_decisions(config, args.limit, args.include_jev)
    if args.operation in ("all", "prune"):
        report["prune"] = _benchmark_pruning(config, args.limit, args.include_jev)
    print(json.dumps(report, indent=2))
    return 0


def _check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'ok' if ok else '!!'}] {label}: {detail}")
    return ok


def _runtime_version() -> tuple[bool, str]:
    try:
        from importlib.metadata import version

        return True, f"nobodywho {version('nobodywho')}"
    except Exception:  # noqa: BLE001 - any failure means "not installed"
        return False, "nobodywho not installed in this interpreter"


def cmd_doctor(args: argparse.Namespace) -> int:
    config = cfg.load()
    mode, source = cfg.mode(config)
    print(f"mode: {mode} (from {source})")
    print(f"config: {cfg.config_path()}")
    print(f"ledger: {cfg.ledger_path()}")
    healthy = _check("mode valid", mode in cfg.MODES, mode)

    print("jev:")
    jev = JevProvider.from_config(config)
    enabled = jev_enabled(config)
    print(f"  switch: {'enabled (fallback only in local-first)' if enabled else 'DISABLED (hard)'}")
    ok, why = jev.available()
    key_file = Path(str(jev.key_file)).expanduser() if jev.key_file else None
    if key_file and key_file.exists():
        perms = stat.S_IMODE(key_file.stat().st_mode)
        _check("key file", perms & 0o077 == 0, f"{key_file} (mode {perms:o}; contents not shown)")
    jev_ready = _check("available", ok, why)
    if mode in ("jev", "shadow", "compare") and enabled:
        healthy &= jev_ready
    print(f"  endpoint: {jev.endpoint}  model: {jev.model}  timeout: {jev.timeout_s:g}s")

    runtime_ok, runtime = _runtime_version()
    local_names = {"local": ("local", NobodyWhoProvider.from_config(config))}
    for tier in cfg.TIER_NAMES:
        local_names[f"tier{tier}"] = (tier, NobodyWhoProvider.for_tier(config, tier))
    for label, (key, provider) in local_names.items():
        print(f"nobodywho [{label}]:")
        ok, why = provider.available()
        needed = (label == "local" and mode in ("local", "shadow", "compare")) or (
            label.startswith("tier") and mode == "local-first"
        )
        ready = _check("model", ok, why if not ok else str(provider.model_path))
        if needed:
            healthy &= ready
        if ok:
            info = provider.model_info()
            print(
                f"  model: {info.get('name')}  quantization: {info.get('quantization')}  "
                f"samples: {provider.samples}  gpu: {provider.use_gpu}"
            )
        print(f"  worker: {_worker_status(config, key)}")
    runtime_ready = _check("runtime", runtime_ok, runtime)
    if mode in ("local", "shadow", "compare", "local-first"):
        healthy &= runtime_ready
    print(f"acceptance: {config.get('acceptance')}")

    print("cline:")
    rule = DEFAULT_RULE_DIR / RULE_NAME
    _check("rule", rule.exists(), str(rule))
    _check(
        "recursion guard",
        os.environ.get("DECISION_ROUTER_ACTIVE") != "1",
        "not inside a decision call",
    )

    if args.live:
        print("live:")
        request = DecisionRequest.from_dict(dict(SAMPLE_REQUEST, id=f"doctor-{os.getpid()}"))
        checks: list[tuple[str, Any]] = [(k, p) for k, (_, p) in local_names.items()]
        if enabled:
            checks.insert(0, ("jev", jev))
        else:
            print("  [--] jev: skipped, JEV is disabled")
        for name, provider in checks:
            if name != "jev" and not provider.available()[0]:
                continue
            result = provider.decide(request)
            label = "ABSTAIN" if result.abstain else result.choice
            detail = (
                f"{label} via {result.model} in {result.latency_ms} ms, "
                f"{result.confidence_kind}={result.confidence}"
                if result.ok
                else f"{result.fallback_reason}: {result.error}"
            )
            healthy &= _check(name, result.ok, detail)
    print("healthy" if healthy else "problems found")
    return 0 if healthy else 1


def cmd_ledger(args: argparse.Namespace) -> int:
    for record in Ledger(cfg.ledger_path()).tail(args.n):
        if record.get("skipped"):
            outcome = f"skipped={record['skipped']}"
        else:
            label = "ABSTAIN" if record.get("abstain") else record.get("choice")
            outcome = (
                f"{record.get('provider')}:{record.get('model')} -> {label} "
                f"{record.get('confidence_kind')}={record.get('confidence')} "
                f"{record.get('latency_ms')}ms"
            )
            if record.get("error"):
                reason = record.get("provider_reason") or record.get("fallback_reason")
                outcome += f" error={reason}"
            if record.get("tier"):
                outcome += f" tier={record['tier']} accepted={record.get('accepted')}"
                if record.get("escalation_reason"):
                    outcome += f" escalated={record['escalation_reason']}"
        agreement = record.get("agreement")
        suffix = "" if agreement is None else f" agreement={agreement}"
        print(
            f"{record.get('ts')} [{record.get('caller', 'unknown')}] "
            f"{record.get('mode')}/{record.get('role')} {record.get('request_id')} "
            f"{outcome}{suffix}"
        )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    records = Ledger(cfg.ledger_path()).tail(10**9)
    if args.caller:
        records = [r for r in records if r.get("caller", "unknown") == args.caller]
    if args.since:
        records = [r for r in records if str(r.get("ts", "")) >= args.since]
    print(json.dumps(summarize(records), indent=2))
    return 0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Routing (`ask`) and pruning (`prune`) statistics, kept apart."""
    asks = [r for r in records if r.get("operation", "ask") == "ask"]
    prunes = [r for r in records if r.get("operation") == "prune"]
    return {"ask": summarize_asks(asks), "prune": summarize_prunes(prunes),
            "jev_calls_total": summarize_asks(asks)["jev_calls"]
            + sum(1 for r in prunes if r.get("jev_used"))}  # fmt: skip


def summarize_prunes(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_caller: dict[str, dict[str, Any]] = {}
    providers: dict[str, int] = {}
    fallbacks: dict[str, int] = {}
    latencies: list[int] = []
    for r in records:
        c = by_caller.setdefault(str(r.get("caller", "unknown")),
                                 {"calls": 0, "chars_in": 0, "chars_out": 0,
                                  "tier1_success": 0, "tier2_escalation": 0,
                                  "jev_fallback": 0, "native_truncation": 0,
                                  "latencies_ms": []})  # fmt: skip
        c["calls"] += 1
        c["chars_in"] += int(r.get("original_chars") or 0)
        c["chars_out"] += int(r.get("result_chars") or 0)
        raw_attempts = r.get("attempts")
        attempts: list[dict[str, Any]] = (
            [a for a in raw_attempts if isinstance(a, dict)]
            if isinstance(raw_attempts, list)
            else []
        )
        reached_tier2 = any(
            a.get("provider") == "nobodywho" and a.get("tier") == 2 for a in attempts
        )
        c["tier1_success"] += r.get("provider") == "nobodywho" and r.get("tier") == 1
        c["tier2_escalation"] += (
            r.get("provider") == "nobodywho" and r.get("tier") == 2
        ) or reached_tier2
        c["jev_fallback"] += bool(r.get("jev_used"))
        c["native_truncation"] += bool(r.get("native_fallback")) or r.get("provider") == "native"
        key = f"{r.get('provider')}/tier{r.get('tier')}"
        providers[key] = providers.get(key, 0) + 1
        if r.get("fallback_reason"):
            fallbacks[r["fallback_reason"]] = fallbacks.get(r["fallback_reason"], 0) + 1
        if isinstance(r.get("latency_ms"), int) and r.get("provider") not in ("none",):
            latencies.append(r["latency_ms"])
            c["latencies_ms"].append(r["latency_ms"])
    for c in by_caller.values():
        samples = sorted(c.pop("latencies_ms"))
        c["median_latency_ms"] = samples[len(samples) // 2] if samples else None
    return {
        "calls": len(records),
        "by_caller": by_caller,
        "by_provider": providers,
        "fallback_reasons": fallbacks,
        "jev_used": sum(1 for r in records if r.get("jev_used")),
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
    }


def summarize_asks(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-decision statistics from ledger receipts (several receipts share a request id)."""
    decisions: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        decisions.setdefault(str(r.get("request_id")), []).append(r)

    def count(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in records:
            value = r.get(key)
            if value is not None:
                out[str(value)] = out.get(str(value), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    by_caller: dict[str, dict[str, Any]] = {}
    final_tier: dict[str, int] = {}
    jev_calls = local_first = 0
    latencies: dict[str, list[int]] = {}
    for rows in decisions.values():
        caller = str(rows[0].get("caller", "unknown"))
        stats = by_caller.setdefault(
            caller,
            {
                "decisions": 0,
                "skipped": 0,
                "jev_calls": 0,
                "tier1_success": 0,
                "tier2_escalation": 0,
                "tier3_fallback": 0,
            },
        )
        stats["decisions"] += 1
        stats["skipped"] += any(r.get("skipped") for r in rows)
        calls = sum(
            1
            for r in rows
            if r.get("provider") == "jev" and r.get("provider_reason") != "jev_disabled"
        )
        stats["jev_calls"] += calls
        jev_calls += calls
        lf = [r for r in rows if r.get("mode") == "local-first" and r.get("tier")]
        if lf:
            local_first += 1
            accepted = [r["tier"] for r in lf if r.get("accepted")]
            key = f"tier{accepted[0]}" if accepted else "none"
            final_tier[key] = final_tier.get(key, 0) + 1
            if accepted:
                if accepted[0] == 1:
                    stats["tier1_success"] += 1
                elif accepted[0] == 2:
                    stats["tier2_escalation"] += 1
                else:
                    stats["tier3_fallback"] += 1
        for r in rows:
            if isinstance(r.get("latency_ms"), int) and r.get("provider"):
                latencies.setdefault(
                    f"{r['provider']}/{r.get('tier') or r.get('mode')}", []
                ).append(r["latency_ms"])
    return {
        "receipts": len(records),
        "decisions": len(decisions),
        "by_caller": by_caller,
        "by_mode": count("mode"),
        "jev_calls": jev_calls,
        "local_first": {"decisions": local_first, "accepted_at": final_tier},
        "escalation_reasons": count("escalation_reason"),
        "skipped": count("skipped"),
        "median_latency_ms": {k: sorted(v)[len(v) // 2] for k, v in sorted(latencies.items())},
    }


def cmd_jev(args: argparse.Namespace) -> int:
    if args.action == "enable":
        cfg.update({"jev": {"enabled": True}})
    elif args.action == "disable":
        cfg.update({"jev": {"enabled": False}})
    config = cfg.load()
    enabled = jev_enabled(config)
    mode = cfg.mode(config)[0]
    if not enabled:
        state = "disabled: never called in any mode or environment"
    elif mode == "local-first":
        state = "enabled: dormant, tier 3 fallback only"
    else:
        state = f"enabled (mode {mode})"
    print(f"jev: {state}")
    print(f"model: {config['jev']['model']}  endpoint: {config['jev']['endpoint']}")
    return 0


def _worker_settings(config: dict[str, Any], which: str) -> tuple[dict[str, Any], str]:
    if which == "local":
        return config["local"], "local-worker"
    return cfg.tier_settings(config, which), f"tier{which}-worker"


def _worker(config: dict[str, Any], which: str = "local") -> PersistentWorker:
    settings, name = _worker_settings(config, which)
    return PersistentWorker(
        settings.get("python") or sys.executable, cfg.runtime_dir(),
        settings.get("idle_timeout_s", 900), name=name,
    )  # fmt: skip


def _worker_status(config: dict[str, Any], which: str = "local") -> str:
    settings, _ = _worker_settings(config, which)
    if not settings.get("persistent"):
        return "oneshot (model loaded per decision)"
    pid = _worker(config, which).pid()
    state = f"running (pid {pid})" if pid else "not running (starts on first use)"
    return f"persistent, {state}, idle exit after {settings.get('idle_timeout_s', 900):g}s"


WORKERS = ("local", *cfg.TIER_NAMES)


def cmd_local_status(args: argparse.Namespace) -> int:
    config = cfg.load()
    for which in WORKERS:
        print(
            f"{'local' if which == 'local' else 'tier ' + which}: {_worker_status(config, which)}"
        )
    return 0


def cmd_local_stop(args: argparse.Namespace) -> int:
    config = cfg.load()
    for which in WORKERS if args.tier is None else (args.tier,):
        stopped = _worker(config, which).stop()
        label = "local" if which == "local" else f"tier {which}"
        print(f"{label} worker {'stopped' if stopped else 'was not running'}")
    return 0


def cmd_local_pull(args: argparse.Namespace) -> int:
    try:
        import nobodywho
    except ImportError:
        print("nobodywho is not installed in this interpreter", file=sys.stderr)
        return 1
    from .gguf_info import describe

    config = cfg.load()
    section = config["local"] if args.tier == "local" else config["tiers"].get(args.tier, {})
    source = args.source or section.get("source")
    if not source:
        print(f"no --source given and tier {args.tier} has none configured", file=sys.stderr)
        return 2
    path = Path(nobodywho.download_model(source))
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if args.expect_sha256 and digest.hexdigest() != args.expect_sha256.lower():
        print(f"sha256 mismatch: got {digest.hexdigest()}; config not changed", file=sys.stderr)
        return 1
    info = describe(path)
    pinned = {
        "source": source,
        "model_path": str(path),
        "model_sha256": digest.hexdigest(),
        "model_info": info,
    }
    cfg.update({"local": pinned} if args.tier == "local" else {"tiers": {args.tier: pinned}})
    print(f"{'local' if args.tier == 'local' else 'tier ' + args.tier} model: {path}")
    print(f"name: {info.get('name')}  quantization: {info.get('quantization')}")
    print(f"size: {path.stat().st_size} bytes  sha256: {digest.hexdigest()}")
    return 0


def cmd_install_rule(args: argparse.Namespace) -> int:
    dest = Path(args.dest).expanduser() if args.dest else DEFAULT_RULE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / RULE_NAME
    with resources.as_file(resources.files("decision_router") / "cline_rule.md") as src:
        shutil.copyfile(src, target)
    print(f"installed {target}")
    return 0


def build_pruner(config: dict[str, Any]) -> Pruner:
    pc = config["prune"]
    tiers = []
    for t in cfg.TIER_NAMES:
        settings = pc.get(f"tier{t}", {})
        if settings.get("enabled", True) is False:
            continue
        provider = NobodyWhoProvider.for_tier(config, t)
        tiers.append(
            Tier(int(t), "nobodywho", provider.model_name,
                 local_judge(provider, timeout_s=settings.get("timeout_s")),
                 max_blocks=int(settings.get("max_blocks", 8)))
        )  # fmt: skip
    jev = JevProvider.from_config(config)
    return Pruner(
        tiers,
        Tier(3, "jev", config["jev"]["model"], jev_judge(jev, jev.secret_literals()),
             max_blocks=int(pc.get("jev_max_blocks", 60))),
        jev_enabled=jev_enabled(config),
        block_lines=int(pc.get("block_lines", 30)),
        hard_cap_factor=float(pc.get("hard_cap_factor", 2.0)),
        archive_dir=cfg.prune_archive_dir() if pc.get("archive", True) else None,
        secret_literals=jev.secret_literals(),
    )  # fmt: skip


def _prune_text(args: argparse.Namespace, config: dict[str, Any], text: str) -> tuple[str, Any]:
    """Prunes `text`; on any failure returns it unchanged (the tool must keep working)."""
    try:
        request = PruneRequest(
            output=text, caller=caller_name(args.caller),
            command=args.command, goal=args.goal,
            budget_chars=int(args.budget_chars or config["prune"]["budget_chars"]),
            preserve=tuple(args.preserve or ()),
        )  # fmt: skip
        pruner = build_pruner(config)
        literals = JevProvider.from_config(config).secret_literals()
        result = pruner.prune(request)
        ledger = Ledger(cfg.ledger_path(), literals, request.caller)
        ledger.write([ledger.prune_receipt(request, result)])
        return result.text, result
    except Exception as error:  # noqa: BLE001 - pruning must never break the calling tool
        return _native_prune_after_error(text, args, config, error)


def _native_prune_after_error(
    text: str, args: argparse.Namespace, config: dict[str, Any], error: Exception
) -> tuple[str, PruneResult]:
    """A broken provider/config still gets a deterministic, bounded native excerpt."""
    try:
        budget = max(
            MIN_BUDGET_CHARS,
            int(args.budget_chars or config["prune"]["budget_chars"]),
        )
    except (TypeError, ValueError, KeyError):
        budget = MIN_BUDGET_CHARS
    caller = caller_name(args.caller)
    request: Any
    if len(text) <= MAX_OUTPUT_CHARS:
        request = PruneRequest(
            output=text,
            caller=caller,
            budget_chars=budget,
        )
    else:
        from types import SimpleNamespace

        request = SimpleNamespace(
            output=text, preserve=(), caller=caller, command=None, id=os.urandom(8).hex()
        )
    pc = config.get("prune", {}) if isinstance(config, dict) else {}
    block_lines = int(pc.get("block_lines", 30))
    p = plan(request, block_lines)
    votes = {block: "drop" for block in p.blocks}
    if p.blocks:
        votes[p.blocks[0]] = "keep"
        votes[p.blocks[-1]] = "keep"
    excerpt, kept = fit(p, votes, budget, "decision prune (native)")
    cap_factor = float(pc.get("hard_cap_factor", 2.0))
    excerpt = hard_cap(excerpt, int(budget * cap_factor))
    fallback = Pruner([], jev_enabled=False, archive_dir=None)
    excerpt = fallback._finish(excerpt, None, len(kept), p)
    result = PruneResult(
        text=excerpt,
        provider="native",
        tier="native",
        original_chars=len(text),
        result_chars=len(excerpt),
        original_tokens=estimate_tokens(text),
        result_tokens=estimate_tokens(excerpt),
        kept_lines=len(kept),
        total_lines=len(text.splitlines()),
        fallback_reason="pruning_error",
        error=type(error).__name__,
        native_fallback=True,
        compression_ratio=round(len(excerpt) / len(text), 4) if text else 1.0,
        attempts=[
            {
                "tier": "native",
                "provider": "native",
                "outcome": "accepted",
                "reason": "pruning_error",
            }
        ],
    )
    try:
        ledger = Ledger(cfg.ledger_path(), caller=caller)
        ledger.write([ledger.prune_receipt(request, result)])
    except Exception as accounting_error:  # noqa: BLE001 - never break the calling tool
        import logging

        logging.getLogger(__name__).debug(
            "Could not write native prune receipt (%s)", type(accounting_error).__name__
        )
    return excerpt, result


def cmd_prune(args: argparse.Namespace) -> int:
    config = cfg.load()
    if args.action == "status":
        return cmd_prune_status(config)
    if args.action == "test":
        return cmd_prune_test(args, config)
    nested = os.environ.get("DECISION_PRUNE_ACTIVE") == "1"
    if args.run:
        env = dict(os.environ, DECISION_PRUNE_ACTIVE="1")
        try:
            child = subprocess.Popen(args.run, stdout=subprocess.PIPE, env=env)
        except OSError as error:
            print(f"decision prune: cannot run {args.run[0]}: {error.strerror}", file=sys.stderr)
            return 127
        buffer, streaming = bytearray(), False
        assert child.stdout is not None
        fd = child.stdout.fileno()
        for chunk in iter(lambda: os.read(fd, 1 << 16), b""):
            if not streaming and len(buffer) + len(chunk) > STREAM_AFTER_BYTES:
                # Too big to hold: stream the rest untouched rather than buffer it all.
                sys.stdout.buffer.write(bytes(buffer))
                buffer, streaming = bytearray(), True
            if streaming:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            else:
                buffer += chunk
        code = child.wait()
        if streaming:
            return code if code >= 0 else 128 - code
        raw = bytes(buffer)
        if args.command is None:
            args.command = " ".join(args.run)[:300]
    else:
        raw, code = sys.stdin.buffer.read(), 0
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        sys.stdout.buffer.write(raw)  # binary output passes through untouched
        return code if code >= 0 else 128 - code
    if nested:
        out, result = text, None
    else:
        out, result = _prune_text(args, config, text)
    if args.json and result is not None:
        _print(result.to_dict())
    else:
        sys.stdout.write(out if out.endswith("\n") or not out else out + "\n")
    sys.stdout.flush()
    return code if code >= 0 else 128 - code


def cmd_prune_status(config: dict[str, Any]) -> int:
    pc = config["prune"]
    print("operation: prune (extractive; separate from `decision ask`)")
    print(f"budget: {pc['budget_chars']} chars, blocks of {pc['block_lines']} lines, "
          f"hard cap x{pc['hard_cap_factor']}")  # fmt: skip
    for t in cfg.TIER_NAMES:
        settings = pc.get(f"tier{t}", {})
        provider = NobodyWhoProvider.for_tier(config, t)
        ok, why = provider.available()
        state = "disabled" if settings.get("enabled", True) is False else ("ready" if ok else why)
        print(f"tier {t}: nobodywho {provider.model_name or '-'} "
              f"(max {settings.get('max_blocks')} blocks, timeout {settings.get('timeout_s')}s) [{state}]")  # fmt: skip
    print(f"tier 3: typesafe {config['jev']['model']} "
          f"[{'dormant fallback' if jev_enabled(config) else 'DISABLED (hard)'}]")  # fmt: skip
    print("final fallback: native truncation (critical lines + head/tail)")
    print(f"archive: {cfg.prune_archive_dir() if pc.get('archive', True) else 'off'}")
    return 0


def cmd_prune_test(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from .prune_corpus import cases

    pruner = build_pruner(config)
    pruner.archive_dir = None
    failures = 0
    for name, command, output, must_keep in cases():
        request = PruneRequest(output=output, caller="selftest", command=command,
                               budget_chars=int(args.budget_chars or config["prune"]["budget_chars"]))  # fmt: skip
        result = pruner.prune(request)
        missing = [m for m in must_keep if m not in result.text]
        failures += bool(missing)
        print(f"{'ok ' if not missing else 'MISSING'} {name:12s} {result.original_chars:>7} -> "
              f"{result.result_chars:>6} chars via {result.provider}"
              f"{'/' + str(result.model) if result.model else ''} tier={result.tier} "
              f"{result.latency_ms} ms"
              f"{' fallback=' + str(result.fallback_reason) if result.fallback_reason else ''}"
              f"{' missing=' + repr(missing) if missing else ''}")  # fmt: skip
    print(f"jev calls: {pruner.jev_calls}")
    return 1 if failures else 0


PRUNE_CALLERS = ("cline", "codex", "claude", "csmart", "freebuff", "opencode2")

SHIM = """#!/usr/bin/bash
# decision-router shell shim for {caller} (generated by `decision adapter shim {caller}`).
# Runs the real bash. When stdout is captured (not a terminal) and bash was asked to
# run a command string (-c), the command runs under the shared `decision prune`,
# which keeps the exit status and stderr and only shortens long stdout.
# Rollback: stop pointing {caller} at this file.
wants_prune=0
if [ "${{DECISION_PRUNE_ACTIVE:-}}" != 1 ] && [ ! -t 1 ]; then
    for arg in "$@"; do
        case "$arg" in
            --) break ;;
            -*c*) case "$arg" in --*) ;; *) wants_prune=1 ;; esac ;;
            -*) ;;
            *) break ;;
        esac
    done
fi
if [ "$wants_prune" = 1 ] && [ -x "{decision}" ]; then
    exec "{decision}" prune --caller {caller} -- /usr/bin/bash "$@"
fi
exec /usr/bin/bash "$@"
"""


def shim_path(caller: str) -> Path:
    return cfg.data_dir() / "shims" / caller / "bash"


def cmd_adapter(args: argparse.Namespace) -> int:
    decision = shutil.which("decision") or str(Path.home() / ".local/bin/decision")
    path = Path(args.path).expanduser() if args.path else shim_path(args.caller)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SHIM.format(caller=args.caller, decision=decision))
    path.chmod(0o755)
    print(path)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("provider", help="show or switch the decision provider mode")
    s.add_argument("mode", nargs="?", choices=cfg.MODES)
    s.set_defaults(func=cmd_provider)

    s = sub.add_parser("ask", help="route one decision request (JSON on stdin)")
    s.add_argument("--json")
    s.add_argument("--file")
    s.add_argument("--mode", choices=cfg.MODES)
    s.add_argument("--caller", help="calling environment, e.g. cline, codex, claude")
    s.add_argument("--bypass", action="store_true")
    s.add_argument("--user-directive")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("jev", help="the JEV fallback switch")
    s.add_argument("action", choices=("status", "enable", "disable"))
    s.set_defaults(func=cmd_jev)

    s = sub.add_parser("stats", help="usage statistics from the ledger")
    s.add_argument("--caller")
    s.add_argument("--since", help="ISO timestamp, e.g. 2026-09-24")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("doctor", help="check configuration and providers")
    s.add_argument("--live", action="store_true", help="also call every available provider")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("test", help="run a canned live decision through the router")
    s.add_argument("--mode", choices=cfg.MODES)
    s.set_defaults(func=cmd_test)

    s = sub.add_parser("benchmark", help="explicitly benchmark local decision and pruning models")
    s.add_argument(
        "kind", nargs="?", choices=("quality", "latency"), default="quality",
        help="quality: per-tier answers (default); latency: end-to-end timings per model",
    )  # fmt: skip
    s.add_argument("--operation", choices=("decision", "prune", "all"), default="all")
    s.add_argument(
        "--limit", type=int, default=0, help="limit each benchmark to the first N fixtures"
    )
    s.add_argument(
        "--include-jev", action="store_true", help="only consider JEV if globally enabled"
    )
    latency = s.add_argument_group("latency (never contacts JEV)")
    latency.add_argument("--models", nargs="*", metavar="LABEL",
                         help="configured models to include: local tier1 tier2 (default: all)")  # fmt: skip
    latency.add_argument("--model", action="append", metavar="GGUF", help="also measure this file")
    latency.add_argument("--installed", action="store_true", help="every cached GGUF model")
    latency.add_argument("--calls", type=int, default=20, help="warm decision calls")
    latency.add_argument("--prune-reps", type=int, default=5,
                         help="calls per small/medium prune fixture (large: 1)")  # fmt: skip
    latency.add_argument("--sizes", help="prune sizes, e.g. small,medium (default: all)")
    latency.add_argument("--budget-s", type=float, default=180,
                         help="stop repeating an operation after this many seconds")  # fmt: skip
    latency.add_argument("--timeout-s", type=float, default=600, help="per call")
    latency.add_argument("--cpu", action="store_true", help="load models without GPU offload")
    latency.add_argument("--json", action="store_true", help="the full report as JSON")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("ledger", help="show recent receipts")
    s.add_argument("-n", type=int, default=20)
    s.set_defaults(func=cmd_ledger)

    s = sub.add_parser("local", help="local model management")
    local = s.add_subparsers(dest="local_command", required=True)
    pull = local.add_parser("pull", help="download and pin a local model")
    pull.add_argument("--tier", choices=WORKERS, default="local")
    pull.add_argument("--source")
    pull.add_argument("--expect-sha256", help="refuse to pin the file unless it has this hash")
    pull.set_defaults(func=cmd_local_pull)
    local.add_parser("status", help="show the persistent workers").set_defaults(
        func=cmd_local_status
    )
    stop = local.add_parser("stop", help="stop persistent workers (frees RAM/VRAM)")
    stop.add_argument("--tier", choices=WORKERS)
    stop.set_defaults(func=cmd_local_stop)

    s = sub.add_parser("prune", help="shorten long tool output (extractive, local-first)")
    s.add_argument("action", nargs="?", choices=("status", "test"))
    s.add_argument("--caller")
    s.add_argument("--budget-chars", type=int)
    s.add_argument("--goal")
    s.add_argument("--command")
    s.add_argument("--preserve", action="append", help="keep lines containing this text")
    s.add_argument("--json", action="store_true", help="print the PruneResult as JSON")
    s.set_defaults(func=cmd_prune, run=None)

    s = sub.add_parser("adapter", help="thin adapters for other tools")
    adapter = s.add_subparsers(dest="adapter_command", required=True)
    shim = adapter.add_parser("shim", help="write the bash shim a tool uses for pruning")
    shim.add_argument("caller", choices=PRUNE_CALLERS)
    shim.add_argument("--path", help="where to write it (default: the shared shims directory)")
    shim.set_defaults(func=cmd_adapter)

    s = sub.add_parser("install-rule", help="install the Cline rule")
    s.add_argument("--dest")
    s.set_defaults(func=cmd_install_rule)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run: list[str] | None = None
    if argv[:1] == ["prune"] and "--" in argv:  # decision prune [opts] -- COMMAND ARGS...
        cut = argv.index("--")
        argv, run = argv[:cut], argv[cut + 1 :] or None
    args = parser().parse_args(argv)
    if run is not None:
        args.run = run
    return args.func(args)
