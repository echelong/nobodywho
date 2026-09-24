"""`decision benchmark latency`: end-to-end latency and quality per local model and operation.

Every call is measured the way a coding client makes it: a fresh `decision`
process (start-up, config, socket, persistent worker, inference, validation,
ledger) against a private temporary config in which the measured model is
tier 1, there is no tier 2, and JEV is disabled with no key file. TypeSafe is
never contacted and the real config, ledger and workers are not touched.

Rows: decision (cold, then warm calls over the benchmark cases), prune by
output size (small/medium/large fixtures), and prune:judge (keep/drop accuracy
of the one batched judgement on labelled blocks).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config as cfg
from .gguf_info import describe

SIZES = ("small", "medium", "large")
# Escalation reasons that mean the local tier produced no answer at all.
FAILURES = ("provider_error", "timeout", "invalid_output", "worker_failure", "model_unavailable")


@dataclass
class Model:
    label: str
    path: str
    use_gpu: bool
    info: dict[str, Any]

    @property
    def name(self) -> str:
        return self.info.get("name") or Path(self.path).stem


def percentile(values: list[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _installed() -> list[str]:
    try:
        import nobodywho

        return [path for path, _ in nobodywho.get_cached_models() if path.endswith(".gguf")]
    except Exception:  # noqa: BLE001 - fall back to the default cache location
        root = Path.home() / ".cache/nobodywho/models"
        return [str(p) for p in root.rglob("*.gguf")] if root.is_dir() else []


def select_models(
    config: dict[str, Any], labels: list[str], paths: list[str], installed: bool, cpu: bool
) -> list[Model]:
    """Configured models (local, tiers) and extra files; smallest first, no duplicates.

    With no selection at all, every configured model is measured.
    """
    found: dict[str, str] = {}
    configured = {"local": config["local"]}
    configured.update({f"tier{t}": config["tiers"].get(t, {}) for t in cfg.TIER_NAMES})
    everything = not labels and not paths and not installed
    for label, settings in configured.items():
        path = settings.get("model_path")
        if path and (everything or label in labels) and os.path.isfile(path):
            found.setdefault(os.path.realpath(path), label)
    for path in paths + (_installed() if installed else []):
        if os.path.isfile(path):
            found.setdefault(os.path.realpath(path), "extra")
    models = [Model(label, path, not cpu, describe(path)) for path, label in found.items()]
    return sorted(models, key=lambda m: os.path.getsize(m.path))


class Session:
    """A private router environment in which one model is tier 1."""

    def __init__(self, model: Model, timeout_s: float) -> None:
        base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
        self.root = Path(tempfile.mkdtemp(prefix="drb-", dir=base))
        self.model = model
        for sub in ("cfg", "state", "run"):
            (self.root / sub).mkdir(mode=0o700)
        tier = {
            "model_path": model.path,
            "model_info": model.info,
            "use_gpu": model.use_gpu,
            "persistent": True,
            "idle_timeout_s": 600,
            "timeout_s": timeout_s,
        }
        config = {
            "mode": "local-first",
            "jev": {"enabled": False, "key_file": str(self.root / "no-key")},
            "tiers": {"1": tier, "2": {"model_path": None}},
            "prune": {"archive": False, "tier1": {"timeout_s": timeout_s},
                      "tier2": {"enabled": False}},
        }  # fmt: skip
        (self.root / "cfg" / "config.json").write_text(json.dumps(config))
        self.env = {
            k: v for k, v in os.environ.items()
            if k not in ("TYPESAFE_API_KEY", "DECISION_ROUTER_MODE", "DECISION_ROUTER_BYPASS",
                         "DECISION_ROUTER_ACTIVE", "DECISION_PRUNE_ACTIVE")
        }  # fmt: skip
        self.env.update(
            DECISION_ROUTER_CONFIG_DIR=str(self.root / "cfg"),
            DECISION_ROUTER_STATE_DIR=str(self.root / "state"),
            DECISION_ROUTER_RUNTIME_DIR=str(self.root / "run"),
            DECISION_ROUTER_CAPTURE_WORKER_LOG="1",
        )
        self.problems: list[str] = []

    def note_worker_log(self) -> None:
        """Keeps out-of-memory and load failures the worker printed (before it is replaced)."""
        try:
            log = (self.root / "run" / "tier1-worker.log").read_text(errors="replace")
        except OSError:
            return
        for pattern, label in ((r"ErrorOutOfDeviceMemory|out of device memory", "GPU OOM"),
                               (r"Failed to load model", "load failed")):  # fmt: skip
            if re.search(pattern, log, re.IGNORECASE) and label not in self.problems:
                self.problems.append(label)

    def call(self, args: list[str], stdin: str, timeout_s: float) -> tuple[float, dict[str, Any]]:
        started = time.perf_counter()
        done = subprocess.run(
            [sys.executable, "-m", "decision_router", *args], input=stdin, capture_output=True,
            text=True, env=self.env, timeout=timeout_s, check=False,
        )  # fmt: skip
        wall = (time.perf_counter() - started) * 1000
        try:
            return wall, json.loads(done.stdout)
        except ValueError:
            return wall, {"error": (done.stderr or done.stdout)[-300:]}

    def worker_pid(self) -> int | None:
        try:
            return int((self.root / "run" / "tier1-worker.pid").read_text())
        except (OSError, ValueError):
            return None

    def stop(self) -> None:
        self.note_worker_log()
        subprocess.run([sys.executable, "-m", "decision_router", "local", "stop"],
                       env=self.env, capture_output=True, check=False, timeout=30)  # fmt: skip

    def close(self) -> None:
        self.stop()
        shutil.rmtree(self.root, ignore_errors=True)


def memory(pid: int | None) -> tuple[float | None, int | None]:
    """(worker RSS MiB, worker VRAM MiB); VRAM needs nvidia-smi."""
    rss = vram = None
    if pid:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            found = re.search(r"^VmRSS:\s+(\d+)", status, re.MULTILINE)
            rss = round(int(found.group(1)) / 1024) if found else None
        except OSError:
            pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout  # fmt: skip
            vram = next((int(m) for p, m in (line.split(",") for line in out.splitlines())
                         if p.strip() == str(pid)), 0)  # fmt: skip
        except (OSError, ValueError, subprocess.TimeoutExpired):
            vram = None
    return rss, vram


def offload(gpu_layers: Any, model: Model) -> str:
    layers = model.info.get("layers")
    if not model.use_gpu or gpu_layers == 0:
        return "CPU"
    if not isinstance(gpu_layers, int):
        return "unknown"
    if isinstance(layers, int):
        return f"{min(gpu_layers, layers)}/{layers} layers"
    return f"{gpu_layers} layers"


def _summary(walls: list[float]) -> dict[str, Any]:
    def r(value: float | None) -> int | None:
        return None if value is None else round(value)

    return {"calls": len(walls), "p50_ms": r(percentile(walls, 0.5)),
            "p95_ms": r(percentile(walls, 0.95)), "p99_ms": r(percentile(walls, 0.99))}  # fmt: skip


def bench_decision(session: Session, calls: int, budget_s: float, timeout_s: float) -> dict:
    from .benchmark import CASES

    def request(i: int) -> str:
        case = CASES[i % len(CASES)]
        return json.dumps({"id": f"lat-{i}", "question_id": "benchmark", "state": case["state"],
                           "question": case["question"], "choices": case["choices"],
                           "allow_abstain": True, "risk": "low"})  # fmt: skip

    session.stop()
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for i in range(calls + 1):
        if i > 1 and time.monotonic() - started > budget_s:
            break
        wall, out = session.call(["ask", "--caller", "benchmark"], request(i), timeout_s)
        attempt = (out.get("attempts") or [{}])[0]
        decision = out.get("decision") or {}
        details = decision.get("details") or {}
        expected = CASES[i % len(CASES)].get("expected")
        chosen = "ABSTAIN" if attempt.get("abstain") else attempt.get("choice")
        failed = not attempt or attempt.get("escalation_reason") in FAILURES
        answered = decision.get("provider") == "nobodywho"  # details belong to tier 1
        rows.append({
            "wall_ms": wall, "provider_ms": attempt.get("latency_ms"),
            "sample_ms": details.get("sample_ms") if answered else None,
            "gpu_layers": details.get("gpu_layers") if answered else None,
            "load_ms": details.get("load_ms") if answered else None,
            "expected": expected, "chosen": chosen,
            "error": (attempt.get("escalation_reason") or out.get("error") or "no attempt")
            if failed else None,
        })  # fmt: skip
    rss, vram = memory(session.worker_pid())
    cold, warm = rows[0], rows[1:]
    objective = [r for r in warm if r["expected"] is not None and not r["error"]]
    samples = [s for r in warm for s in (r["sample_ms"] or [])]
    overhead = [r["wall_ms"] - r["provider_ms"] for r in warm if r["provider_ms"] is not None]
    gpu_layers = next((r["gpu_layers"] for r in rows if r["gpu_layers"] is not None), None)
    return {
        "operation": "decision", "cold_ms": round(cold["wall_ms"]),
        "cold_load_ms": cold["load_ms"], "first_warm_ms": round(warm[0]["wall_ms"]) if warm else None,
        **_summary([r["wall_ms"] for r in warm]),
        "sample_p50_ms": round(percentile(samples, 0.5) or 0),
        "samples_per_call": round(len(samples) / max(1, sum(1 for r in warm if r["sample_ms"])), 2),
        "cli_overhead_p50_ms": round(percentile(overhead, 0.5) or 0),
        "quality_pass": sum(r["chosen"] == r["expected"] for r in objective),
        "quality_total": len(objective), "errors": sum(1 for r in rows if r["error"]),
        "error_reasons": sorted({str(r["error"]) for r in rows if r["error"]}),
        "rss_mib": rss, "vram_mib": vram, "gpu_layers": gpu_layers,
    }  # fmt: skip


def bench_prune(session: Session, reps: int, sizes: tuple[str, ...], budget_s: float,
                timeout_s: float) -> list[dict]:  # fmt: skip
    from .prune_corpus import sized_cases

    budget = cfg.DEFAULTS["prune"]["budget_chars"]
    session.stop()
    cold_ms: float | None = None
    rows = []
    for size in sizes:
        walls, passes, ratios, judged = [], 0, [], []
        started = time.monotonic()
        for _, (name, command, output, required) in (c for c in sized_cases() if c[0] == size):
            source = set(output.splitlines())
            for rep in range(1 if size == "large" else reps):
                if rep and time.monotonic() - started > budget_s:
                    break
                wall, out = session.call(
                    ["prune", "--json", "--caller", "benchmark", "--command", command], output,
                    timeout_s,
                )  # fmt: skip
                if cold_ms is None:
                    cold_ms = wall
                    continue  # the cold call is reported separately
                text = out.get("text", "")
                invented = [line for line in text.splitlines() if line not in source
                            and not line.startswith(("[... ", "[decision prune:"))]  # fmt: skip
                kept = all(fact in text for fact in required)
                bounded = len(text) <= 2 * budget + 200
                model_ok = out.get("provider") == "nobodywho" and out.get("tier") == 1
                passes += kept and not invented and bounded and model_ok
                walls.append(wall)
                ratios.append(out.get("compression_ratio", 1.0))
                judged += [a.get("judged_blocks") for a in out.get("attempts", [])
                           if a.get("outcome") == "accepted" and a.get("tier") == 1]  # fmt: skip
        rss, vram = memory(session.worker_pid())
        rows.append({
            "operation": f"prune:{size}", "cold_ms": round(cold_ms) if cold_ms and not rows
            else None, **_summary(walls), "quality_pass": passes, "quality_total": len(walls),
            "compression_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "judged_blocks_max": max((j for j in judged if j is not None), default=None),
            "rss_mib": rss, "vram_mib": vram,
        })  # fmt: skip
    return rows


def bench_judgement(session: Session, timeout_s: float) -> dict:
    """Keep/drop accuracy of the batched local judgement on labelled blocks."""
    from .providers.nobodywho import NobodyWhoProvider, PersistentWorker
    from .prune import Plan, PruneRequest, PruneTierError
    from .prune_corpus import judgement_cases
    from .prune_providers import local_judge

    provider = NobodyWhoProvider(
        session.model.path, use_gpu=session.model.use_gpu, timeout_s=timeout_s,
        runner=PersistentWorker(sys.executable, session.root / "run", 600, name="tier1-worker"),
    )  # fmt: skip
    judge = local_judge(provider, timeout_s=timeout_s)
    keep_ok = keep_total = drop_ok = drop_total = 0
    walls: list[float] = []
    failures = 0
    for command, labelled in judgement_cases():
        lines: list[str] = []
        blocks: list[tuple[int, int]] = []
        for _, block in labelled:
            blocks.append((len(lines), len(lines) + len(block)))
            lines += block
        started = time.perf_counter()
        try:
            votes = judge(PruneRequest(output="\n".join(lines), command=command),
                          Plan(lines, set(), blocks), blocks)  # fmt: skip
        except PruneTierError:
            failures += 1
            continue
        walls.append((time.perf_counter() - started) * 1000)
        for (label, _), block in zip(labelled, blocks):
            if label == "keep":
                keep_total += 1
                keep_ok += votes[block] == "keep"
            else:
                drop_total += 1
                drop_ok += votes[block] == "drop"
    return {
        "operation": "prune:judge", "cold_ms": None, **_summary(walls),
        "quality_pass": keep_ok + drop_ok, "quality_total": keep_total + drop_total,
        "useful_kept": f"{keep_ok}/{keep_total}", "noise_dropped": f"{drop_ok}/{drop_total}",
        "failures": failures, "measured": "judge call through the worker (no CLI start-up)",
    }  # fmt: skip


def run(config: dict[str, Any], args: Any) -> dict[str, Any]:
    models = select_models(config, args.models or [], args.model or [], args.installed, args.cpu)
    sizes = tuple(s for s in (args.sizes or "small,medium,large").split(",") if s in SIZES)
    report: dict[str, Any] = {"jev": "not contacted (disabled, no key, local-only benchmark)",
                              "models": []}  # fmt: skip
    for model in models:
        session = Session(model, args.timeout_s)
        entry: dict[str, Any] = {"model": model.name, "file": Path(model.path).name,
                                 "label": model.label, "use_gpu": model.use_gpu, "rows": []}  # fmt: skip
        try:
            gpu_layers = None
            if args.operation in ("decision", "all"):
                row = bench_decision(session, args.calls, args.budget_s, args.timeout_s)
                gpu_layers = row["gpu_layers"]
                entry["rows"].append(row)
            if args.operation in ("prune", "all"):
                entry["rows"] += bench_prune(session, args.prune_reps, sizes, args.budget_s,
                                             args.timeout_s)  # fmt: skip
                entry["rows"].append(bench_judgement(session, args.timeout_s))
                if gpu_layers is None:
                    _, out = session.call(["ask", "--caller", "benchmark"], json.dumps({
                        "question_id": "probe", "state": "probe", "question": "Pick one.",
                        "choices": {"first": "First.", "second": "Second."}}), args.timeout_s)  # fmt: skip
                    gpu_layers = ((out.get("decision") or {}).get("details") or {}).get(
                        "gpu_layers"
                    )
            entry["gpu_offload"] = offload(gpu_layers, model)
        except subprocess.TimeoutExpired:
            entry["error"] = "timeout"
        finally:
            session.close()
        if session.problems:
            entry["problems"] = session.problems
            entry["gpu_offload"] = (
                f"{entry.get('gpu_offload', 'unknown')} ({', '.join(session.problems)})"
            )
        report["models"].append(entry)
    return report


def table(report: dict[str, Any]) -> str:
    head = ("MODEL", "OPERATION", "COLD", "P50", "P95", "P99", "RAM", "VRAM", "GPU OFFLOAD",
            "QUALITY PASS RATE")  # fmt: skip
    rows = [head]

    def ms(value: Any) -> str:
        if value is None:
            return "-"
        return f"{value / 1000:.1f}s" if value >= 10_000 else f"{value}ms"

    for entry in report["models"]:
        name = f"{entry['model']}{'' if entry['use_gpu'] else ' (cpu)'}"
        if entry.get("error"):
            rows.append((name, "-", "-", "-", "-", "-", "-", "-", "-", entry["error"]))
        for r in entry["rows"]:
            total = r.get("quality_total") or 0
            if r.get("calls") == 0 or (r.get("errors") and r["errors"] >= r.get("calls", 0)):
                reasons = ", ".join(r.get("error_reasons") or []) or "no successful call"
                rows.append((name, r["operation"], "-", "-", "-", "-", "-", "-",
                             entry.get("gpu_offload", "-"), f"FAILED: {reasons}"))  # fmt: skip
                continue
            quality = f"{r['quality_pass']}/{total} ({r['quality_pass'] / total:.0%})" if total \
                else "-"  # fmt: skip
            if r["operation"] == "prune:judge":
                quality += f" kept {r['useful_kept']} dropped {r['noise_dropped']}"
            elif r["operation"].startswith("prune:") and r.get("compression_ratio") is not None:
                quality += f" ratio {r['compression_ratio']:.2f}"
            rows.append((
                name, r["operation"], ms(r.get("cold_ms")), ms(r.get("p50_ms")),
                ms(r.get("p95_ms")), ms(r.get("p99_ms")),
                f"{r['rss_mib']}MiB" if r.get("rss_mib") else "-",
                f"{r['vram_mib']}MiB" if r.get("vram_mib") is not None else "-",
                entry.get("gpu_offload", "-"), quality,
            ))  # fmt: skip
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(head))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip()
                     for row in rows)  # fmt: skip
