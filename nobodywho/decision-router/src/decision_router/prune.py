"""Output pruning: a second operation, separate from closed-choice decisions.

A coding agent's long tool output is shortened *extractively*: every line in
the result is copied verbatim from the original, in order, and gaps are
marked. A model only votes keep/drop on blocks of lines that carry no
critical evidence. Critical lines (errors, failures, file:line references,
test ids, commands, exit statuses, warnings, diff headers, the head and the
tail) are kept deterministically, so no provider can summarise them away.

Providers are tried local-first: tier 1, tier 2, then JEV (only when enabled),
and finally a deterministic native truncation, which always succeeds, so a
pruning failure never breaks the calling tool.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .sanitize import redact

MAX_OUTPUT_CHARS = 8 << 20
MIN_BUDGET_CHARS = 500
DEFAULT_BUDGET_CHARS = 12_000
BLOCK_PROMPT_CHARS = 900
KEEP, DROP = "keep", "drop"
GRAMMAR = f'root ::= "{KEEP}" | "{DROP}"'

SYSTEM_PROMPT = (
    "You triage blocks of a command's output for a software coding agent. "
    "Answer keep if the block holds anything the agent may need: results, "
    "errors, warnings, file paths, line numbers, test names, identifiers, "
    "versions, counts or a final status. Answer drop only for repetitive "
    "progress, download or build noise that the agent does not need."
)

# Lines kept no matter what any provider says.
CRITICAL = re.compile(
    r"(?ix)"
    r"\b(error|errors|err!|fail|failed|failure|failures|fatal|panic|panicked|exception|traceback"
    r"|assert(ion)?(error)?|warning|warn|denied|refused|not\ found|no\ such|cannot|can't|unable"
    r"|segmentation|abort(ed)?|killed|timeout|timed\ out|deprecat\w*|critical|unresolved"
    r"|conflict|rejected|missing|invalid|undefined|mismatch)\b"
    r"|\b(exit(ed)?\s*(status|code)?|returned|status)\s*[:=]?\s*-?\d+"
    r"|(?<!\w)\d+\s+(?-i:passed|failed|skipped|xfailed|errors?|warnings?)\b"  # summaries
    r"|[\w./-]+\.\w{1,6}:\d+"  # file.py:42 / src/lib.rs:10:5
    r"|[\w./-]+\.\w{1,6}\(\d+,\d+\)"  # tsc: file.ts(88,14)
    r"|^\s*caused\ by\b"
    r"|\bline\s+\d+\b"
    r"|^\s*(E\s{2,}|FAILED|ERROR|FAIL|---\s+FAIL)"  # pytest / go test failures
    r"|^\s*(\$|>|\#)\s+\S"  # echoed commands and prompts
    r"|^(diff\ --git|index\ \w+\.\.\w+|@@\ |\+\+\+\ |---\ a/|---\ /dev/null)"
    r"|^\s*\S.*\|\s+\d+\s+[+-]*\s*$"  # git --stat lines
    r"|\b\d+\s+files?\s+changed\b"
    r"|^\s*(npm|yarn|pnpm)\s+(ERR|WARN)"
    r"|\berror(\[\w+\])?:"
    r"|\bTS\d{4}\b"
    r"|^\s*at\s+\S+\s*\(.*:\d+"  # JS stack frames
)


@dataclass(frozen=True)
class PruneRequest:
    output: str
    caller: str = "unknown"
    command: str | None = None
    goal: str | None = None
    budget_chars: int = DEFAULT_BUDGET_CHARS
    preserve: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise TypeError("output must be text")
        if len(self.output) > MAX_OUTPUT_CHARS:
            raise ValueError(f"output exceeds {MAX_OUTPUT_CHARS} characters")
        if not isinstance(self.budget_chars, int) or self.budget_chars < MIN_BUDGET_CHARS:
            raise ValueError(f"budget_chars must be an integer >= {MIN_BUDGET_CHARS}")
        if any(not isinstance(p, str) or not p for p in self.preserve):
            raise ValueError("preserve hints must be non-empty strings")


@dataclass
class PruneResult:
    text: str
    provider: str
    model: str | None = None
    tier: int | str | None = None
    original_chars: int = 0
    result_chars: int = 0
    original_tokens: int = 0
    result_tokens: int = 0
    kept_lines: int = 0
    total_lines: int = 0
    latency_ms: int = 0
    fallback_reason: str | None = None
    error: str | None = None
    jev_used: bool = False
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


# ---------------------------------------------------------------- planning


@dataclass
class Plan:
    lines: list[str]
    critical: set[int]
    blocks: list[tuple[int, int]]  # [start, end) of non-critical runs a provider judges
    repetitive: list[tuple[int, int]] = field(default_factory=list)  # dropped without a model


_TEMPLATE = re.compile(r"0x[0-9a-f]+|\d+(\.\d+)*", re.IGNORECASE)


def template(line: str) -> str:
    return " ".join(_TEMPLATE.sub("#", line).split())


def is_repetitive(lines: list[str]) -> bool:
    """A run of lines that differ only in numbers (progress, counters, fetch logs)."""
    return len(lines) >= 5 and len({template(line) for line in lines}) <= 2


def plan(request: PruneRequest, block_lines: int = 30, context: int = 2,
         head: int = 5, tail: int = 15) -> Plan:  # fmt: skip
    lines = request.output.splitlines()
    n = len(lines)
    hints = [h.lower() for h in request.preserve]
    critical: set[int] = set(range(min(head, n))) | set(range(max(0, n - tail), n))
    for i, line in enumerate(lines):
        low = line.lower()
        if CRITICAL.search(line) or any(h in low for h in hints):
            critical.update(range(max(0, i - context), min(n, i + context + 1)))
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if i in critical:
            i += 1
            continue
        start = i
        while i < n and i not in critical and i - start < block_lines:
            i += 1
        blocks.append((start, i))
    repetitive: list[tuple[int, int]] = []
    judged: list[tuple[int, int]] = []
    for b in blocks:
        if not is_repetitive(lines[b[0] : b[1]]):
            judged.append(b)
        elif repetitive and repetitive[-1][1] == b[0]:  # one run, however many blocks
            repetitive[-1] = (repetitive[-1][0], b[1])
        else:
            repetitive.append(b)
    # A repetitive run keeps its first line as an example of what was omitted.
    critical.update(b[0] for b in repetitive)
    return Plan(lines, critical, judged, repetitive)


def block_prompt(request: PruneRequest, lines: list[str], block: tuple[int, int]) -> str:
    text = "\n".join(lines[block[0] : block[1]])
    if len(text) > BLOCK_PROMPT_CHARS:
        text = text[: BLOCK_PROMPT_CHARS // 2] + "\n[...]\n" + text[-BLOCK_PROMPT_CHARS // 2 :]
    parts = []
    if request.goal:
        parts.append(f"Agent goal: {request.goal.strip()[:500]}")
    if request.command:
        parts.append(f"Command: {request.command.strip()[:200]}")
    parts.append(f"Output lines {block[0] + 1}-{block[1]} of {len(lines)}:")
    parts.append(text)
    parts.append("Answer keep or drop.")
    return "\n".join(parts)


def assemble(lines: list[str], keep: set[int], label: str) -> str:
    out: list[str] = []
    gap = 0
    for i, line in enumerate(lines):
        if i in keep:
            if gap:
                out.append(f"[... {gap} line{'s' if gap != 1 else ''} omitted by {label} ...]")
                gap = 0
            out.append(line)
        else:
            gap += 1
    if gap:
        out.append(f"[... {gap} line{'s' if gap != 1 else ''} omitted by {label} ...]")
    return "\n".join(out)


def fit(
    p: Plan, votes: dict[tuple[int, int], str], budget: int, label: str
) -> tuple[str, set[int]]:
    """Keep critical lines and kept blocks; drop kept blocks middle-out until under budget."""
    keep = set(p.critical)
    kept_blocks = [b for b in p.blocks if votes.get(b, KEEP) == KEEP]
    for b in kept_blocks:
        keep.update(range(*b))
    text = assemble(p.lines, keep, label)
    if len(text) <= budget:
        return text, keep
    n = len(p.lines)
    # Drop the blocks furthest from both ends first: the middle of a log is the least useful.
    kept_blocks.sort(key=lambda b: min(b[0], n - b[1]), reverse=True)
    for b in kept_blocks:
        keep.difference_update(range(*b))
        keep.update(p.critical & set(range(*b)))
        text = assemble(p.lines, keep, label)
        if len(text) <= budget:
            break
    return text, keep


def hard_cap(text: str, cap: int) -> str:
    """Last resort when critical lines alone exceed the cap: head and tail, marked."""
    if len(text) <= cap:
        return text
    half = cap // 2 - 60
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted} characters of critical output truncated ...]\n{text[-half:]}"


# ---------------------------------------------------------------- engine

# A block judge: (request, plan, blocks) -> {block: "keep"|"drop"}; raises PruneTierError.
Judge = Callable[[PruneRequest, Plan, list[tuple[int, int]]], dict[tuple[int, int], str]]


class PruneTierError(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason


@dataclass
class Tier:
    tier: int
    provider: str
    model: str | None
    judge: Judge
    max_blocks: int = 24


class Pruner:
    def __init__(
        self,
        tiers: list[Tier],
        jev: Tier | None = None,
        *,
        jev_enabled: bool = True,
        block_lines: int = 30,
        hard_cap_factor: float = 2.0,
        archive_dir: Path | None = None,
        secret_literals: tuple[str, ...] = (),
    ) -> None:
        self.tiers = tiers
        self.jev = jev
        self.jev_enabled = jev_enabled
        self.block_lines = block_lines
        self.hard_cap_factor = hard_cap_factor
        self.archive_dir = archive_dir
        self.secret_literals = secret_literals
        self.jev_calls = 0

    def prune(self, request: PruneRequest) -> PruneResult:
        started = time.monotonic()
        original = request.output
        total_lines = len(original.splitlines())

        def result(**fields: Any) -> PruneResult:
            return PruneResult(
                original_chars=len(original), original_tokens=estimate_tokens(original),
                total_lines=total_lines, **fields,
            )  # fmt: skip

        if len(original) <= request.budget_chars:
            return self._done(
                result(text=original, provider="none", fallback_reason="under_budget"),
                started, total_lines,
            )  # fmt: skip

        p = plan(request, self.block_lines)
        archive = self._archive(request)
        attempts: list[dict[str, Any]] = []
        chain = list(self.tiers)
        if self.jev is not None:
            chain.append(self.jev)
        previous_reason: str | None = None
        for tier in chain:
            if tier is self.jev and not self.jev_enabled:
                attempts.append({"tier": 3, "provider": "jev", "outcome": "skipped",
                                 "reason": "jev_disabled"})  # fmt: skip
                previous_reason = "jev_disabled"
                continue
            t0 = time.monotonic()
            candidates = sorted(p.blocks, key=lambda b: b[1] - b[0], reverse=True)[
                : tier.max_blocks
            ]
            try:
                if tier is self.jev:
                    self.jev_calls += 1
                votes = tier.judge(request, p, candidates) if candidates else {}
                if set(votes.values()) - {KEEP, DROP} or set(votes) - set(candidates):
                    raise PruneTierError("invalid_output", "votes outside keep/drop")
            except PruneTierError as error:
                attempts.append({"tier": tier.tier, "provider": tier.provider, "model": tier.model,
                                 "outcome": "failed", "reason": error.reason,
                                 "latency_ms": _ms(t0)})  # fmt: skip
                previous_reason = error.reason
                continue
            label = f"decision prune ({tier.provider})"
            text, keep = fit(p, votes, request.budget_chars, label)
            text = self._finish(hard_cap(text, self._cap(request)), archive, len(keep), p)
            attempts.append({"tier": tier.tier, "provider": tier.provider, "model": tier.model,
                             "outcome": "accepted", "judged_blocks": len(candidates),
                             "dropped_blocks": sum(v == DROP for v in votes.values()),
                             "latency_ms": _ms(t0)})  # fmt: skip
            return self._done(
                result(
                    text=text, provider=tier.provider, model=tier.model, tier=tier.tier,
                    kept_lines=len(keep), fallback_reason=previous_reason,
                    jev_used=tier is self.jev, attempts=attempts,
                ),
                started, len(keep),
            )  # fmt: skip

        # Native truncation: critical lines plus the head and tail blocks, always succeeds.
        blocks = sorted(p.blocks)
        votes = {b: DROP for b in blocks}
        if blocks:
            votes[blocks[0]] = KEEP
            votes[blocks[-1]] = KEEP
        text, keep = fit(p, votes, request.budget_chars, "decision prune (native)")
        text = self._finish(hard_cap(text, self._cap(request)), archive, len(keep), p)
        attempts.append({"tier": "native", "provider": "native", "outcome": "accepted"})
        return self._done(
            result(
                text=text, provider="native", tier="native", kept_lines=len(keep),
                fallback_reason=previous_reason or "no_provider", attempts=attempts,
                jev_used=any(a.get("provider") == "jev" and a["outcome"] != "skipped"
                             for a in attempts),
            ),
            started, len(keep),
        )  # fmt: skip

    def _cap(self, request: PruneRequest) -> int:
        return int(request.budget_chars * self.hard_cap_factor)

    def _archive(self, request: PruneRequest) -> Path | None:
        if self.archive_dir is None:
            return None
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.archive_dir / f"{request.caller}-{request.id}.txt"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(request.output)
            old = sorted(self.archive_dir.glob("*.txt"), key=lambda q: q.stat().st_mtime)
            for stale in old[:-200]:
                stale.unlink(missing_ok=True)
            return path
        except OSError:
            return None

    def _finish(self, text: str, archive: Path | None, kept: int, p: Plan) -> str:
        """Adds the footer; never returns something longer than the original."""
        original = "\n".join(p.lines)
        where = f"; full output: {archive}" if archive else ""
        out = f"{text}\n[decision prune: kept {kept} of {len(p.lines)} lines{where}]"
        return original if len(out) >= len(original) else out

    def _done(self, result: PruneResult, started: float, kept: int) -> PruneResult:
        result.result_chars = len(result.text)
        result.result_tokens = estimate_tokens(result.text)
        result.kept_lines = kept
        result.latency_ms = _ms(started)
        if result.error:
            result.error = redact(result.error, self.secret_literals)[:200]
        return result


def _ms(t0: float) -> int:
    return round((time.monotonic() - t0) * 1000)
