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

import json
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .sanitize import redact

MAX_OUTPUT_CHARS = 8 << 20
MIN_BUDGET_CHARS = 500
DEFAULT_BUDGET_CHARS = 12_000
BLOCK_PROMPT_CHARS = 900
EXCERPT_CHARS = 240  # per block in the one batched local judgement
KEEP, DROP = "keep", "drop"
GRAMMAR = f'root ::= "{KEEP}" | "{DROP}"'
REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
CALLER_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

SYSTEM_PROMPT = (
    "You shorten a command's output for a software coding agent. Errors, warnings, "
    "failures, file:line references, commands and summary lines are always kept "
    "separately; you only see the remaining blocks. For each block answer drop if it "
    "is routine noise the agent does not need (passing tests, progress, downloads, "
    "successful compile, install or build steps, repeated status lines) and keep if "
    "it holds information the agent may need (data the command was run to show, "
    "unusual messages, configuration, results or values)."
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
    # file.py:42 / src/lib.rs:10:5 / tsc file.ts(88,14); anchored at a token start, which
    # finds the same lines without retrying the greedy path match at every offset.
    r"|(?<![\w./-])[\w./-]+\.\w{1,6}(?::\d+|\(\d+,\d+\))"
    r"|^\s*caused\ by\b"
    r"|\bline\s+\d+\b"
    r"|^\s*(E\s{2,}|FAILED|ERROR|FAIL|---\s+FAIL)"  # pytest / go test failures
    r"|^\s*(\$|>|\#)\s+\S"  # echoed commands and prompts
    r"|^(diff\ --git|index\ \w+\.\.\w+|@@\ |\+\+\+\ |---\ a/|---\ /dev/null)"
    r"|^\s*\S.*\|\s+\d+\s*[+-]*\s*$"  # git --stat lines
    r"|\b\d+\s+files?\s+changed\b"
    r"|^\s*(npm|yarn|pnpm)\s+(ERR|WARN)"
    r"|\berror(\[\w+\])?:"
    r"|\bTS\d{4}\b"
    r"|^\s*at\s+\S+\s*\(.*:\d+"  # JS stack frames
    r"|^\s*(STDOUT|STDERR):\s*$"  # combined command streams
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
        if not isinstance(self.caller, str) or not CALLER_ID.fullmatch(self.caller):
            raise ValueError("caller must be a short identifier")
        if not isinstance(self.id, str) or not REQUEST_ID.fullmatch(self.id):
            raise ValueError("id must be a short request identifier")
        for field_name, value in (("command", self.command), ("goal", self.goal)):
            if value is not None and (not isinstance(value, str) or len(value) > 2_000):
                raise ValueError(f"{field_name} must be text no longer than 2000 characters")
        if not isinstance(self.preserve, (tuple, list)):
            raise TypeError("preserve hints must be a list or tuple")
        if any(not isinstance(p, str) or not p for p in self.preserve):
            raise ValueError("preserve hints must be non-empty strings")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be an object")
        try:
            metadata_size = len(json.dumps(self.metadata, ensure_ascii=False))
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be JSON serializable") from error
        if metadata_size > 2_000:
            raise ValueError("metadata exceeds 2000 characters")


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
    native_fallback: bool = False
    compression_ratio: float = 1.0
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be text")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty name")
        for name, value in (
            ("model", self.model),
            ("fallback_reason", self.fallback_reason),
            ("error", self.error),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be text or None")
        if self.tier is not None and (
            not isinstance(self.tier, (int, str)) or isinstance(self.tier, bool)
        ):
            raise TypeError("tier must be an integer, string, or None")
        for name in (
            "original_chars",
            "result_chars",
            "original_tokens",
            "result_tokens",
            "kept_lines",
            "total_lines",
            "latency_ms",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if not isinstance(self.compression_ratio, (int, float)) or isinstance(
            self.compression_ratio, bool
        ):
            raise TypeError("compression_ratio must be numeric")
        if not 0 <= self.compression_ratio <= 1:
            raise ValueError("compression_ratio must be within [0, 1]")
        if not isinstance(self.jev_used, bool) or not isinstance(self.native_fallback, bool):
            raise TypeError("provider flags must be booleans")
        if not isinstance(self.attempts, list) or any(
            not isinstance(a, dict) for a in self.attempts
        ):
            raise TypeError("attempts must be a list of objects")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _PlanRequest(Protocol):
    @property
    def output(self) -> str: ...

    @property
    def preserve(self) -> tuple[str, ...] | list[str]: ...


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


# ---------------------------------------------------------------- planning


@dataclass
class Plan:
    lines: list[str]
    critical: set[int]
    blocks: list[tuple[int, int]]  # [start, end) of non-critical runs a provider judges
    repetitive: list[tuple[int, int]] = field(default_factory=list)  # dropped without a model
    required: set[int] = field(default_factory=set)  # lines whose text must survive verbatim


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def strip_ansi(text: str) -> str:
    """Colour and cursor escape sequences carry no evidence for the agent."""
    return _ANSI.sub("", text) if "\x1b" in text else text


_TEMPLATE = re.compile(r"0x[0-9a-f]+|\d+(\.\d+)*", re.IGNORECASE)


def template(line: str) -> str:
    return " ".join(_TEMPLATE.sub("#", line).split())


def is_repetitive(lines: list[str]) -> bool:
    """A run of lines that differ only in numbers (progress, counters, fetch logs)."""
    return len(lines) >= 5 and len({template(line) for line in lines}) <= 2


def plan(request: _PlanRequest, block_lines: int = 30, context: int = 2,
         head: int = 5, tail: int = 15, text: str | None = None) -> Plan:  # fmt: skip
    """Critical lines, judged blocks and repetitive runs of `text` (default: the output).

    Every line is matched against CRITICAL once. Runs of three or more identical
    consecutive lines keep their first line; the copies are omitted without a model.
    """
    lines = (request.output if text is None else text).splitlines()
    n = len(lines)
    hints = [h.lower() for h in request.preserve]
    required = {
        i for i, line in enumerate(lines)
        if CRITICAL.search(line) or (hints and any(h in line.lower() for h in hints))
    }  # fmt: skip
    critical: set[int] = set(range(min(head, n))) | set(range(max(0, n - tail), n))
    for i in required:
        critical.update(range(max(0, i - context), min(n, i + context + 1)))
    copies: set[int] = set()
    i = 1
    while i < n:
        if lines[i] and lines[i] == lines[i - 1]:
            j = i
            while j < n and lines[j] == lines[i - 1]:
                j += 1
            if j - i >= 2:
                copies.update(range(i, j))
            i = j
        else:
            i += 1
    critical -= copies
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if i in critical or i in copies:
            i += 1
            continue
        start = i
        while i < n and i not in critical and i not in copies and i - start < block_lines:
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
    return Plan(lines, critical, judged, repetitive, required - copies)


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


def excerpt(lines: list[str], block: tuple[int, int], limit: int = EXCERPT_CHARS) -> str:
    text = "\n".join(lines[block[0] : block[1]])
    if len(text) > limit:
        text = text[: limit * 2 // 3] + " [...] " + text[-(limit // 3) :]
    return text


def batch_prompt(request: PruneRequest, lines: list[str], blocks: list[tuple[int, int]]) -> str:
    """One prompt for every block a local tier judges: a single generation per prune."""
    parts = []
    if request.goal:
        parts.append(f"Agent goal: {request.goal.strip()[:500]}")
    if request.command:
        parts.append(f"Command: {request.command.strip()[:200]}")
    for k, b in enumerate(blocks, 1):
        parts.append(f"Block {k} (lines {b[0] + 1}-{b[1]} of {len(lines)}):\n{excerpt(lines, b)}")
    parts.append(
        f"Answer keep or drop for each of the {len(blocks)} blocks, in order, separated by spaces."
    )
    return "\n\n".join(parts)


def votes_grammar(count: int) -> str:
    """Exactly `count` space-separated keep/drop verdicts."""
    return "root ::= v" + ' " " v' * (count - 1) + f'\nv ::= "{KEEP}" | "{DROP}"'


def _marker(gap: int, label: str) -> str:
    return f"[... {gap} line{'s' if gap != 1 else ''} omitted by {label} ...]"


def assemble(lines: list[str], keep: set[int], label: str) -> str:
    out: list[str] = []
    gap = 0
    for i, line in enumerate(lines):
        if i in keep:
            if gap:
                out.append(_marker(gap, label))
                gap = 0
            out.append(line)
        else:
            gap += 1
    if gap:
        out.append(_marker(gap, label))
    return "\n".join(out)


def _gaps(n: int, keep: set[int]) -> dict[int, int]:
    """start -> end of every run of omitted lines."""
    gaps: dict[int, int] = {}
    i = 0
    while i < n:
        if i in keep:
            i += 1
            continue
        start = i
        while i < n and i not in keep:
            i += 1
        gaps[start] = i
    return gaps


def fit(
    p: Plan, votes: dict[tuple[int, int], str], budget: int, label: str
) -> tuple[str, set[int]]:
    """Keep critical lines and kept blocks; drop kept blocks middle-out until under budget.

    The assembled size is updated per dropped block (blocks never hold critical
    lines), so large outputs are assembled once instead of once per block.
    """
    keep = set(p.critical)
    kept_blocks = [b for b in p.blocks if votes.get(b, KEEP) == KEEP]
    for b in kept_blocks:
        keep.update(range(*b))
    lines, n = p.lines, len(p.lines)
    by_start = _gaps(n, keep)
    by_end = {end: start for start, end in by_start.items()}

    def marker(start: int, end: int) -> int:
        return len(_marker(end - start, label)) + 1

    size = sum(len(lines[i]) + 1 for i in keep) + sum(marker(s, e) for s, e in by_start.items())
    if size - 1 > budget:
        # Drop the blocks furthest from both ends first: the middle of a log is the least useful.
        kept_blocks.sort(key=lambda b: min(b[0], n - b[1]), reverse=True)
        for start, end in kept_blocks:
            keep.difference_update(range(start, end))
            size -= sum(len(lines[i]) + 1 for i in range(start, end))
            new_start, new_end = start, end
            if start in by_end:  # merge with the gap just before
                new_start = by_end.pop(start)
                size -= marker(new_start, start)
                del by_start[new_start]
            if end in by_start:  # and the gap just after
                new_end = by_start.pop(end)
                size -= marker(end, new_end)
                del by_end[new_end]
            by_start[new_start], by_end[new_end] = new_end, new_start
            size += marker(new_start, new_end)
            if size - 1 <= budget:
                break
    return assemble(lines, keep, label), keep


def candidates(p: Plan, budget: int, max_blocks: int, label: str) -> list[tuple[int, int]]:
    """The blocks whose verdict can change the result, nearest the ends first.

    `fit` keeps blocks from both ends inwards until the budget is full, so only
    blocks that could fit in the room left by the critical lines are worth a
    model's time. Twice that room is judged so dropped blocks can be replaced.
    """
    if max_blocks <= 0 or not p.blocks:
        return []
    room = budget - len(assemble(p.lines, p.critical, label))
    if room <= 0:
        return []
    n = len(p.lines)
    chosen: list[tuple[int, int]] = []
    total = 0
    for b in sorted(p.blocks, key=lambda b: (min(b[0], n - b[1]), b[0])):
        if len(chosen) >= max_blocks or total >= 2 * room:
            break
        chosen.append(b)
        total += sum(len(line) + 1 for line in p.lines[b[0] : b[1]])
    return sorted(chosen)


def hard_cap(text: str, cap: int) -> str:
    """Last resort when critical lines alone exceed the cap: head and tail, marked."""
    if len(text) <= cap:
        return text
    lines = text.splitlines(keepends=True)
    marker_reserve = 90
    edge_budget = max(1, (cap - marker_reserve) // 2)
    head: list[str] = []
    tail: list[str] = []
    head_size = 0
    tail_size = 0
    for line in lines:
        if head_size + len(line) > edge_budget:
            break
        head.append(line)
        head_size += len(line)
    for line in reversed(lines[len(head) :]):
        if tail_size + len(line) > edge_budget:
            break
        tail.insert(0, line)
        tail_size += len(line)
    while True:
        omitted = max(0, len(text) - head_size - tail_size)
        marker = f"[... {omitted} characters of critical output truncated ...]"
        separator_before = "" if not head or head[-1].endswith("\n") else "\n"
        separator_after = "" if not tail or marker.endswith("\n") else "\n"
        out = "".join(head) + separator_before + marker + separator_after + "".join(tail)
        if len(out) <= cap or not head and not tail:
            return out[:cap] if len(out) > cap else out
        if len(head) >= len(tail) and head:
            head_size -= len(head.pop())
        elif tail:
            tail_size -= len(tail.pop(0))


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

        p = plan(request, self.block_lines, text=strip_ansi(original))
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
            label = f"decision prune ({tier.provider})"
            judged = candidates(p, request.budget_chars, tier.max_blocks, label)
            try:
                if tier is self.jev:
                    self.jev_calls += 1
                votes = tier.judge(request, p, judged) if judged else {}
                if set(votes.values()) - {KEEP, DROP} or set(votes) - set(judged):
                    raise PruneTierError("invalid_output", "votes outside keep/drop")
            except PruneTierError as error:
                attempts.append({"tier": tier.tier, "provider": tier.provider, "model": tier.model,
                                 "outcome": "failed", "reason": error.reason,
                                 "latency_ms": _ms(t0)})  # fmt: skip
                previous_reason = error.reason
                continue
            text, keep = fit(p, votes, request.budget_chars, label)
            text = self._finish(hard_cap(text, self._cap(request)), archive, len(keep), p)
            missing = self._missing_required_facts(p, text)
            if missing:
                attempts.append({"tier": tier.tier, "provider": tier.provider, "model": tier.model,
                                 "outcome": "failed", "reason": "pruning_quality_failure",
                                 "missing_facts": len(missing), "latency_ms": _ms(t0)})  # fmt: skip
                previous_reason = "pruning_quality_failure"
                continue
            attempts.append({"tier": tier.tier, "provider": tier.provider, "model": tier.model,
                             "outcome": "accepted", "judged_blocks": len(judged),
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
                native_fallback=True,
                jev_used=any(a.get("provider") == "jev" and a["outcome"] != "skipped"
                             for a in attempts),
            ),
            started, len(keep),
        )  # fmt: skip

    @staticmethod
    def _missing_required_facts(p: Plan, text: str) -> list[str]:
        """Critical and hinted lines (matched once, in `plan`) absent from the result."""
        required = {p.lines[i] for i in p.required}
        return [line for line in required if line and line not in text]

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
                f.write(redact(request.output, self.secret_literals))
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
        if len(out) < len(original):
            return out
        return text if len(text) <= len(original) else original

    def _done(self, result: PruneResult, started: float, kept: int) -> PruneResult:
        result.result_chars = len(result.text)
        result.result_tokens = estimate_tokens(result.text)
        result.kept_lines = kept
        result.latency_ms = _ms(started)
        result.compression_ratio = (
            round(result.result_chars / result.original_chars, 4) if result.original_chars else 1.0
        )
        if result.error:
            result.error = redact(result.error, self.secret_literals)[:200]
        return result


def _ms(t0: float) -> int:
    return round((time.monotonic() - t0) * 1000)
