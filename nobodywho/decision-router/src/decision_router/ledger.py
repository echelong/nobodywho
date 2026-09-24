"""Append-only JSONL decision ledger.

Receipts hold hashes and outcomes, not content: no state, source code,
environment or credentials. The question text is redacted and truncated.
Writing never raises; a broken ledger must not break the agent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract import DecisionRequest, DecisionResult
from .sanitize import redact, redact_value

QUESTION_CHARS = 300
_DETAIL_KEYS = (
    "model_alias",
    "usage",
    "model_file",
    "model_name",
    "quantization",
    "model_sha256",
    "temperature",
    "seeds",
    "runtime",
    "load_ms",
    "sample_ms",
    "worker",
    "model_reused",
)


def repo_id(cwd: str) -> str | None:
    """The name of the enclosing git work tree, if any."""
    path = Path(cwd)
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate.name
    return None


class Ledger:
    def __init__(self, path: Path, secret_literals: tuple[str, ...] = ()) -> None:
        self.path = Path(path)
        self.secret_literals = secret_literals

    def request_fields(self, request: DecisionRequest, mode: str) -> dict[str, Any]:
        cwd = os.getcwd()
        return {
            "request_id": request.id,
            "cwd": cwd,
            "repo": repo_id(cwd),
            "mode": mode,
            "state_sha256": request.state_hash(),
            "payload_sha256": request.payload_hash(),
            "question_id": request.question_id,
            "question": redact(request.question, self.secret_literals)[:QUESTION_CHARS],
            "choices": list(request.choices),
            "allow_abstain": request.allow_abstain,
            "risk": request.risk,
        }

    def receipt(
        self,
        request: DecisionRequest,
        mode: str,
        role: str,
        result: DecisionResult | None,
        agreement: bool | None = None,
        skipped: str | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **self.request_fields(request, mode),
            "role": role,
        }
        if skipped:
            record.update(provider=None, skipped=skipped)
        if result is not None:
            record.update(
                provider=result.provider,
                model=result.model,
                choice=result.choice,
                abstain=result.abstain,
                confidence=result.confidence,
                confidence_kind=result.confidence_kind,
                distribution=result.distribution,
                votes=result.votes,
                samples=result.samples,
                latency_ms=result.latency_ms,
                fallback_reason=result.fallback_reason,
                error=result.error,
                details={k: result.details[k] for k in _DETAIL_KEYS if k in result.details},
            )
        record["agreement"] = agreement
        return redact_value(record, self.secret_literals)

    def write(self, records: list[dict[str, Any]]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def tail(self, count: int = 20) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-count:]
        except OSError:
            return []
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        return records
