"""Block judges for `decision prune`: local NobodyWho tiers and the JEV fallback.

Both only answer keep/drop per block; neither ever writes text into the result.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Protocol

from .providers.jev import JevProvider, load_key
from .prune import (
    DROP,
    GRAMMAR,
    KEEP,
    SYSTEM_PROMPT,
    Plan,
    PruneRequest,
    PruneTierError,
    block_prompt,
)
from .sanitize import redact

JEV_BATCH = 20
JEV_MAX_REQUESTS = 3

_LOCAL_REASONS = {
    "local_model_unavailable": "model_unavailable",
    "local_runtime_unavailable": "model_unavailable",
    "local_error": "worker_failure",
}


class PromptRunner(Protocol):
    """What a local tier needs to offer: NobodyWhoProvider.run_prompts."""

    def run_prompts(
        self, system_prompt: str, grammar: str, prompts: list[str], temperature: float,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...  # fmt: skip


def local_judge(provider: PromptRunner, temperature: float = 0.1, timeout_s: float | None = None):
    def judge(request: PruneRequest, p: Plan, blocks: list[tuple[int, int]]) -> dict:
        prompts = [block_prompt(request, p.lines, b) for b in blocks]
        try:
            output = provider.run_prompts(SYSTEM_PROMPT, GRAMMAR, prompts, temperature, timeout_s)
        except subprocess.TimeoutExpired:
            raise PruneTierError("timeout") from None
        except Exception as error:  # noqa: BLE001 - any worker problem escalates
            raise PruneTierError("worker_failure", type(error).__name__) from None
        if not isinstance(output, dict) or output.get("error"):
            reason = (
                output.get("reason", "local_error") if isinstance(output, dict) else "local_error"
            )
            raise PruneTierError(_LOCAL_REASONS.get(reason, "provider_error"))
        votes = output.get("outputs")
        if not isinstance(votes, list) or len(votes) != len(blocks):
            raise PruneTierError("invalid_output", "wrong number of votes")
        return dict(zip(blocks, votes))

    return judge


def jev_judge(provider: JevProvider, literals: tuple[str, ...] = ()):
    """Keep/drop per block via JEV choice questions; secrets are redacted before sending."""

    def judge(request: PruneRequest, p: Plan, blocks: list[tuple[int, int]]) -> dict:
        key = load_key(provider.key_file)
        if not key:
            raise PruneTierError("model_unavailable", "TypeSafe key is missing")
        secrets = (*literals, key)
        votes: dict[tuple[int, int], str] = {}
        batches = [blocks[i : i + JEV_BATCH] for i in range(0, len(blocks), JEV_BATCH)]
        for batch in batches[:JEV_MAX_REQUESTS]:
            questions = {
                f"block_{i}": {
                    "type": "choice",
                    "instructions": "Does the coding agent need this block of command output?",
                    "criteria": {
                        KEEP: "It holds results, errors, warnings, paths, identifiers, counts or status.",
                        DROP: "It is repetitive progress, download or build noise.",
                    },
                }
                for i, _ in enumerate(batch)
            }
            state: dict[str, Any] = {
                "goal": redact(request.goal or "", secrets)[:500],
                "command": redact(request.command or "", secrets)[:200],
                "blocks": {
                    f"block_{i}": redact(block_prompt(request, p.lines, b), secrets)
                    for i, b in enumerate(batch)
                },
            }
            body = json.dumps({"model": provider.model, "state": state, "questions": questions})
            headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
            try:
                status, text = provider.transport(
                    provider.endpoint, headers, body.encode(), provider.timeout_s
                )
            except TimeoutError:
                raise PruneTierError("timeout") from None
            except Exception:  # noqa: BLE001 - network problems end the tier
                raise PruneTierError("provider_error", "JEV unreachable") from None
            if status != 200:
                raise PruneTierError("provider_error", f"HTTP {status}")
            try:
                answers = json.loads(text)["answers"]
                for i, b in enumerate(batch):
                    choice = answers[f"block_{i}"]["choice"]
                    if choice not in (KEEP, DROP):
                        raise ValueError(choice)
                    votes[b] = choice
            except (ValueError, KeyError, TypeError):
                raise PruneTierError("invalid_output", "malformed JEV answer") from None
        return votes

    return judge
