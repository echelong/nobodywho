"""Local NobodyWho provider: zero API cost, grammar-constrained, offline.

The model can only emit one of the allowed option ids (or ABSTAIN when
allowed) because every sample is decoded under a GBNF grammar of exactly
those ids. NobodyWho does not expose per-option probabilities, so this
provider never reports a probability. Instead it draws a few seeded samples,
permuting option order per sample to expose position bias, and reports the
winner's vote share as `sample_stability`. That number is a stability proxy,
not a calibrated probability, and is labelled accordingly.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..contract import ABSTAIN, SAMPLE_STABILITY, DecisionRequest, DecisionResult
from ..gguf_info import describe
from ..sanitize import short_error

NAME = "nobodywho"
WORKER = Path(__file__).resolve().parent.parent / "local_worker.py"

SYSTEM_PROMPT = (
    "You are a routing classifier for a software coding agent. Read the state, "
    "then answer the question with exactly one option id from the list. "
    "Output only the option id."
)

# (job, timeout_s) -> worker output dict; raises subprocess.TimeoutExpired on timeout.
Runner = Callable[[dict[str, Any], float], dict[str, Any]]


def grammar_for(allowed: tuple[str, ...]) -> str:
    """A GBNF grammar whose only sentences are the allowed ids."""
    return "root ::= " + " | ".join(f'"{option}"' for option in allowed)


def render_prompt(payload: dict[str, Any], order: list[str]) -> str:
    state = payload["state"]
    evidence = state["evidence"]
    if not isinstance(evidence, str):
        evidence = json.dumps(evidence, indent=1, ensure_ascii=False)
    lines = [f"State (risk: {state['risk']}):", evidence.strip(), ""]
    lines.append(f"Question: {payload['question'].strip()}")
    lines.append("")
    lines.append("Options:")
    for option in order:
        description = payload["options"].get(option)
        lines.append(f"- {option}: {description}" if description else f"- {option}")
    lines.append("")
    lines.append("Answer with exactly one option id.")
    return "\n".join(lines)


def plan_samples(request: DecisionRequest, samples: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic seeds and per-sample option orders (ABSTAIN stays last)."""
    payload = request.payload()
    plan = []
    for i in range(samples):
        sample_seed = seed + i
        order = list(request.choices)
        random.Random(sample_seed).shuffle(order)
        if request.allow_abstain:
            order.append(ABSTAIN)
        plan.append({"seed": sample_seed, "prompt": render_prompt(payload, order)})
    return plan


def subprocess_runner(python: str) -> Runner:
    def run(job: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        env = dict(os.environ, DECISION_ROUTER_ACTIVE="1")
        env.pop("TYPESAFE_API_KEY", None)  # the local worker never needs it
        completed = subprocess.run(
            [python, str(WORKER)],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        try:
            return json.loads(completed.stdout)
        except ValueError:
            tail = (completed.stderr or "").strip().splitlines()[-1:] or [""]
            return {
                "error": f"worker exited {completed.returncode}: {tail[0]}",
                "reason": "local_error",
            }

    return run


class NobodyWhoProvider:
    name = NAME

    def __init__(
        self,
        model_path: str | None,
        samples: int = 3,
        temperature: float = 0.7,
        seed: int = 1234,
        n_ctx: int = 2048,
        use_gpu: bool = False,
        timeout_s: float = 60,
        model_info: dict[str, Any] | None = None,
        runner: Runner | None = None,
        python: str | None = None,
    ) -> None:
        if not 1 <= int(samples) <= 15:
            raise ValueError("samples must be between 1 and 15")
        self.model_path = str(Path(model_path).expanduser()) if model_path else None
        self.samples = int(samples)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.n_ctx = int(n_ctx)
        self.use_gpu = bool(use_gpu)
        self.timeout_s = float(timeout_s)
        self._model_info = model_info
        self.runner = runner or subprocess_runner(python or sys.executable)

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> NobodyWhoProvider:
        local = config["local"]
        path = local.get("model_path")
        info = local.get("model_info")
        # Identity recorded by `decision local pull`; ignored if the path was changed since.
        if isinstance(info, dict) and path and info.get("file") == Path(path).name:
            info = dict(info, sha256=local.get("model_sha256"))
        else:
            info = None
        return cls(
            model_path=path,
            samples=local.get("samples", 3),
            temperature=local.get("temperature", 0.7),
            seed=local.get("seed", 1234),
            n_ctx=local.get("n_ctx", 2048),
            use_gpu=local.get("use_gpu", False),
            timeout_s=local.get("timeout_s", 60),
            model_info=info,
            python=local.get("python"),
            **kwargs,
        )

    def model_info(self) -> dict[str, Any]:
        if self._model_info is None:
            self._model_info = describe(self.model_path) if self.model_path else {}
        return self._model_info

    def available(self) -> tuple[bool, str]:
        if not self.model_path:
            return False, "no local model configured (run: decision local pull)"
        if not os.path.isfile(self.model_path):
            return False, "local model file not found"
        return True, "ok"

    def decide(self, request: DecisionRequest) -> DecisionResult:
        started = time.monotonic()

        def elapsed() -> int:
            return round((time.monotonic() - started) * 1000)

        ok, why = self.available()
        if not ok:
            return DecisionResult.failure(NAME, why, "local_model_unavailable")

        info = self.model_info()
        details: dict[str, Any] = {
            "model_file": info.get("file"),
            "model_name": info.get("name"),
            "quantization": info.get("quantization"),
            "temperature": self.temperature,
            "seeds": [self.seed + i for i in range(self.samples)],
        }
        if info.get("sha256"):
            details["model_sha256"] = info["sha256"]
        model = Path(self.model_path or "").stem or None

        job = {
            "model_path": self.model_path,
            "system_prompt": SYSTEM_PROMPT,
            "grammar": grammar_for(request.allowed()),
            "samples": plan_samples(request, self.samples, self.seed),
            "temperature": self.temperature,
            "n_ctx": self.n_ctx,
            "use_gpu": self.use_gpu,
        }
        try:
            output = self.runner(job, self.timeout_s)
        except subprocess.TimeoutExpired:
            return DecisionResult.failure(
                NAME, "timeout", "local_timeout", latency_ms=elapsed(),
                model=model, details=details,
            )  # fmt: skip
        except Exception as error:  # noqa: BLE001 - reported as data
            return DecisionResult.failure(
                NAME, short_error(error), "local_error", latency_ms=elapsed(),
                model=model, details=details,
            )  # fmt: skip

        latency = elapsed()
        if not isinstance(output, dict):
            output = {"error": "no output", "reason": "local_error"}
        if output.get("error"):
            return DecisionResult.failure(
                NAME, short_error(output["error"]), output.get("reason", "local_error"),
                latency_ms=latency, model=model, details=details,
            )  # fmt: skip
        for key in ("runtime", "load_ms", "sample_ms"):
            if key in output:
                details[key] = output[key]
        return aggregate(request, output.get("outputs"), latency, model, details)


def aggregate(
    request: DecisionRequest,
    outputs: object,
    latency_ms: int,
    model: str | None,
    details: dict[str, Any],
) -> DecisionResult:
    """Turns constrained samples into a vote and a sample-stability proxy."""

    def fail(error: str, reason: str) -> DecisionResult:
        return DecisionResult.failure(
            NAME, error, reason, latency_ms=latency_ms, model=model, details=details
        )

    if not isinstance(outputs, list) or not outputs:
        return fail("local worker returned no samples", "local_malformed_response")
    allowed = set(request.allowed())
    if not all(isinstance(o, str) and o in allowed for o in outputs):
        return fail("sample outside the allowed set", "local_malformed_response")

    ranked = Counter(outputs).most_common()
    total = len(outputs)
    winner, count = ranked[0]
    tie = len(ranked) > 1 and ranked[1][1] == count
    abstain = not tie and winner == ABSTAIN
    return DecisionResult(
        provider=NAME,
        choice=None if tie or abstain else winner,
        abstain=abstain,
        model=model,
        latency_ms=latency_ms,
        confidence=count / total,
        confidence_kind=SAMPLE_STABILITY,
        votes=dict(ranked),
        samples=total,
        fallback_reason="local_no_majority" if tie else None,
        details=details,
    )
