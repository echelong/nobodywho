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

import fcntl
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import config as cfg
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


def _worker_env() -> dict[str, str]:
    env = dict(os.environ, DECISION_ROUTER_ACTIVE="1")
    env.pop("TYPESAFE_API_KEY", None)  # the local worker never needs it
    return env


def subprocess_runner(python: str) -> Runner:
    """One fresh worker process per decision: loads the model every time."""

    def run(job: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        completed = subprocess.run(
            [python, str(WORKER)],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_worker_env(),
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


class PersistentWorker:
    """A per-user worker process that keeps the model loaded between decisions.

    It listens on a Unix socket in a 0700 runtime directory (no network port),
    is started on first use, exits by itself after `idle_timeout_s` without
    jobs, and is restarted when it is missing, stale or has died. A job that
    exceeds its timeout kills the worker, so a stuck inference never blocks
    the next decision.
    """

    def __init__(
        self,
        python: str,
        runtime_dir: Path,
        idle_timeout_s: float = 900,
        start_timeout_s: float = 15,
        name: str = "local-worker",
    ) -> None:
        self.python = python
        self.name = name
        self.runtime_dir = Path(runtime_dir)
        self.socket_path = self.runtime_dir / f"{name}.sock"
        self.pid_path = self.runtime_dir / f"{name}.pid"
        self.lock_path = self.runtime_dir / f"{name}.lock"
        self.idle_timeout_s = float(idle_timeout_s)
        self.start_timeout_s = float(start_timeout_s)

    def _connect(self) -> socket.socket:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.connect(str(self.socket_path))
        except OSError:
            conn.close()
            raise
        return conn

    def pid(self) -> int | None:
        """The live worker's pid, if the pid file points at our worker."""
        try:
            pid = int(self.pid_path.read_text().strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            return None
        # Any install's worker counts, so `stop` still works after a reinstall moved WORKER.
        if b"--serve" in cmdline and any(arg.endswith(b"local_worker.py") for arg in cmdline):
            return pid
        return None

    def stop(self) -> bool:
        pid = self.pid()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pid = None
        for path in (self.socket_path, self.pid_path):
            try:
                path.unlink()
            except OSError:
                pass
        return pid is not None

    def _start(self, deadline: float) -> socket.socket:
        self.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.runtime_dir, 0o700)
        with open(self.lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # one starter at a time
            try:
                return self._connect()  # another caller started it meanwhile
            except OSError:
                self.stop()  # clear a stale socket or a hung worker
            subprocess.Popen(
                [
                    self.python, str(WORKER), "--serve", str(self.socket_path),
                    "--idle-timeout", str(self.idle_timeout_s),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_worker_env(),
                start_new_session=True,
                close_fds=True,
            )  # fmt: skip
            limit = min(deadline, time.monotonic() + self.start_timeout_s)
            while time.monotonic() < limit:
                try:
                    return self._connect()
                except OSError:
                    time.sleep(0.02)
        raise RuntimeError("local worker did not start")

    def __call__(self, job: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        payload = (json.dumps(job) + "\n").encode()
        error: Exception | None = None
        for _ in range(2):  # one restart if the worker vanished or died mid-job
            try:
                try:
                    conn = self._connect()
                except OSError:
                    conn = self._start(deadline)
                with conn:
                    conn.settimeout(max(0.01, deadline - time.monotonic()))
                    conn.sendall(payload)
                    line = conn.makefile("rb").readline()
                if line:
                    return json.loads(line)
                error = ConnectionError("worker closed the connection")
            except TimeoutError:
                self.stop()
                raise subprocess.TimeoutExpired("local-worker", timeout_s) from None
            except (OSError, ValueError, RuntimeError) as exc:
                error = exc
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("local-worker", timeout_s)
        return {"error": f"local worker failed: {error}", "reason": "local_error"}


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
        """The provider for `local` mode (and shadow/compare)."""
        return cls.from_settings(config["local"], **kwargs)

    @classmethod
    def for_tier(cls, config: dict[str, Any], tier: str, **kwargs: Any) -> NobodyWhoProvider:
        """The provider for one local-first tier, with its own worker."""
        return cls.from_settings(
            cfg.tier_settings(config, tier), worker_name=f"tier{tier}-worker", **kwargs
        )

    @classmethod
    def from_settings(
        cls, local: dict[str, Any], worker_name: str = "local-worker", **kwargs: Any
    ) -> NobodyWhoProvider:
        path = local.get("model_path")
        info = local.get("model_info")
        # Identity recorded by `decision local pull`; ignored if the path was changed since.
        if isinstance(info, dict) and path and info.get("file") == Path(path).name:
            info = dict(info, sha256=local.get("model_sha256"))
        else:
            info = None
        python = local.get("python") or sys.executable
        if "runner" not in kwargs and local.get("persistent"):
            kwargs["runner"] = PersistentWorker(
                python, cfg.runtime_dir(), local.get("idle_timeout_s", 900), name=worker_name
            )
        return cls(
            model_path=path,
            samples=local.get("samples", 3),
            temperature=local.get("temperature", 0.7),
            seed=local.get("seed", 1234),
            n_ctx=local.get("n_ctx", 2048),
            use_gpu=local.get("use_gpu", False),
            timeout_s=local.get("timeout_s", 60),
            model_info=info,
            python=python,
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

    def decide(self, request: DecisionRequest, attempt: int = 0) -> DecisionResult:
        """`attempt` > 0 draws a fresh, still deterministic, set of seeds."""
        started = time.monotonic()
        seed = self.seed + attempt * self.samples

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
            "seeds": [seed + i for i in range(self.samples)],
            "worker": "persistent" if isinstance(self.runner, PersistentWorker) else "oneshot",
        }
        if info.get("sha256"):
            details["model_sha256"] = info["sha256"]
        model = Path(self.model_path or "").stem or None

        job = {
            "model_path": self.model_path,
            "system_prompt": SYSTEM_PROMPT,
            "grammar": grammar_for(request.allowed()),
            "samples": plan_samples(request, self.samples, seed),
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
        for key in ("runtime", "load_ms", "sample_ms", "model_reused"):
            if key in output:
                details[key] = output[key]
        return aggregate(request, output.get("outputs"), latency, model, details)

    def run_prompts(
        self,
        system_prompt: str,
        grammar: str,
        prompts: list[str],
        temperature: float,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Runs one grammar-constrained completion per prompt on this model's worker.

        Returns the worker output ({"outputs": [...], ...} or {"error", "reason"});
        raises subprocess.TimeoutExpired on timeout.
        """
        ok, why = self.available()
        if not ok:
            return {"error": why, "reason": "local_model_unavailable"}
        job = {
            "model_path": self.model_path,
            "system_prompt": system_prompt,
            "grammar": grammar,
            "samples": [{"seed": self.seed + i, "prompt": p} for i, p in enumerate(prompts)],
            "temperature": temperature,
            "n_ctx": self.n_ctx,
            "use_gpu": self.use_gpu,
        }
        return self.runner(job, self.timeout_s if timeout_s is None else timeout_s)

    @property
    def model_name(self) -> str | None:
        return Path(self.model_path).stem if self.model_path else None


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
