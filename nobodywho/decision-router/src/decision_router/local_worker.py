"""NobodyWho worker that runs grammar-constrained samples out of process.

Runs out of process so the router can enforce a hard timeout and survive a
native crash. Job (JSON): model_path, system_prompt, grammar,
samples [{seed, prompt}], temperature, n_ctx, use_gpu, and optionally:
  choices       the grammar's complete answers; generation stops as soon as
                the text so far can only become one of them
  stop_when     {min_share, min_margin}: stop drawing samples once no
                remaining sample can change the winner or its acceptance
  cpu_fallback  default true: a model that does not fit in free VRAM loads on
                CPU; false: the job fails fast instead
Result (JSON): {"outputs": [...], "sample_ms": [...], "planned": N,
"load_ms": N, "model_reused": bool, "gpu_layers": N, "runtime": "..."}.
It never downloads: `model_path` must be an existing local file.

Two ways to run it:
  python local_worker.py                      one job on stdin, result on stdout
  python local_worker.py --serve SOCKET       keep the model loaded and answer one
                                              JSON-line job per connection on a
                                              user-private Unix socket; exits after
                                              --idle-timeout seconds without jobs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import sys
import time
from collections import Counter
from pathlib import Path

MAX_JOB_BYTES = 1 << 20


class _LoadLog(logging.Handler):
    """Keeps what NobodyWho logs while loading a model: its GPU layer plan and warnings."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.gpu_layers: int | None = None
        self.warnings: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        found = re.search(r"\bgpu_layers=(\d+)", message)
        if found:
            self.gpu_layers = int(found.group(1))
        elif record.levelno >= logging.WARNING and len(self.warnings) < 5:
            self.warnings.append(message[-300:])


_LOAD_LOG = _LoadLog()
_nobodywho_log = logging.getLogger("nobodywho")
_nobodywho_log.setLevel(logging.INFO)
_nobodywho_log.addHandler(_LOAD_LOG)
_nobodywho_log.propagate = False


# What a fully offloaded model needs beyond its file: KV cache and compute buffers.
GPU_HEADROOM_MIB = 600


def free_vram_mib() -> int | None:
    """Free memory of the emptiest NVIDIA GPU, or None when it cannot be read."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout  # fmt: skip
        return max(int(line) for line in out.split())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def gpu_fits(model_path: str, free_mib: int | None) -> bool:
    """False when the model cannot be fully offloaded.

    NobodyWho then offloads as many layers as fill free VRAM and reserves nothing
    for the context, so the load or the first context fails with an out-of-memory
    error instead of running partially offloaded.
    """
    if free_mib is None:
        return True  # unknown: let NobodyWho decide
    need = os.path.getsize(model_path) / (1 << 20) + GPU_HEADROOM_MIB
    return need <= free_mib


def gpu_fits_soon(model_path: str, wait_s: float = 2.0) -> bool:
    """`gpu_fits`, re-checked briefly: an evicted worker's VRAM is released after it exits."""
    deadline = time.monotonic() + wait_s
    while not gpu_fits(model_path, free_vram_mib()):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)
    return True


def settled(outputs: list[str], planned: int, min_share: float, min_margin: int) -> bool:
    """True when no remaining sample can change the winner or whether it is accepted.

    Worst case: every remaining sample goes to the runner-up, and the winner's
    share is counted against all planned samples.
    """
    if len(outputs) >= planned:
        return True
    counts = sorted(Counter(outputs).values(), reverse=True)
    if not counts:
        return False
    winner, runner_up = counts[0], counts[1] if len(counts) > 1 else 0
    remaining = planned - len(outputs)
    return winner - runner_up - remaining >= max(1, min_margin) and winner / planned >= min_share


def _generate(chat, prompt: str, choices: list[str] | None) -> str:
    """One completion; with `choices`, stops once the prefix can only become one of them."""
    stream = chat.ask(prompt)
    if not choices:
        return stream.completed().strip()
    text = ""
    while (token := stream.next_token()) is not None:
        text += token
        head = text.strip()
        matches = [choice for choice in choices if choice.startswith(head)] if head else []
        if len(matches) == 1:
            # The grammar admits only the rest of this choice. Not draining the stream
            # saves a decode step; the chat runs the next command after it stops.
            chat.stop_generation()
            return matches[0]
    return text.strip()


def _constrained_sampler(nobodywho, grammar: str, temperature: float, seed: int):
    builder = nobodywho.SamplerBuilder()
    if hasattr(builder, "constrain_with_grammar"):  # nobodywho main
        builder = builder.constrain_with_grammar(grammar)
    else:  # nobodywho 3.0.x
        builder = builder.grammar(grammar, None, "root")
    return builder.temperature(temperature).seed(seed).dist()


def _runtime(nobodywho) -> str:
    version = getattr(nobodywho, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version

            version = dist_version("nobodywho")
        except Exception:  # noqa: BLE001 - version is informational
            version = "unknown"
    return f"nobodywho {version}"


def run(job: dict, cache: dict | None = None) -> dict:
    """Runs one job. With a `cache` dict, the loaded model is kept for the next job."""
    model_path = job["model_path"]
    if not os.path.isfile(model_path):
        return {"error": "local model file not found", "reason": "local_model_unavailable"}
    try:
        import nobodywho
    except ImportError:
        return {"error": "nobodywho is not installed", "reason": "local_runtime_unavailable"}

    cache = {} if cache is None else cache
    model_key = (model_path, bool(job.get("use_gpu")))
    n_ctx = int(job.get("n_ctx", 2048))
    started = time.monotonic()
    reused = cache.get("model_key") == model_key
    if not reused:
        cache.clear()
        _LOAD_LOG.gpu_layers, _LOAD_LOG.warnings = None, []
        use_gpu = model_key[1]
        notes = []
        if use_gpu and not gpu_fits_soon(model_path):
            free = free_vram_mib()
            if not job.get("cpu_fallback", True):
                return {"error": f"model does not fit in free VRAM ({free} MiB)",
                        "reason": "local_gpu_unavailable"}  # fmt: skip
            use_gpu = False
            notes.append(f"model does not fit in free VRAM ({free} MiB): loaded on CPU")
        cache.update(
            model_key=model_key,
            model=nobodywho.Model(model_path, use_gpu_if_available=use_gpu),
            gpu_layers=_LOAD_LOG.gpu_layers if use_gpu else 0,
            load_warnings=notes + list(_LOAD_LOG.warnings),
        )
    if cache.get("n_ctx") != n_ctx:  # a new context, never a model reload
        cache.update(
            n_ctx=n_ctx,
            chat=nobodywho.Chat(
                cache["model"],
                n_ctx=n_ctx,
                system_prompt=job["system_prompt"],
                template_variables={"enable_thinking": False},
            ),
        )
    chat = cache["chat"]
    if chat.get_system_prompt() != job["system_prompt"]:
        chat.set_system_prompt(job["system_prompt"])
    load_ms = round((time.monotonic() - started) * 1000)

    choices = job.get("choices") or None
    stop_when = job.get("stop_when") or None
    planned = len(job["samples"])
    outputs, sample_ms = [], []
    for sample in job["samples"]:
        chat.reset_history()
        chat.set_sampler_config(
            _constrained_sampler(
                nobodywho, job["grammar"], float(job["temperature"]), int(sample["seed"])
            )
        )
        t = time.monotonic()
        outputs.append(_generate(chat, sample["prompt"], choices))
        sample_ms.append(round((time.monotonic() - t) * 1000))
        if stop_when and settled(
            outputs, planned, float(stop_when.get("min_share", 1.0)),
            int(stop_when.get("min_margin", 1)),
        ):  # fmt: skip
            break

    return {
        "outputs": outputs,
        "sample_ms": sample_ms,
        "planned": planned,
        "load_ms": load_ms,
        "model_reused": reused,
        "gpu_layers": cache.get("gpu_layers"),
        "load_warnings": cache.get("load_warnings") or [],
        "runtime": _runtime(nobodywho),
    }


def _safe_run(job: object, cache: dict | None = None) -> dict:
    try:
        if not isinstance(job, dict):
            return {"error": "job must be a JSON object", "reason": "local_error"}
        return run(job, cache)
    except Exception as error:  # noqa: BLE001 - reported to the caller, never raised
        if cache is not None:
            cache.clear()  # never reuse a model/chat left in an unknown state
        return {"error": f"{type(error).__name__}: {error}"[:300], "reason": "local_error"}


def serve(socket_path: Path, idle_timeout: float) -> None:
    """Answers jobs on a Unix socket until idle; the socket is private to this user."""
    pid_path = socket_path.with_suffix(".pid")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    previous_umask = os.umask(0o177)
    try:
        server.bind(str(socket_path))
    finally:
        os.umask(previous_umask)
    server.listen(8)
    server.settimeout(idle_timeout)
    pid_path.write_text(str(os.getpid()))
    cache: dict = {}
    try:
        while True:
            try:
                conn, _ = server.accept()
            except TimeoutError:
                return  # idle: free the model's memory
            with conn:
                conn.settimeout(30)
                try:
                    line = conn.makefile("rb").readline(MAX_JOB_BYTES)
                    result = _safe_run(json.loads(line), cache)
                except (OSError, ValueError) as error:
                    result = {"error": f"bad request: {error}"[:200], "reason": "local_error"}
                try:
                    conn.sendall((json.dumps(result) + "\n").encode())
                except OSError:
                    pass  # the client gave up (timeout); keep serving
    finally:
        server.close()
        for path in (socket_path, pid_path):
            try:
                if path == pid_path and path.read_text().strip() != str(os.getpid()):
                    continue
                path.unlink()
            except OSError:
                pass


def main() -> None:
    os.environ["DECISION_ROUTER_ACTIVE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", type=Path)
    parser.add_argument("--idle-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve, args.idle_timeout)
        return
    try:
        job = json.load(sys.stdin)
    except ValueError as error:
        job = None
        result = {"error": f"bad request: {error}"[:200], "reason": "local_error"}
    if job is not None:
        result = _safe_run(job)
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
