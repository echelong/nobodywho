"""NobodyWho worker that runs grammar-constrained samples out of process.

Runs out of process so the router can enforce a hard timeout and survive a
native crash. Job (JSON): model_path, system_prompt, grammar,
samples [{seed, prompt}], temperature, n_ctx, use_gpu. Result (JSON):
{"outputs": [...], "sample_ms": [...], "load_ms": N, "model_reused": bool, "runtime": "..."}.
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
import os
import socket
import sys
import time
from pathlib import Path

MAX_JOB_BYTES = 1 << 20


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
    key = (model_path, bool(job.get("use_gpu")), int(job.get("n_ctx", 2048)))
    started = time.monotonic()
    reused = cache.get("key") == key
    if not reused:
        cache.clear()
        model = nobodywho.Model(model_path, use_gpu_if_available=key[1])
        cache.update(
            key=key,
            model=model,
            chat=nobodywho.Chat(
                model,
                n_ctx=key[2],
                system_prompt=job["system_prompt"],
                template_variables={"enable_thinking": False},
            ),
        )
    chat = cache["chat"]
    if chat.get_system_prompt() != job["system_prompt"]:
        chat.set_system_prompt(job["system_prompt"])
    load_ms = round((time.monotonic() - started) * 1000)

    outputs, sample_ms = [], []
    for sample in job["samples"]:
        chat.reset_history()
        chat.set_sampler_config(
            _constrained_sampler(
                nobodywho, job["grammar"], float(job["temperature"]), int(sample["seed"])
            )
        )
        t = time.monotonic()
        outputs.append(chat.ask(sample["prompt"]).completed().strip())
        sample_ms.append(round((time.monotonic() - t) * 1000))

    return {
        "outputs": outputs,
        "sample_ms": sample_ms,
        "load_ms": load_ms,
        "model_reused": reused,
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
