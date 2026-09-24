"""Subprocess that runs grammar-constrained NobodyWho samples.

Runs out of process so the router can enforce a hard timeout and survive a
native crash. Input (stdin, JSON): model_path, system_prompt, grammar,
samples [{seed, prompt}], temperature, n_ctx, use_gpu. Output (stdout, JSON):
{"outputs": [...], "sample_ms": [...], "load_ms": N, "runtime": "..."}.
It never downloads: `model_path` must be an existing local file.
"""

from __future__ import annotations

import json
import os
import sys
import time


def _constrained_sampler(nobodywho, grammar: str, temperature: float, seed: int):
    builder = nobodywho.SamplerBuilder()
    if hasattr(builder, "constrain_with_grammar"):  # nobodywho main
        builder = builder.constrain_with_grammar(grammar)
    else:  # nobodywho 3.0.x
        builder = builder.grammar(grammar, None, "root")
    return builder.temperature(temperature).seed(seed).dist()


def run(job: dict) -> dict:
    model_path = job["model_path"]
    if not os.path.isfile(model_path):
        return {"error": "local model file not found", "reason": "local_model_unavailable"}
    try:
        import nobodywho
    except ImportError:
        return {"error": "nobodywho is not installed", "reason": "local_runtime_unavailable"}

    started = time.monotonic()
    model = nobodywho.Model(model_path, use_gpu_if_available=bool(job.get("use_gpu")))
    chat = nobodywho.Chat(
        model,
        n_ctx=int(job.get("n_ctx", 2048)),
        system_prompt=job["system_prompt"],
        template_variables={"enable_thinking": False},
    )
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

    version = getattr(nobodywho, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version

            version = dist_version("nobodywho")
        except Exception:  # noqa: BLE001 - version is informational
            version = "unknown"
    return {
        "outputs": outputs,
        "sample_ms": sample_ms,
        "load_ms": load_ms,
        "runtime": f"nobodywho {version}",
    }


def main() -> None:
    os.environ["DECISION_ROUTER_ACTIVE"] = "1"
    try:
        result = run(json.load(sys.stdin))
    except Exception as error:  # noqa: BLE001 - reported to the parent, never raised
        result = {"error": f"{type(error).__name__}: {error}"[:300], "reason": "local_error"}
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
