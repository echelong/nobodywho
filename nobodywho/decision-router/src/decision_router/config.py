"""Router configuration and XDG locations.

Config: $XDG_CONFIG_HOME/decision-router/config.json
Ledger: $XDG_STATE_HOME/decision-router/ledger.jsonl
Worker: $XDG_RUNTIME_DIR/decision-router/local-worker.sock (persistent local model)

The TypeSafe key is never stored here: it is read at call time from the
existing JEV key file (or TYPESAFE_API_KEY), exactly like the JEV install does.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

MODES = ("off", "jev", "local", "shadow", "compare", "local-first")
DEFAULT_MODE = "local-first"

DEFAULT_LOCAL_MODEL = "huggingface:NobodyWho/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_K_M.gguf"

DEFAULTS: dict[str, Any] = {
    "mode": DEFAULT_MODE,
    "jev": {
        "endpoint": "https://api.typesafe.ai/v1/systemone",
        "model": "jev-latest",
        "key_file": "~/.config/jev/typesafe.key",
        "timeout_s": 20,
        # Hard switch: when false the router never calls JEV, in any mode.
        "enabled": False,
    },
    "local": {
        "source": DEFAULT_LOCAL_MODEL,
        "model_path": None,
        "model_sha256": None,
        "samples": 3,
        "temperature": 0.7,
        "seed": 1234,
        "n_ctx": 2048,
        "use_gpu": False,
        "timeout_s": 60,
        "python": None,
        "persistent": True,
        "idle_timeout_s": 900,
    },
    # local-first: each tier inherits every "local" setting it does not override.
    "tiers": {
        # evict_tiers: GPU workers to stop before this tier cold-starts. An 8 GB card holds
        # one of tier 1 (4B, ~3 GB) and tier 2 (9B, ~5.5 GB), not both, and NobodyWho fills
        # whatever VRAM is left with layers without reserving room for the context.
        "1": {"label": "tier1", "evict_tiers": ["2"]},
        "2": {"label": "tier2", "idle_timeout_s": 120, "timeout_s": 300, "evict_tiers": ["1"]},
    },
    # When a local tier's answer is good enough to stop escalating.
    "acceptance": {
        "min_stability": 0.66,  # winner's vote share (sample_stability, not a probability)
        "min_margin": 1,  # winner votes minus runner-up votes
        "escalate_on_abstain": True,
        "max_local_retries": 0,  # extra attempts per tier with fresh seeds
    },
    # `decision prune`: extractive output pruning, local-first like `decision ask`.
    "prune": {
        "budget_chars": 12000,
        "min_output_chars": 16000,
        "block_lines": 30,
        "hard_cap_factor": 2.0,
        "archive": False,  # output can contain secrets; the shared ledger stores sizes only
        # Blocks a local tier judges in its one generation; 8 short excerpts fit n_ctx 2048.
        "tier1": {"max_blocks": 8, "timeout_s": 45},
        "tier2": {"enabled": True, "max_blocks": 8, "timeout_s": 150},
        "jev_max_blocks": 60,
    },
}

TIER_NAMES = ("1", "2")
MODEL_IDENTITY = ("source", "model_path", "model_sha256", "model_info")


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var, "").strip()
    return Path(value) if value else Path.home() / fallback


def config_dir() -> Path:
    override = os.environ.get("DECISION_ROUTER_CONFIG_DIR", "").strip()
    if override:
        return Path(override)
    return _xdg("XDG_CONFIG_HOME", ".config") / "decision-router"


def state_dir() -> Path:
    override = os.environ.get("DECISION_ROUTER_STATE_DIR", "").strip()
    if override:
        return Path(override)
    return _xdg("XDG_STATE_HOME", ".local/state") / "decision-router"


def runtime_dir() -> Path:
    """Where the persistent local worker keeps its socket: private to this user's session."""
    override = os.environ.get("DECISION_ROUTER_RUNTIME_DIR", "").strip()
    if override:
        return Path(override)
    runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime:
        return Path(runtime) / "decision-router"
    # CLI clients may omit XDG_RUNTIME_DIR even when they share the same login
    # session. Reuse its runtime directory so they cannot keep duplicate GPU
    # workers resident under two socket paths.
    user_runtime = Path(f"/run/user/{os.getuid()}")
    try:
        if user_runtime.stat().st_uid == os.getuid() and os.access(user_runtime, os.W_OK):
            return user_runtime / "decision-router"
    except OSError:
        pass
    return state_dir() / "run"


def data_dir() -> Path:
    override = os.environ.get("DECISION_ROUTER_DATA_DIR", "").strip()
    if override:
        return Path(override)
    return _xdg("XDG_DATA_HOME", ".local/share") / "decision-router"


def prune_archive_dir() -> Path:
    return state_dir() / "prune-archive"


def config_path() -> Path:
    return config_dir() / "config.json"


def ledger_path() -> Path:
    return state_dir() / "ledger.jsonl"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load() -> dict[str, Any]:
    """The effective config: defaults, then the config file, then env overrides."""
    stored: dict[str, Any] = {}
    try:
        stored = json.loads(config_path().read_text())
        if not isinstance(stored, dict):
            stored = {}
    except (OSError, ValueError):
        stored = {}
    config = _merge(DEFAULTS, stored)

    return config


def tier_settings(config: dict[str, Any], tier: str) -> dict[str, Any]:
    """The effective settings of one local-first tier: "local" overlaid with the tier's own.

    The model itself is never inherited: a tier without its own model is unavailable
    rather than silently running the `local` mode's model.
    """
    base = {k: v for k, v in config["local"].items() if k not in MODEL_IDENTITY}
    return _merge(base, config.get("tiers", {}).get(tier, {}))


def mode(config: dict[str, Any]) -> tuple[str, str]:
    """The active mode and where it came from."""
    env = os.environ.get("DECISION_ROUTER_MODE", "").strip().lower()
    if env:
        return env, "env:DECISION_ROUTER_MODE"
    stored = str(config.get("mode", DEFAULT_MODE)).lower()
    return stored, str(config_path()) if config_path().exists() else "default"


def update(changes: dict[str, Any]) -> dict[str, Any]:
    """Merges `changes` into the stored config file (never into defaults)."""
    path = config_path()
    try:
        stored = json.loads(path.read_text())
        if not isinstance(stored, dict):
            stored = {}
    except (OSError, ValueError):
        stored = {}
    stored = _merge(stored, changes)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stored, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    return stored
