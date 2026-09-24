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

MODES = ("off", "jev", "local", "shadow", "compare")
DEFAULT_MODE = "jev"

DEFAULT_LOCAL_MODEL = "huggingface:NobodyWho/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_K_M.gguf"

DEFAULTS: dict[str, Any] = {
    "mode": DEFAULT_MODE,
    "jev": {
        "endpoint": "https://api.typesafe.ai/v1/systemone",
        "model": "jev-latest",
        "key_file": "~/.config/jev/typesafe.key",
        "timeout_s": 20,
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
}


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
    return Path(runtime) / "decision-router" if runtime else state_dir() / "run"


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

    # The existing ds launcher exports these for JEV; honour them.
    if os.environ.get("JEV_BASE_URL", "").strip():
        config["jev"]["endpoint"] = os.environ["JEV_BASE_URL"].strip()
    if os.environ.get("JEV_MODEL", "").strip():
        config["jev"]["model"] = os.environ["JEV_MODEL"].strip()
    if os.environ.get("JEV_KEY_FILE", "").strip():
        config["jev"]["key_file"] = os.environ["JEV_KEY_FILE"].strip()
    return config


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
