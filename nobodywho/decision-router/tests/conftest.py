from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from helpers import FAKE_KEY

from decision_router.ledger import Ledger
from decision_router.providers.nobodywho import PersistentWorker


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own config/state dirs and no real credentials."""
    monkeypatch.setenv("DECISION_ROUTER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("DECISION_ROUTER_STATE_DIR", str(tmp_path / "state"))
    for var in (
        "DECISION_ROUTER_MODE",
        "DECISION_ROUTER_BYPASS",
        "DECISION_ROUTER_ACTIVE",
        "TYPESAFE_API_KEY",
        "JEV_BASE_URL",
        "JEV_MODEL",
        "JEV_KEY_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    # Unix socket paths are length-limited, so the worker gets a short private dir.
    runtime = Path(tempfile.mkdtemp(prefix="drt-"))
    monkeypatch.setenv("DECISION_ROUTER_RUNTIME_DIR", str(runtime))
    yield
    PersistentWorker(sys.executable, runtime).stop()
    shutil.rmtree(runtime, ignore_errors=True)


@pytest.fixture
def key_file(tmp_path) -> Path:
    path = tmp_path / "typesafe.key"
    path.write_text(FAKE_KEY + "\n")
    path.chmod(0o600)
    return path


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "state" / "ledger.jsonl", (FAKE_KEY,))
