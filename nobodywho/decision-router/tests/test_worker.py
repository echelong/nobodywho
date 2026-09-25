from __future__ import annotations

import errno
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from helpers import FAKE_KEY, make_request

from decision_router import config as cfg
from decision_router.providers import nobodywho as local_mod
from decision_router.providers.nobodywho import NobodyWhoProvider, PersistentWorker

MISSING_MODEL_JOB = {"model_path": "/nonexistent/model.gguf", "samples": []}
MODEL = Path(
    os.environ.get(
        "DECISION_ROUTER_TEST_MODEL",
        Path.home()
        / ".cache/nobodywho/models/NobodyWho/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_K_M.gguf",
    )
)


@pytest.fixture
def worker() -> PersistentWorker:
    return PersistentWorker(sys.executable, cfg.runtime_dir(), idle_timeout_s=30)


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_worker_starts_once_and_is_reused(worker):
    assert worker.pid() is None
    first = worker(MISSING_MODEL_JOB, 10)
    pid = worker.pid()
    assert first["reason"] == "local_model_unavailable"
    assert pid is not None
    assert worker(MISSING_MODEL_JOB, 10)["reason"] == "local_model_unavailable"
    assert worker.pid() == pid


def test_worker_is_private_to_the_user(worker):
    worker(MISSING_MODEL_JOB, 10)
    assert stat.S_IMODE(worker.runtime_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(worker.socket_path.stat().st_mode) & 0o077 == 0
    assert stat.S_ISSOCK(worker.socket_path.stat().st_mode)


def test_denied_worker_socket_does_not_restart_or_remove_worker(worker, monkeypatch):
    def denied():
        raise PermissionError(errno.EPERM, "sandbox blocked Unix socket")

    def forbidden(_deadline):
        raise AssertionError("must not stop an existing worker on access denial")

    monkeypatch.setattr(worker, "_connect", denied)
    monkeypatch.setattr(worker, "_start", forbidden)
    result = worker(MISSING_MODEL_JOB, 1)
    assert result["reason"] == "local_error"
    assert "access denied" in result["error"]


def test_worker_env_has_guard_and_no_key(worker, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GITHUB_TOKEN", "sentinel-github-token")
    monkeypatch.setenv("DB_PASSWORD", "sentinel-db-password")
    worker(MISSING_MODEL_JOB, 10)
    environ = Path(f"/proc/{worker.pid()}/environ").read_bytes().split(b"\0")
    assert b"DECISION_ROUTER_ACTIVE=1" in environ
    joined = b"\0".join(environ)
    assert b"TYPESAFE_API_KEY" not in joined and FAKE_KEY.encode() not in joined
    assert b"GITHUB_TOKEN" not in joined and b"sentinel-github-token" not in joined
    assert b"DB_PASSWORD" not in joined and b"sentinel-db-password" not in joined


def test_dead_worker_is_restarted(worker):
    worker(MISSING_MODEL_JOB, 10)
    old = worker.pid()
    assert old is not None
    os.kill(old, signal.SIGKILL)
    assert wait_until(lambda: not Path(f"/proc/{old}").exists() or worker.pid() is None)
    assert worker(MISSING_MODEL_JOB, 10)["reason"] == "local_model_unavailable"
    assert worker.pid() not in (None, old)


def test_stale_socket_is_replaced(worker):
    worker.runtime_dir.mkdir(parents=True, exist_ok=True)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(worker.socket_path))
    stale.close()  # leaves a socket file with no listener
    assert worker(MISSING_MODEL_JOB, 10)["reason"] == "local_model_unavailable"
    assert worker.pid() is not None


def test_worker_exits_when_idle():
    worker = PersistentWorker(sys.executable, cfg.runtime_dir(), idle_timeout_s=0.3)
    worker(MISSING_MODEL_JOB, 10)
    pid = worker.pid()
    assert pid is not None
    assert wait_until(lambda: not worker.socket_path.exists(), timeout=10)
    assert wait_until(lambda: worker.pid() is None, timeout=10)


def test_stuck_worker_times_out_and_is_killed(tmp_path, monkeypatch):
    fake = tmp_path / "local_worker.py"
    fake.write_text(
        "import os, socket, sys, time\n"
        "path = sys.argv[sys.argv.index('--serve') + 1]\n"
        "s = socket.socket(socket.AF_UNIX); s.bind(path); s.listen(1)\n"
        "open(path[:-5] + '.pid', 'w').write(str(os.getpid()))\n"
        "conn, _ = s.accept()\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(local_mod, "WORKER", fake)
    worker = PersistentWorker(sys.executable, cfg.runtime_dir())
    with pytest.raises(subprocess.TimeoutExpired):
        worker(MISSING_MODEL_JOB, 1.0)
    assert worker.pid() is None
    assert not worker.socket_path.exists()


def test_worker_that_cannot_start_fails_safe():
    worker = PersistentWorker("/bin/false", cfg.runtime_dir(), start_timeout_s=0.5)
    started = time.monotonic()
    output = worker(MISSING_MODEL_JOB, 10)
    assert output["reason"] == "local_error"
    assert time.monotonic() - started < 5


def test_stop_only_kills_our_worker(worker):
    worker.runtime_dir.mkdir(parents=True, exist_ok=True)
    worker.pid_path.write_text(str(os.getpid()))  # a pid that is not a worker
    assert worker.pid() is None
    assert worker.stop() is False


def test_config_selects_runner():
    config = cfg.load()
    config["local"]["model_path"] = "/nonexistent/model.gguf"
    assert isinstance(NobodyWhoProvider.from_config(config).runner, PersistentWorker)
    config["local"]["persistent"] = False
    assert not isinstance(NobodyWhoProvider.from_config(config).runner, PersistentWorker)


@pytest.mark.model
@pytest.mark.skipif(not MODEL.is_file(), reason="local GGUF model not present")
def test_real_model_is_loaded_once():
    pytest.importorskip("nobodywho")
    config = cfg.load()
    config["local"].update(model_path=str(MODEL), samples=2)
    provider = NobodyWhoProvider.from_config(config)
    first = provider.decide(make_request())
    second = provider.decide(make_request())
    assert first.error is None and second.error is None, (first, second)
    assert first.details["worker"] == "persistent"
    assert first.details["model_reused"] is False
    assert second.details["model_reused"] is True
    assert second.details["load_ms"] < first.details["load_ms"]
    assert first.votes and second.votes


def _tier_config(tmp_path, tier1_gpu: bool):
    dummy = tmp_path / "dummy.gguf"
    dummy.write_bytes(b"GGUF")
    config = cfg.load()
    config["tiers"]["1"].update(model_path="/nonexistent/t1.gguf", use_gpu=tier1_gpu)
    config["tiers"]["2"].update(model_path=str(dummy), timeout_s=20)
    return config


def test_tier2_cold_start_evicts_gpu_tier1(tmp_path):
    config = _tier_config(tmp_path, tier1_gpu=True)
    tier1 = NobodyWhoProvider.for_tier(config, "1").runner
    assert isinstance(tier1, PersistentWorker)
    tier1(MISSING_MODEL_JOB, 10)  # tier 1 worker is now resident
    assert tier1.pid() is not None
    tier2 = NobodyWhoProvider.for_tier(config, "2")
    tier2.decide(make_request())  # dummy model fails to load; eviction happens first
    assert tier1.pid() is None


def test_warm_tier2_or_cpu_tier1_is_not_evicted(tmp_path):
    config = _tier_config(tmp_path, tier1_gpu=False)
    tier1 = NobodyWhoProvider.for_tier(config, "1").runner
    assert isinstance(tier1, PersistentWorker)
    tier1(MISSING_MODEL_JOB, 10)
    assert NobodyWhoProvider.for_tier(config, "2").evict == []
    config = _tier_config(tmp_path, tier1_gpu=True)
    tier2 = NobodyWhoProvider.for_tier(config, "2")
    runner = tier2.runner
    assert isinstance(runner, PersistentWorker)
    runner(MISSING_MODEL_JOB, 10)  # tier 2 already warm
    tier2._make_room()
    assert tier1.pid() is not None
