from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from decision_router import config as cfg

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "adapters/codex/pre_tool_use.sh"
CLINE = ROOT / "adapters/cline/decision-prune.ts"
SHELL_RUNNER = ROOT / "adapters/shared/decision-run-shell.bash"


def _codex_hook(payload: dict) -> dict:
    result = subprocess.run(
        ["/bin/bash", str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "command",
    [
        "printf simple",
        "printf a | tr a b",
        "true && printf yes || printf no",
        "printf 'quoted string'",
        "printf first\nprintf second",
        "printf output > result.txt; cat result.txt",
        "printf err >&2; printf out; exit 7",
    ],
)
def test_codex_hook_preserves_exact_shell_command(command, tmp_path):
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"cmd": command, "yield_time_ms": 1000},
    }
    assert _codex_hook(payload) == {}


def test_codex_runner_preserves_streams_exit_and_redirection(tmp_path):
    command = "printf 'out'; printf 'err' >&2; printf 'file' > saved.txt; exit 9"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "decision_router",
            "run",
            "--caller",
            "codex",
            "--",
            "/bin/bash",
            "-lc",
            command,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode, result.stdout, result.stderr) == (9, "out", "err")
    assert (tmp_path / "saved.txt").read_text() == "file"


def test_decision_run_broken_config_uses_native_command(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text("{broken")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "decision_router",
            "run",
            "--caller",
            "codex",
            "--",
            "/bin/bash",
            "-c",
            "printf native; exit 7",
        ],
        env=dict(os.environ, DECISION_ROUTER_CONFIG_DIR=str(config_dir)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode, result.stdout) == (7, "native")


def test_codex_hook_ignores_other_tools_and_nested_wrappers():
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"cmd": "echo ok"},
    }
    assert _codex_hook({**payload, "tool_name": "apply_patch"}) == {}
    assert _codex_hook({**payload, "tool_input": {"cmd": "decision_run_shell echo ok"}}) == {}


def test_codex_shell_runner_failure_returns_native_output(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_decision = fake_bin / "decision"
    fake_decision.write_text("#!/bin/sh\nexit 1\n")
    fake_decision.chmod(0o755)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"prune": {"min_output_chars": 10}}))
    command = "printf '12345678901234567890'; printf 'stderr' >&2; exit 7"
    invoke = f'. {shlex.quote(str(SHELL_RUNNER))}; decision_run_shell "$1"'
    result = subprocess.run(
        ["/bin/bash", "-c", invoke, "_", command],
        env=dict(
            os.environ,
            PATH=f"{fake_bin}:{os.environ['PATH']}",
            DECISION_ROUTER_CONFIG_DIR=str(config_dir),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        7,
        "12345678901234567890",
        "stderr",
    )


def test_shared_threshold_skips_without_receipt(tmp_path):
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    config_dir.mkdir()
    config = cfg.DEFAULTS | {
        "mode": "off",
        "jev": {"enabled": False},
        "prune": {"budget_chars": 12000, "min_output_chars": 16000},
    }
    (config_dir / "config.json").write_text(json.dumps(config))
    env = dict(
        os.environ,
        DECISION_ROUTER_CONFIG_DIR=str(config_dir),
        DECISION_ROUTER_STATE_DIR=str(state_dir),
    )
    env.pop("DECISION_PRUNE_ACTIVE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "decision_router",
            "run",
            "--caller",
            "codex",
            "--",
            "/bin/bash",
            "-lc",
            "printf short",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and result.stdout == "short"
    assert not (state_dir / "ledger.jsonl").exists()
    structured = subprocess.run(
        [sys.executable, "-m", "decision_router", "prune", "--json", "--caller", "claude"],
        input="x" * 14000,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(structured.stdout)["fallback_reason"] == "under_threshold"
    assert not (state_dir / "ledger.jsonl").exists()


def test_runtime_dir_reuses_owned_login_session(monkeypatch):
    user_runtime = Path(f"/run/user/{os.getuid()}")
    if (
        not user_runtime.exists()
        or user_runtime.stat().st_uid != os.getuid()
        or not os.access(user_runtime, os.W_OK)
    ):
        pytest.skip("no writable owned login runtime directory")
    monkeypatch.delenv("DECISION_ROUTER_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert cfg.runtime_dir() == user_runtime / "decision-router"


def test_cline_hook_threshold_transform_and_failure_fallback(tmp_path):
    home = tmp_path / "home"
    decision = home / ".local/bin/decision"
    decision.parent.mkdir(parents=True)
    decision.write_text("#!/bin/sh\ncat >/dev/null\nprintf PRUNED\n")
    decision.chmod(0o755)
    config_dir = home / ".config/decision-router"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({"prune": {"min_output_chars": 10}}))
    script = f"""
import plugin from {json.dumps(CLINE.as_uri())};
const hook = plugin.hooks.afterTool;
const context = {{tool: {{name: 'run_commands'}}, result: {{isError: true,
  output: [{{query: 'example', result: 'long result here', success: false}}]}}}};
const first = hook(context);
console.log(JSON.stringify(first));
console.log(JSON.stringify(hook({{...context, result: {{...context.result,
  output: [{{query: 'example', result: 'tiny', success: false}}]}}}})));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        env=dict(os.environ, HOME=str(home)),
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    changed = json.loads(lines[0])["result"]
    assert changed["isError"] is True
    assert changed["output"][0] == {"query": "example", "result": "PRUNED", "success": False}
    assert lines[1] == "undefined"
    decision.write_text("#!/bin/sh\nexit 1\n")
    failed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        env=dict(os.environ, HOME=str(home)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert failed.stdout.splitlines()[0] == "undefined"
    assert "typesafe" not in CLINE.read_text().lower()
    assert "typesafe" not in HOOK.read_text().lower()
