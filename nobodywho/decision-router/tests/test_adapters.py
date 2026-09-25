from __future__ import annotations

import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from decision_router import config as cfg

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "adapters/codex/pre_tool_use.sh"
CLINE = ROOT / "adapters/cline/decision-prune.ts"
SHELL_RUNNER = ROOT / "adapters/shared/decision-run-shell.bash"
POLICY_INSTALLER = ROOT / "adapters/shared/install_policy.py"
CLAUDE_HOOK = ROOT / "adapters/claude/decision-prune/hooks/decision-prune.ts"
CODEX_MCP = ROOT / "adapters/codex/decision_mcp.py"


def _codex_hook(payload: dict) -> dict:
    result = subprocess.run(
        ["/bin/bash", str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_shared_policy_installs_for_six_peers(tmp_path):
    install = runpy.run_path(str(POLICY_INSTALLER))["install"]
    paths = install(tmp_path)
    assert len(paths) == 6
    for path in paths:
        text = path.read_text()
        assert "decision ask --caller" in text
        assert '"state":"brief facts"' in text
        assert "bypass decision router" in text
        assert "Do not run a second pruner" in text
        assert len(text) < 1_200
    assert "csmart" in (tmp_path / ".claude-max/CLAUDE.md").read_text()
    assert "opencode2" in (tmp_path / ".config/opencode/AGENTS.md").read_text()
    assert "decision_ask" in (tmp_path / ".codex/AGENTS.md").read_text()
    assert install(tmp_path) == paths


def test_codex_mcp_has_data_only_tools(monkeypatch):
    namespace = runpy.run_path(str(CODEX_MCP))
    handle = namespace["handle"]
    tools = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert {tool["name"] for tool in tools["result"]["tools"]} == {
        "decision_ask",
        "decision_prune_text",
    }
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '{"follow":"inspect_logs"}', "")

    monkeypatch.setattr(namespace["call_tool"].__globals__["subprocess"], "run", fake_run)
    result = namespace["call_tool"](
        "decision_ask",
        {"state": "error", "question": "Next?", "choices": {"inspect_logs": "Read logs"}},
    )
    assert result["follow"] == "inspect_logs"
    assert calls[0][0][:4] == ["decision", "ask", "--caller", "codex"]
    namespace["call_tool"]("decision_prune_text", {"output": "long output"})
    assert calls[1][0] == ["decision", "prune", "--caller", "codex", "--json"]
    assert calls[1][1]["input"] == "long output"


def test_disabled_jev_provider_is_not_constructed(monkeypatch):
    from decision_router import cli

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled TypeSafe provider was constructed")

    monkeypatch.setattr(cli.JevProvider, "from_config", forbidden)
    config = cfg.load()
    assert config["jev"]["enabled"] is False
    router = cli.build_router(config)
    pruner = cli.build_pruner(config)
    assert "jev" not in router.providers
    assert pruner.jev_enabled is False


def test_claude_hook_never_prunes_an_already_pruned_result():
    if shutil.which("bun") is None:
        pytest.skip("Bun is unavailable")
    script = f"""
import {{ register }} from {json.dumps(CLAUDE_HOOK.as_uri())};
let hook;
register((...args) => {{ hook = args[2]; }}, {{ minChars: 1000 }});
let calls = 0;
const answer = {{ result: {{ stdout: 'x'.repeat(1200) + '[decision prune: kept 2 of 200 lines]',
  stderr: '', exitCode: 0 }} }};
const context = {{ env: {{ get: async () => '/tmp' }},
  process: {{ run: async () => {{ calls++; return {{ exitCode: 0, stdout: '{{}}' }}; }} }},
  fs: {{ read: async () => '' }}, ui: {{ toast: () => {{}} }} }};
const result = await hook(context, {{ command: 'echo test' }}, async () => answer);
console.log(JSON.stringify({{ calls, same: result === answer }}));
"""
    run = subprocess.run(["bun", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == {"calls": 0, "same": True}


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
