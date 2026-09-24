from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import pytest
from helpers import FAKE_KEY, FakeTransport

from decision_router import cli
from decision_router import config as cfg
from decision_router import prune as prune_module
from decision_router.ledger import Ledger
from decision_router.providers.jev import JevProvider
from decision_router.prune import (
    DROP,
    EXCERPT_CHARS,
    KEEP,
    Pruner,
    PruneRequest,
    PruneResult,
    PruneTierError,
    Tier,
    is_repetitive,
    plan,
    votes_grammar,
)
from decision_router.prune_corpus import cases
from decision_router.prune_providers import jev_judge, local_judge

CASES = cases()
LONG = CASES[0][2]  # the pytest log


def judge_all(vote: str, calls: list | None = None):
    def judge(request, p, blocks):
        if calls is not None:
            calls.append(len(blocks))
        return {b: vote for b in blocks}

    return judge


def failing(reason: str, calls: list | None = None):
    def judge(request, p, blocks):
        if calls is not None:
            calls.append(len(blocks))
        raise PruneTierError(reason)

    return judge


def pruner(t1=None, t2=None, jev=None, *, jev_enabled=True, archive_dir=None) -> Pruner:
    tiers = [Tier(1, "nobodywho", "q9b", t1 or judge_all(DROP))]
    if t2 is not None:
        tiers.append(Tier(2, "nobodywho", "q27b", t2))
    return Pruner(
        tiers, Tier(3, "jev", "jev-latest", jev) if jev else None,
        jev_enabled=jev_enabled, archive_dir=archive_dir, secret_literals=(FAKE_KEY,),
    )  # fmt: skip


def body_lines(text: str) -> list[str]:
    return [
        line for line in text.splitlines()
        if not line.startswith("[... ") and not line.startswith("[decision prune:")
    ]  # fmt: skip


# ---------------------------------------------------------------- extractive guarantees


def test_short_output_is_untouched_and_no_provider_runs():
    calls: list = []
    result = pruner(judge_all(DROP, calls)).prune(PruneRequest(output="ok\n" * 10))
    assert result.text == "ok\n" * 10 and result.provider == "none" and calls == []


@pytest.mark.parametrize("name, command, output, must_keep", CASES, ids=[c[0] for c in CASES])
def test_critical_facts_survive_even_if_every_block_is_dropped(name, command, output, must_keep):
    result = pruner(judge_all(DROP)).prune(
        PruneRequest(output=output, command=command, budget_chars=4000)
    )
    missing = [m for m in must_keep if m not in result.text]
    assert not missing, (name, missing)
    assert result.result_chars <= result.original_chars


@pytest.mark.parametrize("name, command, output, must_keep", CASES, ids=[c[0] for c in CASES])
def test_output_is_extractive_and_ordered(name, command, output, must_keep):
    result = pruner(judge_all(KEEP)).prune(PruneRequest(output=output, budget_chars=3000))
    original = output.splitlines()
    position = 0
    for line in body_lines(result.text):
        if "characters of critical output truncated" in line:
            continue
        # Every kept line is copied verbatim, in the original order (hard-cap cuts may split one).
        found = next((i for i in range(position, len(original)) if line in original[i]), None)
        assert found is not None, (name, line[:80])
        position = found


def test_budget_is_respected_when_possible():
    result = pruner(judge_all(KEEP)).prune(PruneRequest(output=CASES[1][2], budget_chars=3000))
    assert result.result_chars <= 3000 + 200  # footer


def test_hard_cap_bounds_all_critical_output():
    search = next(c for c in CASES if c[0] == "search")[2]
    result = pruner().prune(PruneRequest(output=search, budget_chars=5000))
    assert result.result_chars <= 2 * 5000 + 300


def test_never_longer_than_the_input():
    text = "\n".join(f"error line {i}" for i in range(2000))  # everything is critical
    result = pruner().prune(PruneRequest(output=text, budget_chars=10**6 // 10))
    assert result.result_chars <= len(text)


def test_preservation_hints_are_kept():
    lines = [f"noise {i} {'x' * 30}" for i in range(800)]
    lines[400] = "the MAGIC-TOKEN-42 lives here"
    result = pruner(judge_all(DROP)).prune(
        PruneRequest(output="\n".join(lines), preserve=("magic-token",), budget_chars=2000)
    )
    assert "the MAGIC-TOKEN-42 lives here" in result.text


def test_repetitive_runs_are_dropped_without_a_model_and_merged():
    npm = next(c for c in CASES if c[0] == "npm-install")[2]
    calls: list = []
    result = pruner(judge_all(KEEP, calls)).prune(PruneRequest(output=npm))
    assert sum(calls) == 0  # all noise was template-identical
    assert result.text.count("omitted by") <= 4
    assert is_repetitive([f"[{i}/9] step" for i in range(6)])
    assert not is_repetitive(["alpha", "beta", "gamma", "delta", "eps"])


def test_plan_keeps_head_tail_and_context():
    lines = [f"row {i} {'y' * 20}" for i in range(200)]
    lines[100] = "ERROR: boom"
    p = plan(PruneRequest(output="\n".join(lines)))
    assert {0, 4, 199, 185, 98, 99, 100, 101, 102} <= p.critical
    assert 50 not in p.critical


# ---------------------------------------------------------------- local-first tiers


def test_tier1_success_keeps_tier2_and_jev_dormant():
    t2: list = []
    jev: list = []
    p = pruner(judge_all(DROP), judge_all(KEEP, t2), judge_all(KEEP, jev))
    result = p.prune(PruneRequest(output=LONG, budget_chars=3000))
    assert result.tier == 1 and result.provider == "nobodywho" and not result.jev_used
    assert not result.native_fallback and 0 <= result.compression_ratio <= 1
    assert t2 == [] and jev == [] and p.jev_calls == 0


def test_quality_failure_rejects_a_local_result_and_escalates(monkeypatch):
    original_fit = prune_module.fit
    calls = 0

    def lose_critical_once(plan, votes, budget, label):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "[bad local result]", set()
        return original_fit(plan, votes, budget, label)

    monkeypatch.setattr(prune_module, "fit", lose_critical_once)
    result = pruner(judge_all(DROP), judge_all(DROP)).prune(
        PruneRequest(output=LONG, budget_chars=3000)
    )
    assert result.provider == "nobodywho" and result.tier == 2
    assert result.fallback_reason == "pruning_quality_failure"
    assert "assert 500 == 200" in result.text and not result.native_fallback


def test_native_result_marks_both_local_failures_and_disabled_jev():
    result = pruner(
        failing("worker_failure"),
        failing("invalid_output"),
        judge_all(DROP),
        jev_enabled=False,
    ).prune(PruneRequest(output=LONG, budget_chars=3000))
    assert result.provider == "native" and result.native_fallback
    assert result.fallback_reason == "jev_disabled"
    assert result.compression_ratio == round(result.result_chars / result.original_chars, 4)


@pytest.mark.parametrize(
    "reason", ["timeout", "worker_failure", "model_unavailable", "invalid_output", "provider_error"]
)
def test_tier2_only_after_tier1_fails(reason):
    jev: list = []
    p = pruner(failing(reason), judge_all(DROP), judge_all(KEEP, jev))
    result = p.prune(PruneRequest(output=LONG, budget_chars=3000))
    assert result.tier == 2 and result.fallback_reason == reason and jev == []


def test_invalid_votes_escalate():
    def bad(request, p, blocks):
        return {b: "summarise" for b in blocks}

    result = pruner(bad, judge_all(DROP)).prune(PruneRequest(output=LONG, budget_chars=3000))
    assert result.tier == 2 and result.fallback_reason == "invalid_output"


def test_jev_only_after_both_locals_fail_and_exactly_once():
    jev: list = []
    p = pruner(failing("timeout"), failing("worker_failure"), judge_all(DROP, jev))
    result = p.prune(PruneRequest(output=LONG, budget_chars=3000))
    assert result.tier == 3 and result.jev_used and len(jev) == 1 and p.jev_calls == 1
    assert result.fallback_reason == "worker_failure"


def test_jev_disabled_is_hard_and_native_truncation_is_the_safe_end():
    jev: list = []
    p = pruner(failing("timeout"), failing("timeout"), judge_all(KEEP, jev), jev_enabled=False)
    result = p.prune(PruneRequest(output=LONG, budget_chars=3000))
    assert result.provider == "native" and jev == [] and p.jev_calls == 0
    assert not result.jev_used and result.fallback_reason == "jev_disabled"
    assert "assert 500 == 200" in result.text


def test_failed_jev_ends_in_native_truncation():
    p = pruner(failing("timeout"), failing("timeout"), failing("provider_error"))
    result = p.prune(PruneRequest(output=LONG, budget_chars=3000))
    assert result.provider == "native" and result.jev_used


# ---------------------------------------------------------------- provider judges


def test_local_judge_is_one_generation_for_all_blocks():
    class Fake:
        def __init__(self, output=None, raises=None):
            self.output, self.raises = output, raises
            self.prompts: list[str] = []

        def run_prompts(self, system_prompt, grammar, prompts, temperature, timeout_s=None):
            assert grammar == votes_grammar(3)
            self.prompts += prompts
            if self.raises:
                raise self.raises
            return self.output if self.output is not None else {"outputs": ["drop keep drop"]}

    p = plan(PruneRequest(output=LONG))
    blocks = p.blocks[:3]
    fake = Fake()
    votes = local_judge(fake)(PruneRequest(output=LONG, command="pytest"), p, blocks)
    assert list(votes.values()) == [DROP, KEEP, DROP] and list(votes) == blocks
    assert len(fake.prompts) == 1  # one prompt, one generation, however many blocks
    assert "Command: pytest" in fake.prompts[0] and "Block 3 (lines" in fake.prompts[0]
    assert len(fake.prompts[0]) < 3 * (EXCERPT_CHARS + 60) + 300
    for fake, reason in [
        (Fake(raises=subprocess.TimeoutExpired("w", 1)), "timeout"),
        (Fake({"error": "x", "reason": "local_model_unavailable"}), "model_unavailable"),
        (Fake({"error": "x", "reason": "local_error"}), "worker_failure"),
        (Fake({"outputs": [KEEP]}), "invalid_output"),
        (Fake({"outputs": ["keep keep keep", "drop drop drop"]}), "invalid_output"),
        (Fake({"outputs": ["keep maybe drop"]}), "invalid_output"),
    ]:
        with pytest.raises(PruneTierError) as err:
            local_judge(fake)(PruneRequest(output=LONG), p, blocks)
        assert err.value.reason == reason


def test_jev_judge_batches_redacts_and_keeps_key_in_header(key_file):
    answers = {f"block_{i}": {"choice": DROP} for i in range(20)}
    transport = FakeTransport(text=json.dumps({"answers": answers}))
    jev = JevProvider("https://api.typesafe.ai/v1/systemone", key_file=str(key_file),
                      transport=transport)  # fmt: skip
    leaky = LONG.replace("rootdir: /home/dev/app", f"rootdir: {FAKE_KEY}")
    request = PruneRequest(output=leaky)
    p = plan(request, block_lines=5)
    votes = jev_judge(jev)(request, p, p.blocks[:45])
    assert len(transport.calls) == 3 and len(votes) == 45
    for call in transport.calls:
        assert call["headers"]["authorization"].endswith(FAKE_KEY)
        assert FAKE_KEY not in json.dumps(call["body"])
        assert len(call["body"]["questions"]) <= 20


def test_jev_judge_rejects_malformed_answers(key_file):
    transport = FakeTransport(text=json.dumps({"answers": {"block_0": {"choice": "rewrite"}}}))
    jev = JevProvider("https://api.typesafe.ai/v1/systemone", key_file=str(key_file),
                      transport=transport)  # fmt: skip
    p = plan(PruneRequest(output=LONG))
    with pytest.raises(PruneTierError) as err:
        jev_judge(jev)(PruneRequest(output=LONG), p, p.blocks[:1])
    assert err.value.reason == "invalid_output"


# ---------------------------------------------------------------- request contract


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget_chars": 10},
        {"preserve": ("",)},
        {"caller": "not a caller"},
        {"id": "contains spaces"},
        {"metadata": {"too_large": "x" * 2_100}},
    ],
)
def test_prune_request_validation(kwargs):
    with pytest.raises(ValueError):
        PruneRequest(output="x", **kwargs)


@pytest.mark.parametrize("kwargs", [{"preserve": "file.py:12"}, {"metadata": "unsafe"}])
def test_prune_request_rejects_wrong_container_types(kwargs):
    with pytest.raises(TypeError):
        PruneRequest(output="x", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"text": None}, TypeError),
        ({"provider": ""}, ValueError),
        ({"original_chars": -1}, ValueError),
        ({"compression_ratio": 1.1}, ValueError),
        ({"attempts": "invalid"}, TypeError),
    ],
)
def test_prune_result_validation(kwargs: dict[str, Any], exception: type[Exception]):
    values: dict[str, Any] = {"text": "output", "provider": "native"}
    values.update(kwargs)
    with pytest.raises(exception):
        PruneResult(**values)


# ---------------------------------------------------------------- ledger / cli


def test_prune_receipt_never_holds_output(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl", (FAKE_KEY,))
    secret_output = LONG + f"\nleaked {FAKE_KEY}"
    request = PruneRequest(
        output=secret_output, caller="codex", command=f"pytest --token {FAKE_KEY}"
    )
    result = pruner().prune(request)
    ledger.write([ledger.prune_receipt(request, result)])
    text = ledger.path.read_text()
    assert FAKE_KEY not in text and "assert 500 == 200" not in text
    record = json.loads(text)
    assert record["operation"] == "prune" and record["caller"] == "codex"
    assert record["command_name"] == "pytest" and record["jev_used"] is False
    assert record["original_chars"] > record["result_chars"]
    assert "native_fallback" in record and "compression_ratio" in record


def _cli(*args, stdin: str = "", env: dict | None = None):
    child_env = dict(os.environ)
    child_env.pop("DECISION_PRUNE_ACTIVE", None)
    child_env.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "decision_router", *args],
        input=stdin, capture_output=True, text=True, env=child_env,
        timeout=60, check=False,
    )  # fmt: skip


def test_cli_prune_stdin_without_models_falls_back_to_native():
    cfg.update({"jev": {"enabled": False}})
    out = _cli("prune", "--caller", "freebuff", "--budget-chars", "3000", stdin=LONG)
    assert out.returncode == 0
    assert "assert 500 == 200" in out.stdout and "omitted by decision prune (native)" in out.stdout
    stats = json.loads(_cli("stats").stdout)
    assert stats["prune"]["by_caller"]["freebuff"]["calls"] == 1
    assert stats["prune"]["jev_used"] == 0 and stats["jev_calls_total"] == 0


def test_cli_prune_run_preserves_exit_status_and_stderr(tmp_path):
    cfg.update({"jev": {"enabled": False}})
    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys\n"
        "for i in range(3000): print(f'progress {i}/3000 working')\n"
        "print('ERROR: final failure in build.py:12')\n"
        "sys.stderr.write('to stderr\\n')\n"
        "raise SystemExit(3)\n"
    )
    out = _cli("prune", "--caller", "opencode2", "--", sys.executable, str(script))
    assert out.returncode == 3 and "to stderr" in out.stderr
    assert "ERROR: final failure in build.py:12" in out.stdout
    assert len(out.stdout) < 20_000


def test_cli_prune_nested_call_passes_through():
    out = _cli("prune", "--caller", "codex", stdin=LONG, env={"DECISION_PRUNE_ACTIVE": "1"})
    assert out.stdout.rstrip("\n") == LONG.rstrip("\n")


def test_cli_prune_missing_command_reports_127():
    out = _cli("prune", "--", "/nonexistent/tool")
    assert out.returncode == 127


def test_cli_prune_json_and_status(capsys):
    cfg.update({"jev": {"enabled": False}})
    assert cli.main(["prune", "status"]) == 0
    shown = capsys.readouterr().out
    assert "tier 3: typesafe" in shown and "DISABLED" in shown and "native truncation" in shown


@pytest.mark.parametrize("caller", cli.PRUNE_CALLERS)
def test_adapter_shim_supports_each_client(caller, tmp_path, capsys):
    path = tmp_path / caller / "bash"
    assert cli.main(["adapter", "shim", caller, "--path", str(path)]) == 0
    capsys.readouterr()
    shim = path.read_text()
    assert f'" prune --caller {caller} -- ' in shim
    assert path.stat().st_mode & 0o111


def test_prune_is_fast_without_models():
    started = time.monotonic()
    pruner(failing("model_unavailable"), failing("model_unavailable"),
           jev_enabled=False).prune(PruneRequest(output=CASES[3][2]))  # fmt: skip
    assert time.monotonic() - started < 2
