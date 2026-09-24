"""The latency paths: pruning work per call, early stopping, worker start and CLI start-up."""

from __future__ import annotations

import json
import logging
import random
import re
import struct
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import ClassVar

import pytest
from helpers import make_request

from decision_router import config as cfg
from decision_router import latency, local_worker
from decision_router import prune as prune_module
from decision_router.gguf_info import describe
from decision_router.providers.nobodywho import NobodyWhoProvider, PersistentWorker, aggregate
from decision_router.prune import (
    DROP,
    KEEP,
    Plan,
    Pruner,
    PruneRequest,
    Tier,
    assemble,
    candidates,
    fit,
    plan,
)
from decision_router.prune_corpus import cases, judgement_cases, sized_cases

SMALL = [case for size, case in sized_cases() if size == "small"]
LARGE = next(case for size, case in sized_cases() if size == "large")


def reference_fit(p: Plan, votes: dict, budget: int, label: str) -> tuple[str, set[int]]:
    """The original quadratic algorithm: re-assemble after every dropped block."""
    keep = set(p.critical)
    kept = [b for b in p.blocks if votes.get(b, KEEP) == KEEP]
    for b in kept:
        keep.update(range(*b))
    text = assemble(p.lines, keep, label)
    if len(text) <= budget:
        return text, keep
    n = len(p.lines)
    kept.sort(key=lambda b: min(b[0], n - b[1]), reverse=True)
    for b in kept:
        keep.difference_update(range(*b))
        keep.update(p.critical & set(range(*b)))
        text = assemble(p.lines, keep, label)
        if len(text) <= budget:
            break
    return text, keep


# ---------------------------------------------------------------- deterministic pruning work


@pytest.mark.parametrize("name, command, output, must_keep", cases() + SMALL,
                         ids=[c[0] for c in cases() + SMALL])  # fmt: skip
def test_linear_fit_matches_the_original_algorithm(name, command, output, must_keep):
    rng = random.Random(name)
    p = plan(PruneRequest(output=output), block_lines=10)
    for budget in (600, 3000, 12_000, 10**6):
        votes = {b: rng.choice((KEEP, DROP)) for b in p.blocks}
        assert fit(p, votes, budget, "t") == reference_fit(p, votes, budget, "t"), (name, budget)


def test_fit_assembles_once_however_many_blocks_are_dropped(monkeypatch):
    calls = []
    real = prune_module.assemble
    monkeypatch.setattr(prune_module, "assemble", lambda *a: calls.append(1) or real(*a))
    p = plan(PruneRequest(output=LARGE[2]))
    assert len(p.blocks) > 100
    fit(p, {}, 12_000, "t")
    assert len(calls) == 1


def test_each_line_is_matched_against_critical_once(monkeypatch):
    searched = []
    pattern = prune_module.CRITICAL
    proxy = types.SimpleNamespace(search=lambda line: searched.append(1) or pattern.search(line))
    monkeypatch.setattr(prune_module, "CRITICAL", proxy)
    output = SMALL[0][2]
    Pruner([Tier(1, "fake", "m", lambda r, p, b: {x: DROP for x in b})],
           jev_enabled=False).prune(PruneRequest(output=output))  # fmt: skip
    assert len(searched) == len(output.splitlines())


ORIGINAL_FILE_REFS = re.compile(r"(?ix)[\w./-]+\.\w{1,6}:\d+|[\w./-]+\.\w{1,6}\(\d+,\d+\)")


def test_anchored_file_references_match_the_same_lines():
    file_refs = re.compile(r"(?ix)(?<![\w./-])[\w./-]+\.\w{1,6}(?::\d+|\(\d+,\d+\))")
    lines = [line for _, _, output, _ in cases() + SMALL for line in output.splitlines()]
    lines += ["src/lib.rs:10:5 error", "at x (a/b.js:3:1)", "file.ts(88,14): e", "v1.2.3",
              "see ./a.b.c:12", "1.5:2", "foo.bar(1,2)", "no refs here", "x" * 300 + ".py:1"]  # fmt: skip
    for line in lines:
        assert bool(file_refs.search(line)) == bool(ORIGINAL_FILE_REFS.search(line)), line


def test_large_output_prunes_quickly_without_a_model():
    started = time.monotonic()
    result = Pruner([], jev_enabled=False).prune(PruneRequest(output=LARGE[2]))
    assert time.monotonic() - started < 2.5
    assert all(fact in result.text for fact in LARGE[3])


# ---------------------------------------------------------------- what a model is asked


def test_candidates_fill_the_room_from_both_ends_and_stop():
    p = plan(PruneRequest(output=LARGE[2]))
    chosen = candidates(p, 12_000, 24, "t")
    assert 0 < len(chosen) < 24 < len(p.blocks)
    n = len(p.lines)
    distance = sorted(min(b[0], n - b[1]) for b in p.blocks)
    assert sorted(min(b[0], n - b[1]) for b in chosen) == distance[: len(chosen)]
    in_order = sorted(chosen, key=lambda b: (min(b[0], n - b[1]), b[0]))
    size = sum(len(line) + 1 for b in in_order[:-1] for line in p.lines[b[0] : b[1]])
    room = 12_000 - len(assemble(p.lines, p.critical, "t"))
    assert size < 2 * room  # the last block was added while there was still room
    assert len(candidates(p, 12_000, 3, "t")) == 3


def test_no_model_call_when_critical_lines_fill_the_budget():
    judged = []
    search = next(c for c in cases() if c[0] == "search")[2]  # every line is a file:line hit
    result = Pruner([Tier(1, "fake", "m", lambda r, p, b: judged.append(b) or {})],
                    jev_enabled=False).prune(PruneRequest(output=search, budget_chars=3000))  # fmt: skip
    assert judged == [] and result.result_chars < len(search)


def test_one_judgement_per_tier_for_a_large_output():
    calls: list[int] = []
    result = Pruner([Tier(1, "fake", "m", lambda r, p, b: calls.append(len(b)) or
                          {x: DROP for x in b}, max_blocks=8)],
                    jev_enabled=False).prune(PruneRequest(output=LARGE[2]))  # fmt: skip
    assert len(calls) == 1 and 0 < calls[0] <= 8
    assert result.tier == 1 and all(fact in result.text for fact in LARGE[3])


def test_ansi_sequences_are_removed_before_planning():
    _, _, output, must_keep = SMALL[0]
    colored = "\n".join(f"\x1b[32m{line}\x1b[0m" if "PASSED" in line else
                        f"\x1b[1;31m{line}\x1b[0m" for line in output.splitlines())  # fmt: skip
    result = Pruner([], jev_enabled=False).prune(PruneRequest(output=colored))
    assert "\x1b" not in result.text
    assert all(fact in result.text for fact in must_keep)


def test_identical_consecutive_lines_keep_one_copy():
    lines = ["building"] + ["warning: deprecated call in legacy API"] * 3000
    lines += ["ERROR: link failed in main.c:9", "done"]
    result = Pruner([], jev_enabled=False).prune(PruneRequest(output="\n".join(lines)))
    assert result.text.count("warning: deprecated call in legacy API") == 1
    assert "ERROR: link failed in main.c:9" in result.text
    assert "2999 lines omitted" in result.text
    assert "critical output truncated" not in result.text


def test_labelled_judgement_blocks_are_never_critical():
    for _, labelled in judgement_cases():
        for label, block in labelled:
            assert label in (KEEP, DROP)
            assert not any(prune_module.CRITICAL.search(line) for line in block), block


# ---------------------------------------------------------------- early stopping (worker)


@pytest.mark.parametrize(
    ("outputs", "planned", "share", "expected"),
    [
        (["a", "a"], 3, 0.66, True),  # 2 of 3 settle: margin >= 1, share 2/3
        (["a", "a"], 3, 0.9, False),  # a stricter policy needs the third sample
        (["a", "b"], 3, 0.66, False),
        (["a"], 3, 0.66, False),
        (["a", "a", "a"], 5, 0.66, False),  # 3/5 = 0.6 < 0.66 even if the rest agree
        (["a", "a", "a", "a"], 5, 0.66, True),
        (["a", "b", "c"], 3, 0.66, True),  # nothing left to draw
    ],
)
def test_settled(outputs, planned, share, expected):
    assert local_worker.settled(outputs, planned, share, 1) is expected


class FakeStream:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = list(tokens)
        self.pulled = 0

    def next_token(self):
        if not self.tokens:
            return None
        self.pulled += 1
        return self.tokens.pop(0)

    def completed(self):
        text = "".join(self.tokens)
        self.tokens = []
        return text


class FakeChat:
    script: ClassVar[list[list[str]]] = []
    created = 0

    def __init__(self, model, n_ctx=2048, system_prompt="", template_variables=None):
        FakeChat.created += 1
        self.system_prompt = system_prompt
        self.stops = 0
        self.streams: list[FakeStream] = []

    def ask(self, prompt):
        stream = FakeStream(FakeChat.script.pop(0))
        self.streams.append(stream)
        return stream

    def stop_generation(self):
        self.stops += 1

    def reset_history(self):
        pass

    def set_sampler_config(self, config):
        pass

    def get_system_prompt(self):
        return self.system_prompt

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt


def fake_nobodywho(monkeypatch) -> types.SimpleNamespace:
    loads: list[str] = []

    class Model:
        def __init__(self, path, use_gpu_if_available=True):
            loads.append(path)
            logging.getLogger("nobodywho.llm").info("Loading model use_gpu=true, gpu_layers=21")

    class Builder:
        def grammar(self, *a):
            return self

        def temperature(self, t):
            return self

        def seed(self, s):
            return self

        def dist(self):
            return "sampler"

    module = types.SimpleNamespace(Model=Model, Chat=FakeChat, SamplerBuilder=Builder,
                                   __version__="test", loads=loads)  # fmt: skip
    monkeypatch.setitem(sys.modules, "nobodywho", module)
    FakeChat.created = 0
    return module


def job(tmp_path, samples: int, **extra) -> dict:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    return {"model_path": str(model), "system_prompt": "s", "grammar": 'root ::= "x"',
            "samples": [{"seed": i, "prompt": "p"} for i in range(samples)],
            "temperature": 0.7, "n_ctx": 2048, "use_gpu": True, **extra}  # fmt: skip


def test_worker_stops_a_sample_once_its_prefix_is_unique(monkeypatch, tmp_path):
    fake_nobodywho(monkeypatch)
    FakeChat.script = [["inspect", "_tz"], ["re", "try", "_same"]]
    cache: dict = {}
    out = local_worker.run(job(tmp_path, 2, choices=["retry_same", "inspect_tz", "rewrite"]), cache)
    assert out["outputs"] == ["inspect_tz", "retry_same"]
    chat = cache["chat"]
    assert [s.pulled for s in chat.streams] == [1, 2]  # "re" alone is still ambiguous
    assert chat.stops == 2


def test_worker_stops_sampling_once_the_vote_is_settled(monkeypatch, tmp_path):
    fake_nobodywho(monkeypatch)
    FakeChat.script = [["a"], ["a"], ["b"]]
    stop = {"min_share": 0.66, "min_margin": 1}
    out = local_worker.run(job(tmp_path, 3, choices=["a", "b"], stop_when=stop), {})
    assert out["outputs"] == ["a", "a"] and out["planned"] == 3
    assert FakeChat.script == [["b"]]  # the third sample was never drawn


def test_worker_reuses_the_model_across_context_sizes(monkeypatch, tmp_path):
    module = fake_nobodywho(monkeypatch)
    FakeChat.script = [["x"], ["x"], ["x"]]
    cache: dict = {}
    first = local_worker.run(job(tmp_path, 1), cache)
    second = local_worker.run(job(tmp_path, 1, n_ctx=4096), cache)
    third = local_worker.run(job(tmp_path, 1, n_ctx=4096), cache)
    assert len(module.loads) == 1 and FakeChat.created == 2
    assert first["model_reused"] is False and second["model_reused"] is True
    assert first["gpu_layers"] == 21 and third["gpu_layers"] == 21


def test_aggregate_counts_an_early_stop_against_every_planned_sample():
    result = aggregate(make_request(), ["inspect_tz", "inspect_tz"], 5, "m", {}, planned=3)
    assert result.choice == "inspect_tz" and result.confidence == pytest.approx(2 / 3)
    assert result.votes == {"inspect_tz": 2} and result.samples == 2
    assert result.details["samples_planned"] == 3 and result.details["early_stop"] is True


def test_decision_jobs_carry_choices_and_the_acceptance_thresholds(tmp_path):
    jobs = []

    def runner(job, timeout):
        jobs.append(job)
        return {"outputs": ["inspect_tz"] * 2, "planned": 3}

    config = cfg.load()
    config["tiers"]["1"]["model_path"] = str(tmp_path / "m.gguf")
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    result = NobodyWhoProvider.for_tier(config, "1", runner=runner).decide(make_request())
    assert jobs[0]["choices"] == list(make_request().allowed())
    assert jobs[0]["stop_when"] == {"min_share": 0.66, "min_margin": 1}
    assert result.ok and result.confidence == pytest.approx(2 / 3)
    config["tiers"]["1"]["early_stop"] = False
    NobodyWhoProvider.for_tier(config, "1", runner=runner).decide(make_request())
    assert "stop_when" not in jobs[1]


# ---------------------------------------------------------------- worker start, GGUF, CLI


def test_worker_that_dies_during_start_fails_fast():
    worker = PersistentWorker("/bin/false", cfg.runtime_dir(), start_timeout_s=30)
    started = time.monotonic()
    output = worker({"model_path": "/nonexistent", "samples": []}, 60)
    assert output["reason"] == "local_error" and "exited during start" in output["error"]
    assert time.monotonic() - started < 3


def test_worker_socket_path_that_is_too_long_is_refused(tmp_path):
    worker = PersistentWorker(sys.executable, tmp_path / ("d" * 120))
    output = worker({"model_path": "/nonexistent", "samples": []}, 10)
    assert "socket path too long" in output["error"]
    assert not (tmp_path / ("d" * 120)).exists()


def test_gguf_layers_and_string_arrays(tmp_path):
    def s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    vocab = [f"tok{i}" for i in range(5000)]
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 5)
    body += s("general.architecture") + struct.pack("<I", 8) + s("qwen3")
    body += s("general.name") + struct.pack("<I", 8) + s("Tiny")
    body += s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(vocab))
    body += b"".join(s(t) for t in vocab)
    body += s("qwen3.block_count") + struct.pack("<II", 4, 36)
    body += s("general.file_type") + struct.pack("<II", 4, 15)
    path = tmp_path / "tiny.gguf"
    path.write_bytes(body)
    info = describe(path)
    assert info["layers"] == 37 and info["quantization"] == "Q4_K_M" and info["name"] == "Tiny"


def test_cli_start_up_does_not_load_the_http_stack():
    code = ("import sys, decision_router.cli; "
            "print([m for m in ('urllib.request', 'http.client', 'concurrent.futures', 'logging')"
            " if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------- `decision benchmark latency`


def test_latency_session_is_private_and_cannot_reach_jev(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "tsk_should_not_leak")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    session = latency.Session(latency.Model("extra", str(gguf), True, {"layers": 5}), 30)
    try:
        config = json.loads((session.root / "cfg" / "config.json").read_text())
        assert config["jev"]["enabled"] is False
        assert not Path(config["jev"]["key_file"]).exists()
        assert config["tiers"]["1"]["model_path"] == str(gguf)
        assert config["tiers"]["2"]["model_path"] is None
        assert "TYPESAFE_API_KEY" not in session.env
        assert session.env["DECISION_ROUTER_STATE_DIR"].startswith(str(session.root))
    finally:
        session.close()
    assert not session.root.exists()


def test_latency_model_selection(tmp_path):
    small, big = tmp_path / "small.gguf", tmp_path / "big.gguf"
    small.write_bytes(b"GGUF" + b"\0" * 10)
    big.write_bytes(b"GGUF" + b"\0" * 100)
    config = cfg.load()
    config["tiers"]["1"]["model_path"] = str(big)
    config["tiers"]["2"]["model_path"] = str(big)  # the same file is measured once
    assert [m.label for m in latency.select_models(config, [], [], False, False)] == ["tier1"]
    only = latency.select_models(config, [], [str(small)], False, True)
    assert [(m.label, m.use_gpu) for m in only] == [("extra", False)]
    both = latency.select_models(config, ["tier1"], [str(small)], False, False)
    assert [Path(m.path).name for m in both] == ["small.gguf", "big.gguf"]


def test_latency_table_shows_every_column():
    report = {"models": [{"model": "Qwen3 4B", "use_gpu": True, "gpu_offload": "37/37 layers",
                          "rows": [{"operation": "decision", "cold_ms": 1600, "p50_ms": 230,
                                    "p95_ms": 290, "p99_ms": 310, "rss_mib": 1300,
                                    "vram_mib": 2990, "quality_pass": 24,
                                    "quality_total": 24}]}]}  # fmt: skip
    text = latency.table(report)
    header, row = text.splitlines()
    for column in ("MODEL", "OPERATION", "COLD", "P50", "P95", "P99", "RAM", "VRAM",
                   "GPU OFFLOAD", "QUALITY PASS RATE"):  # fmt: skip
        assert column in header
    assert "37/37 layers" in row and "24/24 (100%)" in row and "230ms" in row


# ---------------------------------------------------------------- VRAM guard and eviction


def test_gpu_fit_needs_the_file_plus_headroom(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0" * (4 << 20))
    assert local_worker.gpu_fits(str(model), None)  # unknown: NobodyWho decides
    assert local_worker.gpu_fits(str(model), 4 + local_worker.GPU_HEADROOM_MIB)
    assert not local_worker.gpu_fits(str(model), 3 + local_worker.GPU_HEADROOM_MIB)


def test_model_that_does_not_fit_loads_on_cpu_or_fails_fast(monkeypatch, tmp_path):
    module = fake_nobodywho(monkeypatch)
    gpu_requests = []
    real_model = module.Model

    class Model(real_model):
        def __init__(self, path, use_gpu_if_available=True):
            gpu_requests.append(use_gpu_if_available)
            super().__init__(path, use_gpu_if_available)

    module.Model = Model
    monkeypatch.setattr(local_worker, "free_vram_mib", lambda: 1)
    FakeChat.script = [["x"]]
    out = local_worker.run(job(tmp_path, 1), {})
    assert gpu_requests == [False] and out["gpu_layers"] == 0
    assert "loaded on CPU" in out["load_warnings"][0]
    failed = local_worker.run(job(tmp_path, 1, cpu_fallback=False), {})
    assert failed["reason"] == "local_gpu_unavailable" and gpu_requests == [False]


def test_each_gpu_tier_evicts_the_other_before_a_cold_start(tmp_path):
    config = cfg.load()
    for tier in ("1", "2"):
        config["tiers"][tier].update(model_path=str(tmp_path / f"t{tier}.gguf"), use_gpu=True)
    names = {t: [w.name for w in NobodyWhoProvider.for_tier(config, t).evict] for t in ("1", "2")}
    assert names == {"1": ["tier2-worker"], "2": ["tier1-worker"]}
    config["tiers"]["2"]["use_gpu"] = False  # a CPU tier holds no VRAM worth freeing
    assert NobodyWhoProvider.for_tier(config, "1").evict == []


def test_vram_guard_waits_briefly_for_an_evicted_worker(monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0" * (1 << 20))
    free = iter([0, 0, 10_000])  # the evicted worker's memory shows up on the third read
    monkeypatch.setattr(local_worker, "free_vram_mib", lambda: next(free))
    monkeypatch.setattr(local_worker.time, "sleep", lambda s: None)
    assert local_worker.gpu_fits_soon(str(model))
    monkeypatch.setattr(local_worker, "free_vram_mib", lambda: 0)
    assert not local_worker.gpu_fits_soon(str(model), wait_s=0)


def test_stop_waits_until_the_worker_has_exited():
    worker = PersistentWorker(sys.executable, cfg.runtime_dir(), idle_timeout_s=30)
    worker({"model_path": "/nonexistent", "samples": []}, 10)
    pid = worker.pid()
    assert pid is not None and worker.stop() is True
    state = ""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        pass
    assert state in ("", "Z", "X")
