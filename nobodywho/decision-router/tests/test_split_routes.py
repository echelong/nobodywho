"""Decision and semantic-pruning tiers can use different models safely."""

from __future__ import annotations

import copy

from decision_router import cli
from decision_router import config as cfg
from decision_router.providers.nobodywho import NobodyWhoProvider, PersistentWorker
from decision_router.prune import DROP, KEEP, PruneRequest, PruneTierError, Tier
from decision_router.prune_corpus import cases


def split_config(tmp_path):
    config = copy.deepcopy(cfg.DEFAULTS)
    specialist = tmp_path / "local-jev-tev-specialist-v1.gguf"
    four_b = tmp_path / "Qwen_Qwen3-4B-Q4_K_M.gguf"
    nine_b = tmp_path / "Qwen3.5-9B-Q4_K_M.gguf"
    for model in (specialist, four_b, nine_b):
        model.write_bytes(b"GGUF")
    config["tiers"]["1"].update(
        model_path=str(specialist),
        use_gpu=True,
        persistent=True,
        classifier={"id": "local-jev-tev-specialist-v1"},
    )
    config["tiers"]["2"].update(model_path=str(nine_b), use_gpu=True, persistent=True)
    config["prune_tiers"]["1"] = {
        "model_path": str(four_b),
        "use_gpu": True,
        "persistent": True,
    }
    return config


def test_split_models_and_worker_names(tmp_path):
    config = split_config(tmp_path)
    decision = NobodyWhoProvider.for_tier(config, "1")
    prune = NobodyWhoProvider.for_tier(config, "1", operation="prune")
    assert decision.model_name == "local-jev-tev-specialist-v1"
    assert prune.model_name == "Qwen_Qwen3-4B-Q4_K_M"
    assert isinstance(decision.runner, PersistentWorker)
    assert isinstance(prune.runner, PersistentWorker)
    assert decision.runner.name == "tier1-worker"
    assert prune.runner.name == "prune-tier1-worker"
    assert decision.runner.socket_path != prune.runner.socket_path
    assert "classifier" in cfg.tier_settings(config, "1")
    assert "classifier" not in cfg.tier_settings(config, "1", operation="prune")


def test_nine_b_eviction_covers_both_resident_primary_workers(tmp_path):
    config = split_config(tmp_path)
    nine_b = NobodyWhoProvider.for_tier(config, "2")
    assert {worker.name for worker in nine_b.evict} == {
        "tier1-worker",
        "prune-tier1-worker",
    }


def test_builders_and_display_follow_split_config(tmp_path, monkeypatch):
    config = split_config(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled JEV provider was constructed")

    monkeypatch.setattr(cli.JevProvider, "from_config", forbidden)
    router = cli.build_router(config)
    pruner = cli.build_pruner(config)
    tier1 = router.tiers[1]
    assert isinstance(tier1, NobodyWhoProvider)
    assert tier1.model_name == "local-jev-tev-specialist-v1"
    assert [tier.model for tier in pruner.tiers] == [
        "Qwen_Qwen3-4B-Q4_K_M",
        "Qwen3.5-9B-Q4_K_M",
    ]
    assert "jev" not in router.providers
    assert pruner.jev_enabled is False
    display = cli.describe_mode(config)
    assert "D1: nobodywho / local-jev-tev-specialist-v1" in display
    assert "P1: nobodywho / Qwen_Qwen3-4B-Q4_K_M" in display


def test_legacy_config_reuses_decision_tier_for_pruning(tmp_path):
    config = split_config(tmp_path)
    del config["prune_tiers"]
    assert (
        cfg.tier_settings(config, "1", operation="prune")["model_path"]
        == (config["tiers"]["1"]["model_path"])
    )
    assert cfg.tier_worker_name(config, "1", operation="prune") == "tier1-worker"


def test_p0_bypasses_all_split_models(tmp_path):
    config = split_config(tmp_path)
    pruner = cli.build_pruner(config)
    output = "running\n" + "progress 5%\n" * 400 + "error: failed at src/main.rs:42\n"
    result = pruner.prune(PruneRequest(output=output, budget_chars=1000))
    assert result.provider == "deterministic" and result.tier == 0
    assert all(a.get("provider") != "nobodywho" for a in result.attempts)


def test_split_prune_receipt_and_nine_b_fallback(tmp_path):
    config = split_config(tmp_path)
    pruner = cli.build_pruner(config)
    calls: list[str] = []

    def keep(_request, _plan, blocks):
        calls.append("4b")
        return {block: KEEP for block in blocks}

    def fallback(_request, _plan, blocks):
        calls.append("9b")
        return {block: DROP for block in blocks}

    pruner.tiers = [
        Tier(1, "nobodywho", "Qwen_Qwen3-4B-Q4_K_M", keep),
        Tier(2, "nobodywho", "Qwen3.5-9B-Q4_K_M", fallback),
    ]
    request = PruneRequest(output=cases()[0][2], budget_chars=4000)
    accepted = pruner.prune(request)
    assert accepted.provider == "nobodywho" and accepted.tier == 1
    assert accepted.model == "Qwen_Qwen3-4B-Q4_K_M"
    assert calls == ["4b"]

    def fail(_request, _plan, _blocks):
        calls.append("4b-failed")
        raise PruneTierError("worker_failure")

    pruner.tiers[0].judge = fail
    calls.clear()
    escalated = pruner.prune(request)
    assert calls == ["4b-failed", "9b"]
    assert escalated.tier == 2 and escalated.model == "Qwen3.5-9B-Q4_K_M"


def test_model_specific_stats_do_not_mix_routes():
    ask = [
        {
            "request_id": "a",
            "caller": "codex",
            "provider": "nobodywho",
            "tier": 1,
            "model": "local-jev-tev-specialist-v1",
            "latency_ms": 100,
        },
    ]
    prune = [
        {"caller": "codex", "provider": "deterministic", "tier": 0, "latency_ms": 5},
        {
            "caller": "codex",
            "provider": "nobodywho",
            "tier": 1,
            "model": "Qwen_Qwen3-4B-Q4_K_M",
            "latency_ms": 220,
        },
    ]
    assert cli.summarize_asks(ask)["median_latency_ms_by_model"] == {
        "nobodywho/tier1/local-jev-tev-specialist-v1": 100,
    }
    assert cli.summarize_prunes(prune)["median_latency_ms_by_model"] == {
        "nobodywho/tier1/Qwen_Qwen3-4B-Q4_K_M": 220,
    }
    assert cli.summarize_prunes(prune)["median_latency_ms"] is None
    assert cli.summarize_prunes(prune)["median_latency_ms_by_route"] == {
        "deterministic/tier0": 5,
        "nobodywho/tier1": 220,
    }
