"""Dataset-provenance guarantees: generation source vs reproducer, and digests.

These tests pin the honest provenance relationship the manifests record:

- `generation_source_sha256` — the generator source that produced the splits;
- `current_reproducer_sha256` — the committed generator that reproduces them
  byte-for-byte.

They also seal the split digests of `local-jev-decision-dataset-v1`: any
generator change that alters the split bytes must fail here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from decision_router import specialist as spec

SPECIALIST_DIR = Path(__file__).resolve().parent.parent / "specialist"
sys.path.insert(0, str(SPECIALIST_DIR))

import provenance
import verify_artifact

# The sealed split digests of local-jev-decision-dataset-v1 (1150/198/198).
CANONICAL_DIGESTS = {
    "train": "6a5b225dbc0693b37a30d73f6c24bc417ef06d10f7766c5a257fd149af44ee19",
    "validation": "7575739347b538a926b1c72e98b6cb4300f30d45f3d0af9ae71be7ec59f09238",
    "test": "1f2cd19d048dfb683501830e580b26e5b99165849b983c483eebc33a1354d159",
}
GENERATOR = SPECIALIST_DIR / "generate_dataset.py"


def run_generator(out_dir: Path, *extra: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--out-dir", str(out_dir), *extra],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((out_dir / "dataset.manifest.json").read_text())


def digests(manifest: dict) -> dict:
    return {name: split["digest"] for name, split in manifest["splits"].items()}


def test_fresh_generation_records_equal_source_and_reproducer(tmp_path):
    manifest = run_generator(tmp_path / "data", "--train", "24", "--validation", "8", "--test", "8")
    generator_sha = provenance.sha256_file(GENERATOR)
    assert manifest["generation_source_sha256"] == generator_sha
    assert manifest["current_reproducer_sha256"] == generator_sha
    assert manifest["requested_counts"] == {"train": 24, "validation": 8, "test": 8}
    assert provenance.check(manifest, generator=GENERATOR, split_dir=tmp_path / "data") == []


def test_generation_is_deterministic_and_byte_identical(tmp_path):
    first = run_generator(tmp_path / "a", "--train", "24", "--validation", "8", "--test", "8")
    second = run_generator(tmp_path / "b", "--train", "24", "--validation", "8", "--test", "8")
    assert digests(first) == digests(second)


def test_committed_generator_reproduces_the_sealed_split_digests(tmp_path):
    """The 1150/198/198 split bytes are sealed: regeneration must match."""
    manifest = run_generator(tmp_path / "data")
    assert digests(manifest) == CANONICAL_DIGESTS
    assert {name: split["examples"] for name, split in manifest["splits"].items()} == {
        "train": 1150,
        "validation": 198,
        "test": 198,
    }
    assert provenance.check(manifest, generator=GENERATOR, split_dir=tmp_path / "data") == []


def test_check_catches_a_stale_reproducer_and_tampered_splits(tmp_path):
    manifest = run_generator(tmp_path / "data", "--train", "24", "--validation", "8", "--test", "8")
    manifest["current_reproducer_sha256"] = "0" * 64
    problems = provenance.check(manifest, generator=GENERATOR)
    assert any("current_reproducer_sha256" in p for p in problems)

    fresh = run_generator(tmp_path / "data2", "--train", "24", "--validation", "8", "--test", "8")
    (tmp_path / "data2" / "train.jsonl").write_text("{}\n")
    problems = provenance.check(fresh, generator=GENERATOR, split_dir=tmp_path / "data2")
    assert any(p.startswith("train:") for p in problems)


def test_check_accepts_an_honest_historical_source_divergence(tmp_path):
    """A dataset predating an output-preserving edit: source != reproducer is
    consistent as long as the reproducer is the generator on disk."""
    manifest = run_generator(tmp_path / "data", "--train", "24", "--validation", "8", "--test", "8")
    manifest["generation_source_sha256"] = "1" * 64  # superseded historical source
    assert provenance.check(manifest, generator=GENERATOR, split_dir=tmp_path / "data") == []


def test_verify_artifact_reports_the_whole_chain(tmp_path):
    data_dir = tmp_path / "data"
    dataset_manifest = run_generator(data_dir, "--train", "24", "--validation", "8", "--test", "8")
    artifact = tmp_path / "local-jev-tev-specialist-v1.gguf"
    artifact.write_bytes(b"GGUF fake artifact for provenance tests")
    artifact_manifest = {
        "classifier_id": spec.SPECIALIST_ID,
        "classifier_family": spec.SPECIALIST_FAMILY,
        "artifact": {
            "file": artifact.name,
            "sha256": provenance.sha256_file(artifact),
            "bytes": artifact.stat().st_size,
            "quantization": "f16",
        },
        "dataset": {
            "dataset_version": dataset_manifest["dataset_version"],
            **provenance.dataset_block(dataset_manifest),
            "splits": dataset_manifest["splits"],
        },
        "runtime_contract": {
            "system_prompt_sha256": spec.system_prompt_sha256(),
            "thinking_enabled": False,
            "option_token_grammar": True,
        },
    }
    manifest_path = tmp_path / "specialist.manifest.json"
    manifest_path.write_text(json.dumps(artifact_manifest))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tiers": {
                    "1": {
                        "model_path": str(artifact),
                        "model_sha256": artifact_manifest["artifact"]["sha256"],
                        "classifier": {"id": spec.SPECIALIST_ID, "manifest": str(manifest_path)},
                    }
                }
            }
        )
    )

    report = verify_artifact.verify(
        artifact=artifact,
        manifest_path=manifest_path,
        dataset_manifest_path=data_dir / "dataset.manifest.json",
        generator=GENERATOR,
        config_path=config_path,
        regenerate=True,
    )
    assert report["ok"], report["problems"]

    # A tampered artifact can never verify.
    artifact.write_bytes(b"GGUF tampered")
    tampered = verify_artifact.verify(
        artifact=artifact,
        manifest_path=manifest_path,
        dataset_manifest_path=data_dir / "dataset.manifest.json",
        generator=GENERATOR,
        config_path=config_path,
        regenerate=False,
    )
    assert not tampered["ok"]
    assert any("artifact digest" in p for p in tampered["problems"])
