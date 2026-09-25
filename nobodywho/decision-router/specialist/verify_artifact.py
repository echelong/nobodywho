"""Verify one Tev-specialist installation end to end, without loading the model.

Checks, in order:

1. the artifact file hashes to the manifest's recorded SHA-256 and byte size;
2. the manifest identity is the registered specialist with the trained prompt
   contract (thinking off, option-token grammar on), and — when run inside the
   runtime virtualenv — matches the runtime's own contract constants;
3. the dataset provenance is consistent: the recorded generation source is a
   digest, the recorded reproducer is the committed generator source on disk,
   and every split file matches its recorded digest; with --regenerate the
   dataset is also regenerated from scratch and must reproduce those digests
   byte-for-byte;
4. the live config pin (model_path / model_sha256 / manifest path) agrees with
   the manifest.

Exit code 0 when every check passes. Standard library only; no
machine-specific paths (all locations are arguments with documented defaults).

Usage:
    python specialist/verify_artifact.py [--artifact PATH] [--manifest PATH]
        [--dataset-manifest PATH] [--generator PATH] [--config PATH]
        [--no-regenerate] [--json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance

SPECIALIST_ID = "local-jev-tev-specialist-v1"
SPECIALIST_FAMILY = "tev-style-specialist"
DEFAULT_ROOT = Path.home() / ".local" / "share" / "decision-router" / "tev-specialist-v1"
DEFAULT_CONFIG = Path.home() / ".config" / "decision-router" / "config.json"


def _regenerated_digests(generator: Path, counts: dict[str, int]) -> dict[str, str] | str:
    """Split digests from a fresh generation, or the error that prevented it."""
    with tempfile.TemporaryDirectory(prefix="tev-verify-") as tmp:
        out_dir = Path(tmp) / "dataset"
        completed = subprocess.run(
            [
                sys.executable,
                str(generator),
                "--out-dir",
                str(out_dir),
                "--train",
                str(int(counts.get("train", 1150))),
                "--validation",
                str(int(counts.get("validation", 200))),
                "--test",
                str(int(counts.get("test", 200))),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
            return "regeneration failed: " + " | ".join(tail)
        manifest = json.loads((out_dir / "dataset.manifest.json").read_text())
        return {name: split["digest"] for name, split in manifest["splits"].items()}


def verify(
    *,
    artifact: Path,
    manifest_path: Path,
    dataset_manifest_path: Path,
    generator: Path,
    config_path: Path | None,
    regenerate: bool = True,
) -> dict[str, Any]:
    """The verification report: every claim checked against the files on disk."""
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    # 1. the artifact itself
    if not artifact.is_file():
        record("artifact", False, f"{artifact} is missing")
        return {"ok": False, "checks": checks, "problems": problems}
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.get("artifact") or {}
    digest = provenance.sha256_file(artifact)
    record(
        "artifact digest",
        digest == declared.get("sha256"),
        f"sha256={digest} declared={declared.get('sha256')}",
    )
    record(
        "artifact size",
        artifact.stat().st_size == declared.get("bytes"),
        f"bytes={artifact.stat().st_size} declared={declared.get('bytes')}",
    )

    # 2. identity and prompt contract
    record(
        "identity",
        manifest.get("classifier_id") == SPECIALIST_ID
        and manifest.get("classifier_family") == SPECIALIST_FAMILY,
        f"{manifest.get('classifier_id')} / {manifest.get('classifier_family')}",
    )
    contract = manifest.get("runtime_contract") or {}
    record(
        "prompt contract",
        contract.get("thinking_enabled") is False
        and contract.get("option_token_grammar") is True
        and bool(contract.get("system_prompt_sha256")),
        f"thinking={contract.get('thinking_enabled')} grammar={contract.get('option_token_grammar')}",
    )
    try:
        from decision_router import specialist as spec  # runtime virtualenv only

        trained = contract.get("system_prompt_sha256")
        record(
            "runtime contract match",
            trained == spec.system_prompt_sha256(),
            f"manifest={trained} runtime={spec.system_prompt_sha256()}",
        )
    except ImportError:
        record("runtime contract match", True, "skipped: decision_router not importable here")

    # 3. dataset provenance
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    provenance_problems = provenance.check(
        dataset_manifest, generator=generator, split_dir=dataset_manifest_path.parent
    )
    if provenance_problems:
        for problem in provenance_problems:
            record("dataset provenance", False, problem)
    else:
        record("dataset provenance", True, f"reproducer={generator.name} and split digests match")
    embedded = provenance.dataset_block(dataset_manifest)
    record(
        "artifact manifest provenance",
        embedded == provenance.dataset_block(manifest.get("dataset") or {}),
        "the artifact manifest records the same dataset provenance fields",
    )
    if regenerate:
        counts = dataset_manifest.get("requested_counts") or {
            name: split.get("examples")
            for name, split in (dataset_manifest.get("splits") or {}).items()
        }
        regenerated = _regenerated_digests(generator, counts)
        if isinstance(regenerated, str):
            record("byte-identical regeneration", False, regenerated)
        else:
            recorded = {
                name: split.get("digest")
                for name, split in (dataset_manifest.get("splits") or {}).items()
            }
            record(
                "byte-identical regeneration",
                regenerated == recorded,
                f"regenerated={regenerated} recorded={recorded}",
            )

    # 4. the live config pin
    if config_path is not None and config_path.is_file():
        config = json.loads(config_path.read_text())
        pins = [
            settings
            for settings in (config.get("tiers") or {}).values()
            if (settings.get("classifier") or {}).get("id") == SPECIALIST_ID
        ]
        if not pins:
            record("config pin", False, "no tier registers the specialist classifier")
        else:
            pin = pins[0]
            record(
                "config pin",
                pin.get("model_sha256") == digest
                and Path(str(pin.get("model_path"))).resolve() == artifact.resolve()
                and Path(str((pin.get("classifier") or {}).get("manifest"))).resolve()
                == manifest_path.resolve(),
                f"model_sha256={pin.get('model_sha256')} model_path={pin.get('model_path')}",
            )
    elif config_path is not None:
        record("config pin", False, f"{config_path} is missing")

    return {"ok": not problems, "checks": checks, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ROOT / "artifact" / "local-jev-tev-specialist-v1.gguf",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_ROOT / "artifact" / "specialist.manifest.json"
    )
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_ROOT / "dataset" / "dataset.manifest.json"
    )
    parser.add_argument(
        "--generator", type=Path, default=Path(__file__).resolve().parent / "generate_dataset.py"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--no-regenerate", action="store_true", help="skip the byte-identical regeneration check"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()
    report = verify(
        artifact=args.artifact,
        manifest_path=args.manifest,
        dataset_manifest_path=args.dataset_manifest,
        generator=args.generator,
        config_path=args.config,
        regenerate=not args.no_regenerate,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            print(f"[{'ok' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}")
        print("verified" if report["ok"] else "FAILED")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
