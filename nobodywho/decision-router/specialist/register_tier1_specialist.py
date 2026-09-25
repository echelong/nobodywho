"""Register the exported Tev specialist as tier 1 (with a timestamped backup).

Writes the live config atomically (tmp + chmod 0600 + replace), mirroring
decision_router.config.update, but replaces the whole tier-1 block instead of
deep-merging it, so no stale generic-model provenance survives. The generic
escalation tier (2) and every other key are untouched byte for byte.

Standard library only, and no machine-specific paths: the config location is
the standard user config path unless --config says otherwise.

Usage:
    python specialist/register_tier1_specialist.py ARTIFACT.gguf MANIFEST.json
        [--config PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

DEFAULT_CONFIG = Path.home() / ".config" / "decision-router" / "config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    digest = manifest["artifact"]["sha256"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
        raise SystemExit("artifact digest does not match the manifest; refusing to register")
    config = json.loads(args.config.read_text())
    tier1 = config.get("tiers", {}).get("1", {})
    backup = args.config.with_name(
        f"config.json.before-tev-specialist-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    shutil.copy2(args.config, backup)
    config.setdefault("tiers", {})["1"] = {
        "label": tier1.get("label", "tier1"),
        "source": f"local-artifact:{manifest['classifier_id']}",
        "model_path": str(artifact),
        "model_sha256": digest,
        "model_info": {
            "file": artifact.name,
            "name": manifest["classifier_id"],
            "size": "0.6B",
            "architecture": "qwen3",
            "quantization": manifest["artifact"]["quantization"],
        },
        "use_gpu": True,
        "persistent": True,
        "idle_timeout_s": tier1.get("idle_timeout_s", 900),
        "evict_tiers": tier1.get("evict_tiers", ["2"]),
        "classifier": {
            "id": manifest["classifier_id"],
            "family": manifest["classifier_family"],
            "kind": manifest["classifier_kind"],
            "version": manifest["version"],
            "thinking": False,
            "option_token_grammar": True,
            "manifest": str(manifest_path),
            "on_identity_mismatch": "fail_closed",
        },
    }
    tmp = args.config.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(args.config)
    print(json.dumps({"registered": str(artifact), "sha256": digest, "backup": str(backup)}))


if __name__ == "__main__":
    main()
