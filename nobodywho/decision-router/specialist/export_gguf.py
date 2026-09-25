"""Merge the selected adapter, convert to GGUF and write the provenance manifest.

Runs in the training virtualenv (torch/peft + the `gguf` package); the runtime
never needs any of it. The manifest it writes is exactly what
`decision_router.specialist.verify` checks: classifier identity, the artifact
SHA-256, and the training-time prompt contract (thinking off, option-token
grammar on).

Every claim in the manifest is a measured digest:

- the GGUF is hashed after conversion, in this process;
- the base model, the adapter, the dataset splits and the converter script are
  hashed from the files on disk;
- the training report is embedded, including which epoch validation selected.

The test split is never read here either; selection happened on validation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from decision_router import specialist as spec


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def die(message: str) -> None:
    print(f"export_gguf: {message}", file=sys.stderr)
    raise SystemExit(2)


def merge_adapter(base_model: Path, adapter: Path, merged_dir: Path) -> None:
    """Merge the LoRA adapter into the base weights and save a plain HF model."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)


def convert_to_gguf(converter: Path, merged_dir: Path, out_file: Path) -> None:
    """Run llama.cpp's convert_hf_to_gguf.py in this interpreter (needs `gguf`)."""
    completed = subprocess.run(
        [
            sys.executable,
            str(converter),
            str(merged_dir),
            "--outfile",
            str(out_file),
            "--outtype",
            "f16",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-6:]
        die("GGUF conversion failed:\n" + "\n".join(tail))
    if not out_file.is_file():
        die(f"the converter reported success but wrote no file at {out_file}")


def build_manifest(
    out_file: Path,
    base_model: Path,
    adapter: Path,
    dataset_dir: Path,
    train_report: Path,
    converter: Path,
) -> dict:
    dataset_manifest = json.loads((dataset_dir / "dataset.manifest.json").read_text())
    contract = dataset_manifest.get("runtime_contract") or {}
    if contract.get("system_prompt_sha256") != spec.system_prompt_sha256():
        die("the dataset manifest was not rendered with this system prompt contract")
    if not (dataset_manifest.get("dataset_version") and dataset_manifest.get("splits")):
        die("the dataset manifest is incomplete")
    report = json.loads(train_report.read_text())
    selection = report.get("selection") or {}
    epoch = selection.get("epoch")
    if not epoch:
        die("the training report records no validation-selected epoch")
    adapter_weights = adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        die(f"the selected adapter has no weights at {adapter_weights}")
    base_weights = base_model / "model.safetensors"
    if not base_weights.is_file():
        # be explicit rather than guessing which shard is which
        die(f"the base model weights were not found at {base_weights}")
    return {
        "classifier_id": spec.SPECIALIST_ID,
        "classifier_family": spec.SPECIALIST_FAMILY,
        "classifier_kind": spec.SPECIALIST_KIND,
        "version": f"{dataset_manifest['dataset_version']}+lora-r16-e{epoch}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact": {
            "file": out_file.name,
            "sha256": sha256_file(out_file),
            "bytes": out_file.stat().st_size,
            "quantization": "f16",
            "architecture": (report.get("base_model") or {}).get("architectures"),
        },
        "base_model": {
            "config_name": (report.get("base_model") or {}).get("config_name"),
            "weights_sha256": sha256_file(base_weights),
        },
        "adapter": {
            "selection": selection,
            "weights_sha256": sha256_file(adapter_weights),
            "lora": report.get("lora"),
        },
        "dataset": {
            "dataset_version": dataset_manifest["dataset_version"],
            "generator_sha256": dataset_manifest["generator_sha256"],
            "splits": {
                name: {"examples": split["examples"], "digest": split["digest"]}
                for name, split in dataset_manifest["splits"].items()
            },
            "test_split_used_for_selection": False,
        },
        "training": {
            "hyperparameters": report.get("hyperparameters"),
            "selection_rule": report.get("selection_rule"),
            "history": report.get("history"),
            "runtime_seconds": report.get("runtime_seconds"),
            "torch": report.get("torch"),
            "gpu": report.get("gpu"),
        },
        "runtime_contract": {
            "system_prompt": spec.SPECIALIST_SYSTEM_PROMPT,
            "system_prompt_sha256": spec.system_prompt_sha256(),
            "renderer": spec.PROMPT_RENDERER,
            "thinking_enabled": False,
            "option_token_grammar": True,
        },
        "converter": {
            "script": converter.name,
            "sha256": sha256_file(converter),
        },
    }


def main() -> None:
    import argparse
    import shutil
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the .gguf file to write")
    parser.add_argument("--work-dir", type=Path, help="keep the merged HF model here")
    args = parser.parse_args()

    for label, path in (
        ("base model", args.base_model), ("adapter", args.adapter),
        ("dataset dir", args.dataset_dir), ("train report", args.train_report),
        ("converter", args.converter),
    ):  # fmt: skip
        if not path.exists():
            die(f"{label} not found: {path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="merged-", dir=args.out.parent))
    try:
        merge_adapter(args.base_model, args.adapter, merged_dir)
        convert_to_gguf(args.converter, merged_dir, args.out)
        manifest = build_manifest(
            args.out, args.base_model, args.adapter, args.dataset_dir,
            args.train_report, args.converter,
        )  # fmt: skip
        manifest_path = args.out.parent / "specialist.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        if args.work_dir is None:
            shutil.rmtree(merged_dir, ignore_errors=True)
    print(json.dumps({
        "artifact": str(args.out),
        "sha256": manifest["artifact"]["sha256"],
        "bytes": manifest["artifact"]["bytes"],
        "manifest": str(manifest_path),
        "version": manifest["version"],
    }))  # fmt: skip


if __name__ == "__main__":
    main()
