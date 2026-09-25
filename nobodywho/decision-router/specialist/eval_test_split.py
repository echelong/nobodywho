"""Evaluate the SELECTED adapter on the held-out test split (after selection).

This is deliberately a separate step that runs only once model selection is
frozen: `train_lora.py` never reads test.jsonl, and nothing here feeds back
into selection, training or the artifact. It reports the honest held-out
numbers for the recovery record.

Runs in the training virtualenv (torch/transformers/peft; see
`specialist/requirements-train.txt`). No machine-specific paths: everything is
an explicit argument.

Usage:
    python specialist/eval_test_split.py DATASET_DIR BASE_MODEL ADAPTER OUT.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# torch/peft/transformers exist only in the dedicated training virtualenv
# (specialist/requirements-train.txt), never in the runtime environment; the
# `ty: ignore` markers below document exactly that boundary and nothing else.
import torch  # ty: ignore[unresolved-import]
from peft import PeftModel  # ty: ignore[unresolved-import]
from transformers import AutoModelForCausalLM, AutoTokenizer  # ty: ignore[unresolved-import]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance
import train_lora as training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("base_model", type=Path)
    parser.add_argument("adapter", type=Path, help="the validation-selected adapter directory")
    parser.add_argument("out", type=Path, help="the JSON report to write")
    args = parser.parse_args()

    manifest = json.loads((args.dataset_dir / "dataset.manifest.json").read_text())
    system_prompt = manifest["runtime_contract"]["system_prompt"]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    test_records = training.load_jsonl(args.dataset_dir / "test.jsonl")
    metrics = training.evaluate(model, tokenizer, test_records, system_prompt)
    report = {
        "split": "test",
        "examples": len(test_records),
        "dataset_digest": manifest["splits"]["test"]["digest"],
        "adapter": str(args.adapter),
        "selection_frozen_before_this_run": True,
        "tools": {**provenance.tool_versions(), "python": platform.python_version()},
        "metrics": metrics,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
