# Tev-style specialist pipeline

This directory builds the primary Local JEV classifier:

    local-jev-tev-specialist-v1
    family: tev-style-specialist
    kind:   specialized-option-token-classifier

Its single job is

    state + question + fixed options -> one option id (or the explicit ABSTAIN)

under a GBNF option-token grammar with thinking disabled. It is a genuinely
fine-tuned artifact, never a renamed generic model: the runtime ties the
declared identity to the SHA-256 of the file a worker actually loads, and a
mismatch can never count as a specialist answer (see
`src/decision_router/specialist.py`).

## Files

- `generate_dataset.py` — deterministic synthetic dataset for the decision
  function (5 template families, labels by construction, slot-key split rule,
  exact + near-duplicate dedup). Writes `train/validation/test.jsonl` plus
  `dataset.manifest.json` with every digest.
- `train_lora.py` — LoRA fine-tune (r=16, alpha=32, bf16) with greedy
  exact-match selection on the VALIDATION split only. `test.jsonl` and the
  approved human-written held-out cases in `decision_router/benchmark.py` are
  never read during training or selection.
- `export_gguf.py` — merges the selected adapter into the base model, converts
  it to GGUF, hashes the result and writes the provenance manifest the runtime
  verifies (`specialist.manifest.json`).

Runs inside a dedicated training virtualenv (torch/transformers/peft); the
runtime virtualenv never needs any of it.

## Provenance guarantees

- Labels are generated from template slots by each family's rule; they never
  come from model output, evaluation results or any EVOLVE outcome.
- The generator is deterministic: the same seed and generator produce
  byte-identical splits (the digests in `dataset.manifest.json` are the proof).
- `dataset.manifest.json` pins `runtime_contract.system_prompt_sha256`; the
  export manifest must carry the same contract and the runtime refuses a
  manifest whose thinking flag or grammar flag disagree.
- The manifest records: base model + digest, adapter digest, dataset file
  digests, training report, quantization, and the exact GGUF SHA-256. That
  digest is what the runtime compares against; nothing else can claim the
  specialist identity.

## Pipeline (the exact commands)

    # 1. dataset (deterministic)
    python specialist/generate_dataset.py --out-dir <data>/tev-specialist-v1/dataset

    # 2. training venv (once)
    python3 -m venv <data>/tev-train-venv
    <data>/tev-train-venv/bin/pip install torch transformers peft accelerate gguf

    # 3. fine-tune + select on validation
    <data>/tev-train-venv/bin/python specialist/train_lora.py \
        --dataset-dir <data>/tev-specialist-v1/dataset \
        --base-model <data>/tev-specialist-v1/base/Qwen3-0.6B \
        --out-dir <data>/tev-specialist-v1/train

    # 4. merge + convert + manifest
    <data>/tev-train-venv/bin/python specialist/export_gguf.py \
        --base-model <data>/tev-specialist-v1/base/Qwen3-0.6B \
        --adapter <data>/tev-specialist-v1/train/adapter-best \
        --dataset-dir <data>/tev-specialist-v1/dataset \
        --train-report <data>/tev-specialist-v1/train/train_report.json \
        --converter /path/to/llama.cpp/convert_hf_to_gguf.py \
        --out <data>/tev-specialist-v1/artifact/local-jev-tev-specialist-v1.gguf

## Registering the artifact

Tier 1 gets a `classifier` block; generic models stay escalation tiers:

    "tiers": {
      "1": {
        "model_path": ".../local-jev-tev-specialist-v1.gguf",
        "model_sha256": "<from specialist.manifest.json>",
        "classifier": {
          "id": "local-jev-tev-specialist-v1",
          "family": "tev-style-specialist",
          "kind": "specialized-option-token-classifier",
          "version": "<artifact version>",
          "thinking": false,
          "option_token_grammar": true,
          "manifest": "artifact/specialist.manifest.json",
          "on_identity_mismatch": "fail_closed"
        }
      },
      "2": { "label": "tier2", "...": "generic escalation model" }
    }

`decision doctor` prints the discovery status. Until a verified artifact is
registered, discovery reports `available: false` — that is the honest state and
a generic tier-1 answer is never labelled a Tev-specialist result.

## Benchmarks

`decision benchmark specialist` measures the cold specialist path (fresh worker
per call), the warm path (persistent worker), the generic baseline and the real
escalation path: n, mean, p50/p95/p99, min, max, valid-output rate, abstention
rate and escalation rate. None of these are calibrated probabilities and no
latency target is an acceptance criterion.
