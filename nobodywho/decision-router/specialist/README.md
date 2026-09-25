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
- `provenance.py` — the shared provenance schema (generation source vs
  reproducer, split digests, tool versions) used by the generator, the export
  and the verifier, and covered by `tests/test_provenance.py`.
- `train_lora.py` — LoRA fine-tune (r=16, alpha=32, bf16) with greedy
  exact-match selection on the VALIDATION split only. `test.jsonl` and the
  approved human-written held-out cases in `decision_router/benchmark.py` are
  never read during training or selection.
- `export_gguf.py` — merges the selected adapter into the base model, converts
  it to GGUF, hashes the result and writes the provenance manifest the runtime
  verifies (`specialist.manifest.json`).
- `eval_test_split.py` — held-out evaluation of the frozen selection on
  `test.jsonl` (run once, after selection; never feeds back).
- `register_tier1_specialist.py` — installs the exported artifact as tier 1
  (atomic config write, timestamped backup, digest-checked).
- `verify_artifact.py` — verifies one installation end to end: artifact digest
  and size, identity and prompt contract, dataset provenance (including
  byte-identical regeneration), and the live config pin.
- `requirements-train.txt` — the exact tool versions of the recorded training
  run (the runtime virtualenv needs none of them).

Runs inside a dedicated training virtualenv (torch/transformers/peft); the
runtime virtualenv never needs any of it.

## Provenance guarantees

- Labels are generated from template slots by each family's rule; they never
  come from model output, evaluation results or any EVOLVE outcome.
- The generator is deterministic: the same seed and generator produce
  byte-identical splits (the digests in `dataset.manifest.json` are the proof;
  `tests/test_provenance.py` seals them).
- Provenance fields never conflate two truths: `generation_source_sha256` is
  the generator source that produced the split files being described (kept
  even when that revision is later superseded), while
  `current_reproducer_sha256` is the committed `generate_dataset.py` that
  regenerates them byte-for-byte. Where they differ the dataset predates an
  output-preserving edit of the generator; that history is recorded in the
  manifest's `notes`, never papered over.
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

    # 2. training venv (once; exact recorded versions)
    python3 -m venv <data>/tev-train-venv
    <data>/tev-train-venv/bin/pip install -r specialist/requirements-train.txt

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

    # 5. held-out evaluation of the frozen selection (once, after selection)
    <data>/tev-train-venv/bin/python specialist/eval_test_split.py \
        <data>/tev-specialist-v1/dataset \
        <data>/tev-specialist-v1/base/Qwen3-0.6B \
        <data>/tev-specialist-v1/train/adapter-best \
        <data>/tev-specialist-v1/test_eval.json

    # 6. install + verify
    python specialist/register_tier1_specialist.py \
        <data>/tev-specialist-v1/artifact/local-jev-tev-specialist-v1.gguf \
        <data>/tev-specialist-v1/artifact/specialist.manifest.json
    python specialist/verify_artifact.py

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

## What 97.98% does and does not mean

`local-jev-tev-specialist-v1` is a constrained option-classification
specialist: state + question + fixed options -> one option id (or the explicit
ABSTAIN), under a GBNF option-token grammar with thinking disabled. It is not
a general reasoning model, and its held-out score must not be read as one.

- 97.98% is exact match on the synthetic held-out split of
  `local-jev-decision-dataset-v1` (194/198, 100% valid option tokens). It is
  reported unchanged: neither lowered nor inflated.
- Much of that synthetic benchmark is easy: a trivial lexical-overlap baseline
  with no training at all reaches 171/198 (86.4%) on the same split, solving
  `root_cause_fix`, `tool_selection` and `minimal_sufficient_action` entirely
  from surface wording.
- `constraint_match` carries most of the meaningful discrimination: it is the
  only family with specialist errors (32/36, all four wrong-proposal picks) and
  the only family where the trivial baseline stays at chance (18/36).
- A harder, naturalistic held-out benchmark (real Local JEV decision traffic
  with adversarially matched options) is future work.

## Reproducibility classification

SEMANTICALLY REPRODUCIBLE (independently reviewed and verified), not
bit-reproducible:

- the dataset is bit-reproducible: the recorded `current_reproducer_sha256`
  regenerates byte-identical split files, and `tests/test_provenance.py` seals
  the digests against future generator changes;
- the trained artifact is reproducible in intent and verifiable by digest (base
  weights, adapter weights, converter script and output GGUF are all SHA-256
  pinned, and tool versions are recorded in `requirements-train.txt` and the
  manifest), but training-time GPU kernels are not bit-deterministic, so a
  re-run may differ in low-order bits while selecting the same epoch;
- `specialist/verify_artifact.py` re-checks the whole chain at any time.
