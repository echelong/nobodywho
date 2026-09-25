"""LoRA fine-tune of a compact base model into the Tev-style decision specialist.

Runs in the separate training venv (torch/transformers/peft), NOT in the runtime
venv. It reads the generated dataset and its manifest (the manifest's
runtime_contract is the single source of the system prompt) and never reads
`test.jsonl`: the held-out test split and the approved benchmark cases stay
untouched until model selection is complete (see specialist/README.md).

Model selection rule (frozen before any test evaluation): the epoch with the
highest greedy exact-match accuracy on the VALIDATION split wins; ties go to the
earlier epoch. The test split is evaluated only after the winner is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Mapping
from pathlib import Path

# torch/transformers/peft exist only in the dedicated training virtualenv
# (specialist/requirements-train.txt), never in the runtime environment; the
# `ty: ignore` markers below document exactly that boundary and nothing else.
import torch  # ty: ignore[unresolved-import]
import transformers  # ty: ignore[unresolved-import]
from peft import LoraConfig, get_peft_model  # ty: ignore[unresolved-import]
from transformers import AutoModelForCausalLM, AutoTokenizer  # ty: ignore[unresolved-import]

SYSTEM_PROMPT_ASSERTION = (
    "the dataset manifest must pin the system prompt its examples were rendered with"
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flat_ids(value) -> list[int]:
    """Plain input ids from whatever container the tokenizer returns.

    transformers 5.x returns a BatchEncoding (a mapping that is not a dict
    subclass) for tokenized chat templates; older versions return a list.
    """
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def chat_ids(tokenizer, system_prompt: str, user_prompt: str) -> list[int]:
    """The exact chat-template ids the runtime GGUF will see (thinking off)."""
    templated = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return flat_ids(templated)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_texts(records: list[dict], tokenizer, system_prompt: str) -> list[dict]:
    """Chat-templated prompt + completion, with the prompt masked out of the loss."""
    texts = []
    eos = tokenizer.eos_token_id
    for record in records:
        prompt_ids = chat_ids(tokenizer, system_prompt, record["prompt"])
        completion_ids = flat_ids(tokenizer(record["completion"], add_special_tokens=False))
        ids = prompt_ids + completion_ids + [eos]
        labels = [-100] * len(prompt_ids) + completion_ids + [eos]
        # input and labels are truncated at the same index, so they stay aligned.
        texts.append({"input_ids": ids[:1024], "labels": labels[:1024]})
    return texts


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[dict]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        item = self.texts[index]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.ones(len(item["input_ids"]), dtype=torch.long),
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(item["input_ids"]) for item in batch)

    def pad(values: list[int], fill: int) -> list[int]:
        return values + [fill] * (width - len(values))

    return {
        "input_ids": torch.tensor(
            [pad(item["input_ids"].tolist(), pad_id) for item in batch], dtype=torch.long
        ),
        "labels": torch.tensor(
            [pad(item["labels"].tolist(), -100) for item in batch], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [pad(item["attention_mask"].tolist(), 0) for item in batch], dtype=torch.long
        ),
    }


@torch.no_grad()
def evaluate(model, tokenizer, records: list[dict], system_prompt: str) -> dict:
    """Greedy exact-match accuracy on a split; never touches the test split."""
    model.eval()
    correct = valid = abstain_correct = abstain_total = 0
    allowed = None
    for record in records:
        choices = set(record["request"]["choices"]) | {"ABSTAIN"}
        prompt_ids = chat_ids(tokenizer, system_prompt, record["prompt"])
        out = model.generate(
            input_ids=torch.tensor([prompt_ids], device=model.device),
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][len(prompt_ids) :], skip_special_tokens=True).strip()
        prediction = text.splitlines()[0].strip() if text else ""
        if prediction in choices:
            valid += 1
        correct += prediction == record["completion"]
        if record["completion"] == "ABSTAIN":
            abstain_total += 1
            abstain_correct += prediction == "ABSTAIN"
        allowed = choices
    model.train()
    return {
        "examples": len(records),
        "exact_match": correct,
        "exact_match_rate": round(correct / max(1, len(records)), 4),
        "valid_option_token_rate": round(valid / max(1, len(records)), 4),
        "abstain_examples": abstain_total,
        "abstain_exact_match": abstain_correct,
        "allowed_tokens_example": sorted(allowed or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    seed_everything(args.seed)

    manifest = json.loads((args.dataset_dir / "dataset.manifest.json").read_text())
    system_prompt = manifest["runtime_contract"]["system_prompt"]
    assert (
        manifest["runtime_contract"]["system_prompt_sha256"]
        == hashlib.sha256(system_prompt.encode()).hexdigest()
    ), SYSTEM_PROMPT_ASSERTION
    train_records = load_jsonl(args.dataset_dir / "train.jsonl")
    validation_records = load_jsonl(args.dataset_dir / "validation.jsonl")
    # The test split is deliberately NOT loaded anywhere in this script.

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )  # fmt: skip
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_texts = build_texts(train_records, tokenizer, system_prompt)
    loader = torch.utils.data.DataLoader(
        JsonlDataset(train_texts),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.eos_token_id),
    )
    steps = (len(loader) // args.grad_accum + 1) * args.epochs
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(10, steps // 20), num_training_steps=steps
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = {"exact_match_rate": -1.0, "epoch": None}
    step = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        optimizer.zero_grad()
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            total_loss += loss.item() * args.grad_accum
            batches += 1
            if batches % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
        metrics = evaluate(model, tokenizer, validation_records, system_prompt)
        entry = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, batches), 4),
            "validation": metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if metrics["exact_match_rate"] > best["exact_match_rate"]:
            best = {"exact_match_rate": metrics["exact_match_rate"], "epoch": epoch}
            model.save_pretrained(args.out_dir / "adapter-best")
    model.save_pretrained(args.out_dir / "adapter-final")
    tokenizer.save_pretrained(args.out_dir / "adapter-best")

    report = {
        "base_model": {
            "path": str(args.base_model),
            "config_name": model.config._name_or_path,
            "architectures": model.config.architectures,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "seed": args.seed,
            "optimizer": "AdamW",
            "schedule": "cosine with linear warmup",
            "precision": "bfloat16",
            "max_seq_len": 1024,
        },
        "dataset": {
            "dataset_version": manifest["dataset_version"],
            "dataset_digests": {
                name: split["digest"] for name, split in manifest["splits"].items()
            },
            "train_examples": len(train_records),
            "validation_examples": len(validation_records),
            "test_split_used": False,
        },
        "selection_rule": "highest greedy exact-match accuracy on the VALIDATION split; ties go to the "
        "earlier epoch; the test split is never loaded by this script",
        "selection": best,
        "history": history,
        "runtime_seconds": round(time.time() - started, 1),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selection": best, "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
