"""Dataset-provenance schema shared by generation, export and verification.

Two distinct truths are recorded and never conflated:

- ``generation_source_sha256`` — the generator source that produced the split
  files being described. For a fresh run that is the running script itself; a
  historical dataset keeps the digest of the (possibly superseded) source that
  actually generated it, even when that exact revision is no longer available.
- ``current_reproducer_sha256`` — the committed generator source that is known
  to regenerate those split files byte-for-byte (same seed, same digests).

Where the two digests differ, the dataset predates an output-preserving edit of
its generator; that history is recorded, never papered over. The split file
digests themselves remain the ground truth: a verifier re-hashes (or
regenerates) the split files and compares them against the manifest.

Standard library only: this module must import in the training virtualenv, the
runtime virtualenv and a bare checkout alike.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any

GENERATION_SOURCE_FIELD = "generation_source_sha256"
CURRENT_REPRODUCER_FIELD = "current_reproducer_sha256"
SHA256_HEX = 64

TOOL_PACKAGES = ("torch", "transformers", "peft", "accelerate", "gguf")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """The digest of a file's raw bytes (never loads more than one chunk)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX
        and all(c in "0123456789abcdef" for c in value)
    )


def fields(generator_sha256: str) -> dict[str, str]:
    """The provenance block a generator writes for files it just produced.

    For a fresh run the generating source and the reproducing source are the
    same script, so both digests are equal by construction.
    """
    return {
        GENERATION_SOURCE_FIELD: generator_sha256,
        CURRENT_REPRODUCER_FIELD: generator_sha256,
    }


def dataset_block(dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    """The provenance fields to carry into a downstream (artifact) manifest."""
    return {
        GENERATION_SOURCE_FIELD: dataset_manifest.get(GENERATION_SOURCE_FIELD),
        CURRENT_REPRODUCER_FIELD: dataset_manifest.get(CURRENT_REPRODUCER_FIELD),
    }


def check(
    dataset_manifest: dict[str, Any],
    *,
    generator: Path | None = None,
    split_dir: Path | None = None,
) -> list[str]:
    """Problems with the recorded provenance relationship (empty = consistent).

    - when ``generator`` is given, the recorded reproducer must be that source;
    - the recorded generation source must be a well-formed digest;
    - when ``split_dir`` is given, every split file must hash to its recorded
      digest (this also catches a manifest edited without its data).

    The two provenance digests are allowed to differ: that is how a historical
    dataset honestly predates an output-preserving edit of its generator.
    """
    problems: list[str] = []
    source = dataset_manifest.get(GENERATION_SOURCE_FIELD)
    if not is_sha256(source):
        problems.append(f"{GENERATION_SOURCE_FIELD} is missing or malformed")
    reproducer = dataset_manifest.get(CURRENT_REPRODUCER_FIELD)
    if not is_sha256(reproducer):
        problems.append(f"{CURRENT_REPRODUCER_FIELD} is missing or malformed")
    elif generator is not None and sha256_file(generator) != reproducer:
        problems.append(
            f"{CURRENT_REPRODUCER_FIELD} does not match {generator.name}: "
            "the recorded reproducer is not the generator source on disk"
        )
    if split_dir is not None:
        for name, split in sorted((dataset_manifest.get("splits") or {}).items()):
            file_name = (split or {}).get("file") or f"{name}.jsonl"
            path = Path(split_dir) / file_name
            digest = (split or {}).get("digest")
            if not path.is_file():
                problems.append(f"{name}: split file {file_name} is missing")
            elif not is_sha256(digest):
                problems.append(f"{name}: recorded split digest is missing or malformed")
            elif sha256_file(path) != digest:
                problems.append(f"{name}: {file_name} does not match its recorded digest")
    return problems


def tool_versions() -> dict[str, str | None]:
    """The exact tool versions of THIS environment, recorded for provenance."""
    versions: dict[str, str | None] = {}
    for package in TOOL_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions
