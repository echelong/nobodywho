"""Reads the model name and quantization from a GGUF header without loading it."""

from __future__ import annotations

import mmap
import re
import struct
from pathlib import Path
from typing import Any

# llama.cpp `llama_ftype` values for `general.file_type`.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}  # fmt: skip

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d"}  # fmt: skip
_STRING, _ARRAY = 8, 9
_WANTED = ("general.name", "general.file_type", "general.size_label", "general.architecture")
_QUANT_IN_NAME = re.compile(r"(?i)(?:^|[-_.])((?:I?Q\d(?:_[A-Z0-9]+)*)|F16|F32|BF16)(?=\.gguf$)")


class _Reader:
    """Sequential little-endian reads over a memory-mapped GGUF header."""

    def __init__(self, data: mmap.mmap) -> None:
        self.data, self.pos = data, 0

    def read(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise ValueError("truncated GGUF header")
        value = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def string(self) -> str:
        length = self.read("<Q")
        if length > 1 << 24 or self.pos + length > len(self.data):
            raise ValueError("implausible GGUF string length")
        text = self.data[self.pos : self.pos + length].decode("utf-8", "replace")
        self.pos += length
        return text

    def skip_strings(self, count: int) -> None:
        """Skips a string array (a tokenizer vocabulary) without decoding it."""
        data, pos, unpack, end = self.data, self.pos, struct.unpack_from, len(self.data)
        for _ in range(count):
            if pos + 8 > end:
                raise ValueError("truncated GGUF header")
            pos += 8 + unpack("<Q", data, pos)[0]
        if pos > end:
            raise ValueError("truncated GGUF header")
        self.pos = pos

    def value(self, kind: int) -> Any:
        if kind in _SCALARS:
            return self.read(_SCALARS[kind])
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item, count = self.read("<I"), self.read("<Q")
            if item in _SCALARS:
                self.pos += struct.calcsize(_SCALARS[item]) * count
            elif item == _STRING:
                self.skip_strings(count)
            else:
                for _ in range(count):
                    self.value(item)
            return None
        raise ValueError(f"unknown GGUF value type {kind}")


def read_metadata(path: str | Path) -> dict[str, Any]:
    """`general.*` identity fields and the block count; stops once all are found."""
    found: dict[str, Any] = {}
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        r = _Reader(data)
        if data[:4] != b"GGUF":
            raise ValueError("not a GGUF file")
        r.pos = 4
        version = r.read("<I")
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        r.read("<Q")  # tensor count
        for _ in range(r.read("<Q")):
            key = r.string()
            value = r.value(r.read("<I"))
            if key in _WANTED or key.endswith(".block_count"):
                found[key] = value
                if all(k in found for k in _WANTED) and any(
                    k.endswith(".block_count") for k in found
                ):
                    break
    return found


def describe(path: str | Path) -> dict[str, Any]:
    """Model identity for receipts: file name, declared name and quantization."""
    path = Path(path)
    info: dict[str, Any] = {"file": path.name}
    try:
        meta = read_metadata(path)
    except (OSError, ValueError):
        meta = {}
    if meta.get("general.name"):
        info["name"] = meta["general.name"]
    if meta.get("general.size_label"):
        info["size"] = meta["general.size_label"]
    if meta.get("general.architecture"):
        info["architecture"] = meta["general.architecture"]
    blocks = next((v for k, v in meta.items() if k.endswith(".block_count")), None)
    if isinstance(blocks, int) and blocks > 0:
        info["layers"] = blocks + 1  # llama.cpp offloads the output layer as one more
    quant = FILE_TYPES.get(meta.get("general.file_type", -1))
    if quant is None:
        match = _QUANT_IN_NAME.search(path.name)
        quant = match.group(1).upper() if match else None
    info["quantization"] = quant
    return info
