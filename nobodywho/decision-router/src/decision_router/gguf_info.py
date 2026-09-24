"""Reads the model name and quantization from a GGUF header without loading it."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, BinaryIO

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


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _string(f: BinaryIO) -> str:
    length = _read(f, "<Q")
    if length > 1 << 24:
        raise ValueError("implausible GGUF string length")
    return f.read(length).decode("utf-8", "replace")


def _value(f: BinaryIO, kind: int) -> Any:
    if kind in _SCALARS:
        return _read(f, _SCALARS[kind])
    if kind == _STRING:
        return _string(f)
    if kind == _ARRAY:
        item, count = _read(f, "<I"), _read(f, "<Q")
        if item in _SCALARS:
            f.seek(struct.calcsize(_SCALARS[item]) * count, 1)
        else:
            for _ in range(count):
                _value(f, item)
        return None
    raise ValueError(f"unknown GGUF value type {kind}")


def read_metadata(path: str | Path) -> dict[str, Any]:
    """`general.*` identity fields; stops as soon as all of them are found."""
    found: dict[str, Any] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = _read(f, "<I")
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        _read(f, "<Q")  # tensor count
        for _ in range(_read(f, "<Q")):
            key = _string(f)
            kind = _read(f, "<I")
            value = _value(f, kind)
            if key in _WANTED:
                found[key] = value
                if len(found) == len(_WANTED):
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
    quant = FILE_TYPES.get(meta.get("general.file_type", -1))
    if quant is None:
        match = _QUANT_IN_NAME.search(path.name)
        quant = match.group(1).upper() if match else None
    info["quantization"] = quant
    return info
