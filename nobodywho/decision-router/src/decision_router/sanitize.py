"""Secret redaction applied before anything leaves the router or reaches a log."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

REDACTED = "<redacted>"

_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\b(?:sk|rk|pk|vck|tsk)[-_][A-Za-z0-9_-]{12,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"(?i)\b([A-Z0-9_]*(?:API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD))\s*[=:]\s*['\"]?[^\s'\"]{6,}"
    ),
]


def redact(text: str, literals: Iterable[str] = ()) -> str:
    """Replaces known secret literals and secret-looking tokens in `text`."""
    for literal in literals:
        if literal and len(literal) >= 8:
            text = text.replace(literal, REDACTED)
    for pattern in _PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def redact_value(value: Any, literals: Iterable[str] = ()) -> Any:
    """`redact` applied to every string inside a JSON-like value."""
    literals = tuple(literals)
    if isinstance(value, str):
        return redact(value, literals)
    if isinstance(value, dict):
        return {k: redact_value(v, literals) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, literals) for v in value]
    return value


def short_error(text: object, literals: Iterable[str] = (), limit: int = 200) -> str:
    """A single-line, credential-free excerpt of an error for receipts."""
    return re.sub(r"\s+", " ", redact(str(text), literals)).strip()[:limit]
