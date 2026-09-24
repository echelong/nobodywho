"""Provider-neutral decision request/result contract.

A `DecisionRequest` is one narrow, closed routing question. Every provider
receives the same sanitized `payload()` rendered from it, so receipts from
different providers can be compared on identical input.

Confidence values are deliberately typed by `confidence_kind`: a TypeSafe JEV
`calibrated_probability` and a local `sample_stability` proxy are different
quantities and must never be compared as if they were the same.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

ABSTAIN = "ABSTAIN"
ABSTAIN_DESCRIPTION = "The evidence in the state is not sufficient to choose one option safely."
RISKS = ("low", "medium", "high")

# Confidence kinds. Never relabel one as another.
CALIBRATED = "calibrated_probability"
SAMPLE_STABILITY = "sample_stability"
NO_CONFIDENCE = "none"

CHOICE_ID = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MIN_CHOICES = 2
MAX_CHOICES = 12
MAX_QUESTION_CHARS = 1_000
MAX_CRITERION_CHARS = 500
MAX_STATE_CHARS = 16_000
MAX_METADATA_CHARS = 2_000


class RequestError(ValueError):
    """The request violates the decision contract."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionRequest:
    """One closed routing question with only the evidence needed to answer it."""

    question: str
    choices: tuple[str, ...]
    state: str | dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    question_id: str = "decision"
    criteria: dict[str, str] = field(default_factory=dict)
    allow_abstain: bool = True
    risk: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate(self)

    @classmethod
    def from_dict(cls, data: Any) -> DecisionRequest:
        """Builds a request from JSON input.

        `choices` may be a list of ids, or an object mapping each id to a
        one-line description (merged into `criteria`).
        """
        if not isinstance(data, dict):
            raise RequestError("request must be a JSON object")
        known = {
            "id",
            "question_id",
            "state",
            "question",
            "choices",
            "criteria",
            "allow_abstain",
            "risk",
            "metadata",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise RequestError(f"unknown request fields: {', '.join(unknown)}")

        raw_choices = data.get("choices")
        criteria = dict(data.get("criteria") or {})
        if isinstance(raw_choices, dict):
            for key, value in raw_choices.items():
                if value is not None and key not in criteria:
                    criteria[key] = value
            choices = tuple(raw_choices)
        elif isinstance(raw_choices, list):
            choices = tuple(raw_choices)
        else:
            raise RequestError("choices must be a list or an object")

        kwargs: dict[str, Any] = {
            "question": data.get("question"),
            "choices": choices,
            "state": data.get("state"),
            "criteria": criteria,
            "allow_abstain": data.get("allow_abstain", True),
            "risk": data.get("risk", "medium"),
            "metadata": data.get("metadata") or {},
        }
        if data.get("id") is not None:
            kwargs["id"] = data["id"]
        if data.get("question_id") is not None:
            kwargs["question_id"] = data["question_id"]
        return cls(**kwargs)

    def allowed(self) -> tuple[str, ...]:
        """The ids a provider may answer with, including ABSTAIN when allowed."""
        return self.choices + ((ABSTAIN,) if self.allow_abstain else ())

    def state_hash(self) -> str:
        return sha256_text(_canonical(self.state))

    def payload(self) -> dict[str, Any]:
        """The single provider-neutral payload every provider renders from."""
        options = {c: self.criteria.get(c) for c in self.choices}
        if self.allow_abstain:
            options[ABSTAIN] = ABSTAIN_DESCRIPTION
        return {
            "question_id": self.question_id,
            "state": {"evidence": self.state, "risk": self.risk},
            "question": self.question,
            "options": options,
        }

    def payload_hash(self) -> str:
        return sha256_text(_canonical(self.payload()))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["choices"] = list(self.choices)
        return data


def validate(request: DecisionRequest) -> None:
    if not isinstance(request.id, str) or not REQUEST_ID.match(request.id):
        raise RequestError("id must be 1-128 chars of [A-Za-z0-9._:-]")
    if not isinstance(request.question_id, str) or not CHOICE_ID.match(request.question_id):
        raise RequestError("question_id must be a snake_case identifier")

    if not isinstance(request.question, str) or not request.question.strip():
        raise RequestError("question must be a non-empty string")
    if len(request.question) > MAX_QUESTION_CHARS:
        raise RequestError(f"question exceeds {MAX_QUESTION_CHARS} characters")

    choices = request.choices
    if not isinstance(choices, tuple) or not all(isinstance(c, str) for c in choices):
        raise RequestError("choices must be strings")
    if not MIN_CHOICES <= len(choices) <= MAX_CHOICES:
        raise RequestError(f"a closed question needs {MIN_CHOICES}-{MAX_CHOICES} choices")
    if len(set(choices)) != len(choices):
        raise RequestError("choices must be unique")
    for choice in choices:
        if not CHOICE_ID.match(choice):
            raise RequestError(f"choice {choice!r} must be snake_case: [a-z][a-z0-9_]{{0,47}}")
        if choice.lower() == ABSTAIN.lower():
            raise RequestError("ABSTAIN is reserved; use allow_abstain instead")

    if not isinstance(request.criteria, dict):
        raise RequestError("criteria must be an object")
    for key, value in request.criteria.items():
        if key not in choices:
            raise RequestError(f"criteria key {key!r} is not one of the choices")
        if not isinstance(value, str) or len(value) > MAX_CRITERION_CHARS:
            raise RequestError(
                f"criteria for {key!r} must be a string of at most {MAX_CRITERION_CHARS} characters"
            )

    if not isinstance(request.allow_abstain, bool):
        raise RequestError("allow_abstain must be a boolean")
    if request.risk not in RISKS:
        raise RequestError(f"risk must be one of {', '.join(RISKS)}")

    if isinstance(request.state, str):
        if not request.state.strip():
            raise RequestError("state must not be empty")
    elif not isinstance(request.state, dict) or not request.state:
        raise RequestError("state must be a non-empty string or object")
    try:
        state_size = len(_canonical(request.state))
    except (TypeError, ValueError) as error:
        raise RequestError("state must be JSON-serialisable") from error
    if state_size > MAX_STATE_CHARS:
        raise RequestError(
            f"state is {state_size} characters; keep it under {MAX_STATE_CHARS} "
            "(compact evidence only)"
        )

    if not isinstance(request.metadata, dict):
        raise RequestError("metadata must be an object")
    try:
        metadata_size = len(_canonical(request.metadata))
    except (TypeError, ValueError) as error:
        raise RequestError("metadata must be JSON-serialisable") from error
    if metadata_size > MAX_METADATA_CHARS:
        raise RequestError(f"metadata exceeds {MAX_METADATA_CHARS} characters")


@dataclass
class DecisionResult:
    """A normalized answer from one provider (or the router itself)."""

    provider: str
    choice: str | None = None
    abstain: bool = False
    model: str | None = None
    latency_ms: int = 0
    confidence: float | None = None
    confidence_kind: str = NO_CONFIDENCE
    distribution: dict[str, float] | None = None
    votes: dict[str, int] | None = None
    samples: int | None = None
    fallback_reason: str | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the provider produced a usable choice or an explicit abstention."""
        return self.error is None and (self.choice is not None or self.abstain)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def failure(
        cls, provider: str, error: str, reason: str, *, latency_ms: int = 0, **extra
    ) -> DecisionResult:
        return cls(
            provider=provider,
            error=error,
            fallback_reason=reason,
            latency_ms=latency_ms,
            **extra,
        )
