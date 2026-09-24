"""TypeSafe JEV provider over the direct systemone API.

Same path as the existing JEV install: POST {model, state, questions} with a
bearer key read from ~/.config/jev/typesafe.key (or TYPESAFE_API_KEY). The key
lives only in the request header; it is never returned, logged or persisted.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..contract import CALIBRATED, DecisionRequest, DecisionResult
from ..sanitize import short_error

NAME = "jev"

# (url, headers, body, timeout_s) -> (status, text); status 0 means no response.
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, str]]


def load_key(key_file: str | None) -> str:
    """The TypeSafe key: the key file wins over the environment, as in the JEV install."""
    if key_file:
        try:
            key = Path(key_file).expanduser().read_text().strip()
            if key:
                return key
        except OSError:
            pass
    return os.environ.get("TYPESAFE_API_KEY", "").strip()


def urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(1_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read(2_000).decode("utf-8", "replace")


class JevProvider:
    name = NAME

    def __init__(
        self,
        endpoint: str,
        model: str = "jev-latest",
        key_file: str | None = "~/.config/jev/typesafe.key",
        timeout_s: float = 20,
        transport: Transport | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.key_file = key_file
        self.timeout_s = float(timeout_s)
        self.transport = transport or urllib_transport

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> JevProvider:
        jev = config["jev"]
        return cls(
            endpoint=jev["endpoint"],
            model=jev["model"],
            key_file=jev.get("key_file"),
            timeout_s=jev.get("timeout_s", 20),
            **kwargs,
        )

    def secret_literals(self) -> tuple[str, ...]:
        key = load_key(self.key_file)
        return (key,) if key else ()

    def available(self) -> tuple[bool, str]:
        if urlparse(self.endpoint).scheme != "https":
            return False, "endpoint_not_https"
        if not load_key(self.key_file):
            return False, "missing_key"
        return True, "ok"

    def build_body(self, request: DecisionRequest) -> dict[str, Any]:
        payload = request.payload()
        return {
            "model": self.model,
            "state": payload["state"],
            "questions": {
                payload["question_id"]: {
                    "type": "choice",
                    "instructions": payload["question"],
                    "criteria": payload["options"],
                }
            },
        }

    def decide(self, request: DecisionRequest) -> DecisionResult:
        started = time.monotonic()

        def elapsed() -> int:
            return round((time.monotonic() - started) * 1000)

        details = {"model_alias": self.model}
        if urlparse(self.endpoint).scheme != "https":
            return DecisionResult.failure(
                NAME, "JEV endpoint must use https", "jev_misconfigured", details=details
            )
        key = load_key(self.key_file)
        if not key:
            return DecisionResult.failure(
                NAME, "TypeSafe key is missing", "jev_unavailable", details=details
            )

        body = json.dumps(self.build_body(request)).encode("utf-8")
        headers = {
            "authorization": f"Bearer {key}",
            "content-type": "application/json",
        }
        try:
            status, text = self.transport(self.endpoint, headers, body, self.timeout_s)
        except TimeoutError:
            return DecisionResult.failure(
                NAME, "timeout", "jev_timeout", latency_ms=elapsed(), details=details
            )
        except Exception as error:  # noqa: BLE001 - network errors must never escape
            reason = getattr(error, "reason", error)
            if isinstance(reason, TimeoutError):
                return DecisionResult.failure(
                    NAME, "timeout", "jev_timeout", latency_ms=elapsed(), details=details
                )
            return DecisionResult.failure(
                NAME,
                short_error(reason, (key,)),
                "jev_unavailable",
                latency_ms=elapsed(),
                details=details,
            )

        latency = elapsed()
        if status != 200:
            return DecisionResult.failure(
                NAME,
                f"HTTP {status}: {short_error(text, (key,), 160)}",
                "jev_http_error",
                latency_ms=latency,
                details=details,
            )
        return self.parse(request, text, latency)

    def parse(self, request: DecisionRequest, text: str, latency: int) -> DecisionResult:
        details: dict[str, Any] = {"model_alias": self.model}

        def malformed(why: str) -> DecisionResult:
            return DecisionResult.failure(
                NAME,
                f"malformed JEV response: {why}",
                "jev_malformed_response",
                latency_ms=latency,
                details=details,
            )

        try:
            data = json.loads(text)
        except ValueError:
            return malformed("not JSON")
        if not isinstance(data, dict):
            return malformed("not an object")
        model = data.get("model") if isinstance(data.get("model"), str) else None
        usage = data.get("usage")
        if isinstance(usage, dict):
            details["usage"] = {k: v for k, v in usage.items() if isinstance(v, (int, float))}
        answers = data.get("answers")
        if not isinstance(answers, dict):
            return malformed("missing answers")
        answer = answers.get(request.question_id)
        if not isinstance(answer, dict):
            return malformed("missing answer for the question")

        allowed = set(request.allowed())
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in allowed:
            return malformed("choice outside the allowed set")

        confidence = answer.get("confidence")
        if not _is_probability(confidence):
            return malformed("confidence is not a probability")

        distribution: dict[str, float] | None = None
        raw = answer.get("probabilities")
        if raw is not None:
            if not isinstance(raw, dict) or not all(
                k in allowed and _is_probability(v) for k, v in raw.items()
            ):
                return malformed("probabilities outside the allowed set")
            distribution = {k: float(v) for k, v in raw.items()}

        abstain = choice == "ABSTAIN"
        return DecisionResult(
            provider=NAME,
            choice=None if abstain else choice,
            abstain=abstain,
            model=model,
            latency_ms=latency,
            confidence=float(confidence),
            confidence_kind=CALIBRATED,
            distribution=distribution,
            details=details,
        )


def _is_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )
