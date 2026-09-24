from __future__ import annotations

import json
from pathlib import Path

from decision_router.contract import DecisionRequest, DecisionResult
from decision_router.ledger import Ledger
from decision_router.providers.jev import JevProvider
from decision_router.router import Router

FAKE_KEY = "tsk_" + "Q7x9" * 20  # 84 chars, never a real key


def make_request(**overrides) -> DecisionRequest:
    data = {
        "id": "req-1",
        "question_id": "next_step",
        "state": "Two retries failed with the same timezone error.",
        "question": "What should the agent do next?",
        "choices": {
            "retry_same": "Retry unchanged.",
            "inspect_tz": "Inspect timezone handling.",
            "rewrite": "Rewrite the parser.",
        },
        "allow_abstain": True,
        "risk": "low",
    }
    data.update(overrides)
    return DecisionRequest.from_dict(data)


def jev_body(choice: str, confidence: float = 0.9, probabilities=None, model="jev-1.13.0"):
    answer = {"type": "choice", "choice": choice, "confidence": confidence}
    if probabilities is not None:
        answer["probabilities"] = probabilities
    return json.dumps({"model": model, "answers": {"next_step": answer}})


class FakeTransport:
    def __init__(self, status: int = 200, text: str = "", raises: BaseException | None = None):
        self.status, self.text, self.raises = status, text, raises
        self.calls: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body)})
        if self.raises is not None:
            raise self.raises
        return self.status, self.text


class StubProvider:
    """A provider returning a fixed result and recording what it was asked."""

    def __init__(self, name: str, result: DecisionResult | None = None, raises=None):
        self.name = name
        self.result = result
        self.raises = raises
        self.requests: list[DecisionRequest] = []

    def decide(self, request):
        self.requests.append(request)
        if self.raises:
            raise self.raises
        return self.result


def ok(provider: str, choice: str | None, *, abstain=False, kind="calibrated_probability"):
    return DecisionResult(
        provider=provider, choice=choice, abstain=abstain, model=f"{provider}-model",
        confidence=0.8, confidence_kind=kind,
    )  # fmt: skip


def failed(provider: str, reason: str = "down"):
    return DecisionResult.failure(provider, "unavailable", reason)


def make_router(ledger: Ledger, jev=None, local=None) -> Router:
    providers = {}
    if jev is not None:
        providers["jev"] = jev
    if local is not None:
        providers["nobodywho"] = local
    return Router(providers, ledger, (FAKE_KEY,))


def jev_provider(key_file: Path, transport: FakeTransport) -> JevProvider:
    return JevProvider(
        endpoint="https://api.typesafe.ai/v1/systemone",
        key_file=str(key_file),
        timeout_s=2,
        transport=transport,
    )
