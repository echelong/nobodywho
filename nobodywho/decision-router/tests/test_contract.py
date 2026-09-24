from __future__ import annotations

import pytest
from helpers import make_request

from decision_router.contract import ABSTAIN, DecisionRequest, RequestError


def test_valid_request_from_choice_object():
    request = make_request()
    assert request.choices == ("retry_same", "inspect_tz", "rewrite")
    assert request.criteria["inspect_tz"] == "Inspect timezone handling."
    assert request.allowed()[-1] == ABSTAIN


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"question": ""}, "question"),
        ({"question": "x" * 1001}, "question exceeds"),
        ({"state": ""}, "state"),
        ({"state": {}}, "state"),
        ({"state": "x" * 20_000}, "compact evidence"),
        ({"risk": "extreme"}, "risk"),
        ({"allow_abstain": "yes"}, "allow_abstain"),
        ({"metadata": {"blob": "x" * 3000}}, "metadata"),
        ({"id": "has spaces"}, "id"),
        ({"question_id": "Next Step"}, "question_id"),
    ],
)
def test_request_validation(overrides, message):
    with pytest.raises(RequestError, match=message):
        make_request(**overrides)


def test_unknown_fields_rejected():
    with pytest.raises(RequestError, match="unknown request fields"):
        DecisionRequest.from_dict(
            {"question": "q?", "choices": ["a", "b"], "state": "s", "command": "rm -rf /"}
        )


def test_non_object_request_rejected():
    with pytest.raises(RequestError):
        DecisionRequest.from_dict(["not", "an", "object"])


@pytest.mark.parametrize(
    "choices, message",
    [
        (["only_one"], "2-12 choices"),
        ([f"c{i}" for i in range(13)], "2-12 choices"),
        (["same", "same"], "unique"),
        (["Retry Same", "other"], "snake_case"),
        (["retry-same", "other"], "snake_case"),
        (["abstain", "other"], "reserved|snake_case"),
        ("not_a_list", "list or an object"),
    ],
)
def test_closed_choice_validation(choices, message):
    with pytest.raises(RequestError, match=message):
        make_request(choices=choices)


def test_criteria_must_describe_known_choices():
    with pytest.raises(RequestError, match="not one of the choices"):
        make_request(choices=["a", "b"], criteria={"c": "unknown"})


def test_abstain_only_offered_when_allowed():
    closed = make_request(allow_abstain=False)
    assert ABSTAIN not in closed.allowed()
    assert ABSTAIN not in closed.payload()["options"]
    assert ABSTAIN in make_request().payload()["options"]


def test_payload_is_provider_neutral_and_stable():
    a, b = make_request(), make_request()
    assert a.payload() == b.payload()
    assert a.payload_hash() == b.payload_hash()
    assert "metadata" not in a.payload()
    assert a.payload()["state"] == {"evidence": a.state, "risk": "low"}
