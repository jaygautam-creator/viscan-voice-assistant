"""Ambiguous patient identity: never guess, ask; resume the original request after the answer."""

import pytest
from conftest import Conversation, FakeLLM, make_client


def test_ambiguous_first_name_asks_and_returns_no_report(convo):
    r = convo.say("Open Priya's latest OPG report")
    assert r["status"] == "needs_clarification"
    assert r["report"] is None and r["sources"] == []
    ids = {c["patient_id"] for c in r["candidates"]}
    # Both Priyas in this clinic; not the Priya in another clinic, not the similar-sounding Riya.
    assert ids == {"P-1001", "P-1002"}
    assert r["context"]["pending_question"]


@pytest.mark.parametrize(
    "answer, expected_report",
    [
        ("Priya Shah", "RPT-1002-01"),
        ("Sharma", "RPT-1001-02"),
        ("the second one", "RPT-1001-02"),  # candidates are listed Shah, Sharma
        ("the one born in 1985", "RPT-1002-01"),
        ("P-1001", "RPT-1001-02"),
    ],
)
def test_clarification_answer_resumes_original_request(convo, answer, expected_report):
    convo.say("Open Priya's latest OPG report")
    r = convo.say(answer)
    assert r["status"] == "ok", r["message"]
    assert r["report"]["report_id"] == expected_report
    assert r["report"]["modality"] == "OPG"  # original intent (latest OPG) preserved
    assert r["context"]["pending_question"] is None


def test_shared_surname_is_ambiguous(convo):
    r = convo.say("Open Sharma's OPG")
    assert r["status"] == "needs_clarification"
    assert {c["patient_id"] for c in r["candidates"]} == {"P-1001", "P-1003"}


def test_unclear_answer_reasks(convo):
    convo.say("Open Priya's latest OPG report")
    r = convo.say("hmm the tall one")
    assert r["status"] == "needs_clarification" and r["report"] is None


def test_fuzzy_name_needs_confirmation_and_can_be_declined(convo):
    r = convo.say("Open Pria Sharma's OPG")  # STT-style misspelling
    assert r["status"] == "needs_clarification"
    assert [c["patient_id"] for c in r["candidates"]] == ["P-1001"]
    assert r["report"] is None
    r = convo.say("no")
    assert r["status"] == "ok" and r["report"] is None and "cancelled" in r["message"].lower()

    convo.say("Open Pria Sharma's OPG")
    r = convo.say("yes")
    assert r["report"]["report_id"] == "RPT-1001-02"


def test_llm_cannot_bypass_identity_rules(settings):
    """Even if the LLM proposes a bare first name, the resolver still enforces disambiguation."""
    llm = FakeLLM(
        tool_calls=[
            (
                "open_report",
                {
                    "patient_reference": "named",
                    "patient_name": "Priya",
                    "modality": "OPG",
                    "which": "latest",
                    "report_date": None,
                },
            ),
            ("resolve_pending_question", {"answer_text": "Priya Shah"}),
        ]
    )
    c = Conversation(make_client(settings, llm))
    r = c.say("Open Priya's latest OPG report")
    assert r["status"] == "needs_clarification" and r["report"] is None
    r = c.say("Priya Shah")
    assert r["report"]["report_id"] == "RPT-1002-01"
