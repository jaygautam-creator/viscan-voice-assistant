"""Feature C (+ bonus Q&A): context-dependent follow-ups with evidence."""

import json
from pathlib import Path

from conftest import Conversation, FakeLLM, make_client

REPORTS = {
    r["report_id"]: r
    for r in json.loads((Path(__file__).resolve().parent.parent / "data" / "reports.json").read_text())
}


def _sentence_text(report_id: str, sid: str) -> str:
    r = REPORTS[report_id]
    return next(s["text"] for sec in ("findings", "impression", "recommendations") for s in r[sec] if s["id"] == sid)


def test_summarize_her_report_uses_conversation_context(convo):
    convo.say("Open Priya's latest OPG report")
    convo.say("Priya Sharma")
    r = convo.say("Summarize her report")
    assert r["status"] == "ok" and r["action"] == "summarize_report"
    assert r["summary"]["report_id"] == "RPT-1001-02"  # the report in context, not another one
    assert r["sources"], "every answer must show its evidence"
    for s in r["sources"]:
        assert s["report_id"] == "RPT-1001-02"
        assert s["text"] == _sentence_text("RPT-1001-02", s["sentence_id"])
    cited = {i for b in r["summary"]["bullets"] for i in b["sentence_ids"]}
    assert cited == {s["sentence_id"] for s in r["sources"]}


def test_followup_without_context_asks_for_patient(convo):
    r = convo.say("Summarize her report")
    assert r["status"] == "needs_clarification" and r["summary"] is None
    r = convo.say("Riya Sharma")
    assert r["status"] == "ok" and r["summary"]["report_id"] == "RPT-1003-02"


def test_pronoun_that_contradicts_context_is_not_resolved(convo):
    convo.say("Open Arjun Mehta's latest OPG")
    r = convo.say("Summarize her report")
    assert r["status"] == "needs_clarification" and r["summary"] is None


def test_grounded_question_answer(convo):
    convo.say("Open Priya Sharma's latest OPG")
    r = convo.say("Is the wisdom tooth close to the nerve?")
    assert r["answer"]["answerable"] is True
    assert "F2" in r["answer"]["sentence_ids"]
    assert "inferior alveolar canal" in _sentence_text("RPT-1001-02", "F2")


def test_question_not_covered_by_report(convo):
    convo.say("Open Priya Sharma's latest OPG")
    r = convo.say("Is there a root fracture?")
    assert r["answer"]["answerable"] is False
    assert r["sources"] == []
    assert "does not mention" in r["answer"]["text"]


def test_llm_summary_claims_must_be_grounded(settings):
    llm = FakeLLM(
        tool_calls=[
            (
                "open_report",
                {
                    "patient_reference": "named",
                    "patient_name": "Priya Sharma",
                    "modality": "OPG",
                    "which": "latest",
                    "report_date": None,
                },
            ),
            (
                "summarize_report",
                {
                    "patient_reference": "context",
                    "patient_name": None,
                    "modality": "ANY",
                    "which": "latest",
                    "report_date": None,
                },
            ),
        ],
        json_outputs=[
            {
                "bullets": [
                    {
                        "text": "Tooth 48 is horizontally impacted close to the inferior alveolar canal.",
                        "sentence_ids": ["I1"],
                    },
                    {"text": "Pulpal involvement of 47.", "sentence_ids": ["I2"]},  # drops "cannot be excluded"
                    {"text": "Extraction of 38 is required.", "sentence_ids": ["R1"]},  # invented tooth
                    {"text": "Mild bone loss.", "sentence_ids": ["F99"]},  # unknown source id
                ]
            }
        ],
    )
    c = Conversation(make_client(settings, llm))
    c.say("Open Priya Sharma's latest OPG")
    r = c.say("Summarize her report")
    s = r["summary"]
    assert [b["text"] for b in s["bullets"]] == [
        "Tooth 48 is horizontally impacted close to the inferior alveolar canal."
    ]
    reasons = {d["text"]: d["reason"] for d in s["dropped"]}
    assert "uncertainty" in reasons["Pulpal involvement of 47."]
    assert "38" in reasons["Extraction of 38 is required."]
    assert "unknown sentence" in reasons["Mild bone loss."]


def test_session_cannot_be_reused_by_another_clinician(client):
    anand = Conversation(client, "dr-anand")
    anand.say("Open Priya Sharma's latest OPG")
    r = client.post(
        "/api/chat",
        json={"text": "Summarize her report", "session_id": anand.session_id},
        headers={"X-Clinician-Id": "dr-kapoor"},
    )
    assert r.status_code == 403
