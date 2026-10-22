"""Feature A: patient report retrieval (correct report, missing reports, authorisation)."""

from conftest import Conversation, FakeLLM, make_client


def test_opens_correct_latest_opg_for_exact_name(convo):
    r = convo.say("Open Priya Sharma's latest OPG report")
    assert r["status"] == "ok"
    assert r["action"] == "open_report"
    # Latest *OPG* (2026-09-18), not the older OPG and not the newer CBCT (2026-09-25).
    assert r["report"]["report_id"] == "RPT-1001-02"
    assert r["report"]["patient"]["patient_id"] == "P-1001"
    assert r["context"]["active_report_id"] == "RPT-1001-02"


def test_report_selectors(convo):
    assert convo.say("Open Priya Sharma's earliest OPG")["report"]["report_id"] == "RPT-1001-01"
    assert convo.say("Show me Priya Sharma's CBCT")["report"]["report_id"] == "RPT-1001-03"
    assert convo.say("Open Rahul Verma's OPG from June 2024")["report"]["report_id"] == "RPT-1006-01"


def test_spoken_style_input_is_normalised(convo):
    r = convo.say("pull up priya shah's o p g")
    assert r["status"] == "ok"
    assert r["report"]["report_id"] == "RPT-1002-01"


def test_missing_report_is_reported_not_substituted(convo):
    r = convo.say("Open Arjun Mehta's latest CBCT")
    assert r["status"] == "not_found"
    assert r["report"] is None  # never falls back to a different modality or patient
    assert {a["modality"] for a in r["available_reports"]} == {"OPG", "IOPA"}
    assert "no CBCT report" in r["message"]


def test_unknown_patient(convo):
    r = convo.say("Open Zara Khan's latest OPG")
    assert r["status"] == "not_found"
    assert r["report"] is None and r["candidates"] == []


def test_patient_in_other_clinic_is_invisible(client):
    anand = Conversation(client, "dr-anand")
    r = anand.say("Open Priya Nair's OPG")
    assert r["status"] == "not_found" and r["report"] is None
    # Direct API access is also refused without revealing existence.
    assert client.get("/api/reports/RPT-2001-01", headers={"X-Clinician-Id": "dr-anand"}).status_code == 404

    kapoor = Conversation(client, "dr-kapoor")
    r = kapoor.say("Open Priya's OPG")  # only one Priya in Dr. Kapoor's clinic -> unique
    assert r["status"] == "ok" and r["report"]["report_id"] == "RPT-2001-01"


def test_role_without_clinical_access_is_forbidden(client):
    desk = Conversation(client, "front-desk-blr")
    r = desk.say("Open Priya Sharma's OPG")
    assert r["status"] == "forbidden" and r["report"] is None
    assert client.get("/api/reports/RPT-1001-02", headers={"X-Clinician-Id": "front-desk-blr"}).status_code == 403


def test_llm_router_tool_call_is_executed_by_deterministic_tools(settings):
    llm = FakeLLM(
        tool_calls=[
            (
                "open_report",
                {
                    "patient_reference": "named",
                    "patient_name": "Priya Shah",
                    "modality": "IOPA",
                    "which": "latest",
                    "report_date": None,
                },
            ),
        ]
    )
    c = Conversation(make_client(settings, llm))
    r = c.say("Bring up Priya Shah's periapical film")
    assert r["interpreter"].startswith("llm:")
    assert r["report"]["report_id"] == "RPT-1002-02"
    assert [t["step"] for t in r["trace"]][:3] == ["interpret", "resolve_patient", "select_report"]
