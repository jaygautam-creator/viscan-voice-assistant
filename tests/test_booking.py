"""Bonus: scan-booking agent. Checks synthetic slots and never books without an explicit yes."""

from datetime import date, datetime

import pytest
from conftest import FIXED_NOW, Conversation, FakeLLM, make_client

from app.features.scheduling import is_affirmative, parse_when

HDR = {"X-Clinician-Id": "dr-anand"}


def _bookings(client, who="dr-anand"):
    return client.get("/api/bookings", headers={"X-Clinician-Id": who}).json()


def test_full_booking_flow_proposes_then_confirms(client):
    c = Conversation(client)
    r = c.say("Book a CBCT for Priya Sharma tomorrow morning")
    assert r["action"] == "book_scan" and r["status"] == "needs_clarification"
    assert r["slots"], "should offer real free slots"
    for s in r["slots"]:
        start = datetime.fromisoformat(s["start"])
        assert s["modality"] == "CBCT" and s["machine_id"].startswith("BLR-CBCT")
        assert start.date() == date(2026, 10, 23) and 9 <= start.hour < 12  # tomorrow morning
    assert _bookings(client) == []

    r = c.say("the first one")
    assert r["status"] == "needs_confirmation"
    assert "Please confirm" in r["message"] and r["booking"] is None
    assert _bookings(client) == []  # still nothing booked

    chosen = r["slots"][0]["slot_id"]
    r = c.say("yes")
    assert r["status"] == "ok" and r["booking"]["slot_id"] == chosen
    assert r["booking"]["patient"]["patient_id"] == "P-1001" and r["booking"]["modality"] == "CBCT"
    assert [b["slot_id"] for b in _bookings(client)] == [chosen]
    # the slot is no longer offered
    free = client.get("/api/slots", params={"modality": "CBCT", "day": "2026-10-23"}, headers=HDR).json()
    assert chosen not in {s["slot_id"] for s in free}


@pytest.mark.parametrize("answer", ["no", "cancel", "hmm, maybe later", "yes but not that one"])
def test_nothing_is_booked_without_explicit_yes(client, answer):
    c = Conversation(client)
    c.say("Book an OPG for Arjun Mehta on Monday")
    c.say("the first one")
    r = c.say(answer)
    assert r["booking"] is None
    assert _bookings(client) == []


def test_llm_saying_yes_is_not_enough(settings):
    """Even if the router claims the clinician answered yes, code checks the clinician's own words."""
    llm = FakeLLM(
        tool_calls=[
            (
                "book_scan",
                {
                    "patient_reference": "named",
                    "patient_name": "Priya Sharma",
                    "modality": "OPG",
                    "when_text": "Monday",
                },
            ),
            ("resolve_pending_question", {"answer_text": "the first one"}),
            ("resolve_pending_question", {"answer_text": "yes"}),  # model over-reads an unclear reply
        ]
    )
    client = make_client(settings, llm)
    c = Conversation(client)
    c.say("Book an OPG for Priya Sharma on Monday")
    c.say("the first one")
    r = c.say("hmm, let me think about it")
    assert r["status"] == "needs_confirmation" and r["booking"] is None
    assert _bookings(client) == []


def test_recommended_scan_from_report_with_evidence(convo):
    convo.say("Open Priya Sharma's latest OPG")
    r = convo.say("Book the scan recommended in her report")
    assert r["action"] == "book_scan"
    assert all(s["modality"] == "CBCT" for s in r["slots"])  # R1: "CBCT ... advised prior to surgical removal of 48"
    assert [(s["report_id"], s["sentence_id"]) for s in r["sources"]] == [("RPT-1001-02", "R1")]


def test_missing_scan_type_is_asked_for(convo):
    r = convo.say("Book a scan for Arjun Mehta tomorrow")  # Arjun's latest report recommends no imaging
    assert r["status"] == "needs_clarification" and r["slots"] == []
    r = convo.say("IOPA")
    assert r["slots"] and all(s["modality"] == "IOPA" for s in r["slots"])


def test_exact_time_goes_straight_to_confirmation(client):
    free = client.get("/api/slots", params={"modality": "OPG", "day": "2026-10-26"}, headers=HDR).json()
    slot = free[0]
    t = datetime.fromisoformat(slot["start"])
    c = Conversation(client)
    r = c.say(f"Book an OPG for Kavya Iyer on 26 October at {t:%H:%M}")
    assert r["status"] == "needs_confirmation" and r["slots"][0]["slot_id"] == slot["slot_id"]


def test_slot_taken_before_confirmation_offers_alternatives(client):
    a, b = Conversation(client), Conversation(client)
    a.say("Book a CBCT for Priya Sharma tomorrow morning")
    ra = a.say("the first one")
    b.say("Book a CBCT for Priya Shah tomorrow morning")
    rb = b.say("the first one")
    assert ra["slots"][0]["slot_id"] == rb["slots"][0]["slot_id"]
    assert b.say("yes")["status"] == "ok"
    r = a.say("yes")  # same slot, now taken
    assert r["booking"] is None and r["status"] == "needs_clarification"
    assert "taken" in r["message"] and r["slots"]


def test_ambiguous_patient_is_resolved_before_booking(convo):
    r = convo.say("Book a CBCT for Priya tomorrow")
    assert r["status"] == "needs_clarification" and r["candidates"]
    r = convo.say("Priya Shah")
    assert r["action"] == "select_candidate" and r["slots"]


def test_receptionist_can_book_but_only_own_clinic(client):
    desk = Conversation(client, "front-desk-blr")
    r = desk.say("Book an OPG for Rahul Verma next week")
    assert r["slots"]
    r = Conversation(client, "front-desk-blr").say("Book an OPG for Priya Nair next week")
    assert r["status"] == "not_found" and r["slots"] == []


def test_non_imaging_appointments_are_declined(convo):
    r = convo.say("Book Priya Sharma for a cleaning on Friday")
    assert r["status"] == "unsupported" and r["slots"] == []


def test_parse_when_and_affirmation():
    today = FIXED_NOW.date()  # Thursday 22 Oct 2026
    w = parse_when("tomorrow morning", today)
    assert w.date_from == date(2026, 10, 23) and w.after.hour == 9 and w.before.hour == 12
    w = parse_when("Friday at 10 am", today)
    assert w.date_from == date(2026, 10, 23) and w.at.hour == 10
    w = parse_when("next week after 2 pm", today)
    assert w.date_from == date(2026, 10, 26) and w.date_to == date(2026, 10, 31) and w.after.hour == 14
    assert parse_when("29/10 at 3", today).at.hour == 15
    assert is_affirmative("Yes, book it") and is_affirmative("go ahead")
    assert not is_affirmative("yes but not that one") and not is_affirmative("no") and not is_affirmative("hmm")
