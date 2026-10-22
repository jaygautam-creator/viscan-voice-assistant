"""Feature B: dictation -> structured draft. No invented observations; uncertainty labelled."""

from conftest import FakeLLM, make_client

HDR = {"X-Clinician-Id": "dr-anand"}


def _dictate(client, transcript, **kw):
    r = client.post("/api/dictations", json={"transcript": transcript, **kw}, headers=HDR)
    assert r.status_code == 201, r.text
    return r.json()


def _findings_section(draft_text: str) -> str:
    """The clinical body only: header (with its timestamp) and the review list are excluded."""
    return draft_text.split("FINDINGS", 1)[1].split("NEEDS REVIEW")[0]


def test_example_dictation_produces_structured_draft(client):
    d = _dictate(client, "Lower right third molar is horizontally impacted", patient_id="P-1001", modality="OPG")
    assert d["status"] == "pending_clinician_review" and d["review_required"] is True
    assert len(d["findings"]) == 1
    f = d["findings"][0]
    assert f["condition"] == "impaction"
    assert f["site"] == {
        "kind": "tooth",
        "fdi": "48",
        "universal": "32",
        "name": "lower right third molar",
        "candidates": [],
        "source": "dictated",
    }
    assert {"name": "orientation", "value": "horizontally"} in f["attributes"]
    assert f["certainty"] == "definite" and f["presence"] == "present"
    assert f["evidence"] == "Lower right third molar is horizontally impacted"
    assert "PENDING CLINICIAN REVIEW" in d["draft_text"]
    assert "Tooth 48 (lower right third molar; Universal #32): Horizontal impaction." in d["draft_text"]


def test_uncertainty_and_negation_are_preserved(client):
    d = _dictate(client, "Possible periapical radiolucency on tooth three six. No caries on 47.")
    by_tooth = {f["site"]["fdi"]: f for f in d["findings"]}
    assert by_tooth["36"]["certainty"] == "uncertain" and by_tooth["36"]["hedge_terms"] == ["Possible"]
    assert by_tooth["47"]["presence"] == "absent" and by_tooth["47"]["observation"] == "No caries"
    assert "UNCERTAIN" in d["draft_text"]


def test_underspecified_site_is_flagged_not_guessed(client):
    d = _dictate(client, "Lower third molar is impacted.", patient_id="P-1001")
    f = d["findings"][0]
    assert f["site"]["kind"] == "tooth_unresolved" and f["site"]["fdi"] is None
    assert set(f["site"]["candidates"]) == {"38", "48"}
    # Approval is blocked until the clinician resolves the site.
    r = client.post(f"/api/dictations/{d['draft_id']}/approve", headers=HDR)
    assert r.status_code == 409


def test_unmappable_speech_is_surfaced_not_dropped(client):
    d = _dictate(client, "Lower right third molar is horizontally impacted. The upper left canine looks a bit odd.")
    assert d["unparsed_segments"] == ["The upper left canine looks a bit odd"]
    assert len(d["findings"]) == 1


def test_llm_unsupported_clinical_claims_are_rejected(settings):
    transcript = "Lower right third molar is horizontally impacted. Possible periapical radiolucency on 36."
    hallucinating = {
        "findings": [
            # grounded, but with an invented modifier
            {
                "evidence_quote": "Lower right third molar is horizontally impacted",
                "condition": "impaction",
                "presence": "present",
                "certainty": "definite",
                "site_text": "48",
                "attributes": [
                    {"name": "orientation", "value": "horizontally"},
                    {"name": "severity", "value": "severe"},
                ],
            },
            # invented diagnosis on a real quote
            {
                "evidence_quote": "Lower right third molar is horizontally impacted",
                "condition": "pericoronitis",
                "presence": "present",
                "certainty": "definite",
                "site_text": "48",
                "attributes": [],
            },
            # quote that was never dictated
            {
                "evidence_quote": "Deep caries on 47",
                "condition": "caries",
                "presence": "present",
                "certainty": "definite",
                "site_text": "47",
                "attributes": [],
            },
            # tooth that was never dictated
            {
                "evidence_quote": "horizontally impacted",
                "condition": "impaction",
                "presence": "present",
                "certainty": "definite",
                "site_text": "38",
                "attributes": [],
            },
            # hedge dropped by the model ("Possible" -> definite)
            {
                "evidence_quote": "Possible periapical radiolucency on 36",
                "condition": "periapical_lesion",
                "presence": "present",
                "certainty": "definite",
                "site_text": "36",
                "attributes": [],
            },
        ],
        "recommendations": [{"evidence_quote": "Extraction of 48 is advised"}],
        "unmapped_segments": [],
    }
    client = make_client(settings, FakeLLM(json_outputs=[hallucinating]))
    d = _dictate(client, transcript)

    accepted = {(f["condition"], f["site"]["fdi"]) for f in d["findings"]}
    assert accepted == {("impaction", "48"), ("periapical_lesion", "36")}
    pa = next(f for f in d["findings"] if f["condition"] == "periapical_lesion")
    assert pa["certainty"] == "uncertain"
    assert any("downgraded" in flag for flag in pa["review_flags"])
    imp = next(f for f in d["findings"] if f["condition"] == "impaction")
    assert {a["value"] for a in imp["attributes"]} == {"horizontally"}

    reasons = " | ".join(r["reason"] for r in d["rejected_claims"])
    assert "Pericoronitis" in reasons and "not stated" in reasons
    assert "not found in the dictation" in reasons
    assert "Tooth 38 was not dictated" in reasons
    assert "Modifier not present" in reasons
    assert d["recommendations"] == []

    body = _findings_section(d["draft_text"])
    for invented in ("ericoronitis", "caries", "38", "severe", "Extraction"):
        assert invented not in body
    assert any("rejected" in r for r in d["review_reasons"])


def test_llm_omissions_are_caught_by_coverage_check(settings):
    transcript = "Lower right third molar is horizontally impacted. Possible periapical radiolucency on 36."
    partial = {
        "findings": [
            {
                "evidence_quote": "Lower right third molar is horizontally impacted",
                "condition": "impaction",
                "presence": "present",
                "certainty": "definite",
                "site_text": "lower right third molar",
                "attributes": [],
            },
        ],
        "recommendations": [],
        "unmapped_segments": [],
    }
    d = _dictate(make_client(settings, FakeLLM(json_outputs=[partial])), transcript)
    assert d["unparsed_segments"] == ["Possible periapical radiolucency on 36"]


def test_llm_outage_falls_back_to_rules(settings):
    d = _dictate(make_client(settings, FakeLLM(fail="timeout")), "Lower right third molar is horizontally impacted")
    assert d["extractor"] == "rules (LLM fallback)"
    assert d["findings"][0]["site"]["fdi"] == "48"


def test_draft_edit_versioning_and_approval(client):
    d = _dictate(client, "Lower third molar is impacted.", patient_id="P-1001", modality="OPG")
    findings = d["findings"]
    findings[0]["site"] = {
        "kind": "tooth",
        "fdi": "48",
        "universal": "32",
        "name": "lower right third molar",
        "candidates": [],
        "source": "clinician",
    }
    body = {"findings": findings, "recommendations": [], "modality": "OPG", "expected_version": d["version"]}
    r = client.put(f"/api/dictations/{d['draft_id']}", json=body, headers=HDR)
    assert r.status_code == 200, r.text
    edited = r.json()
    assert edited["version"] == 2 and edited["edited_by_clinician"] and edited["status"] == "pending_clinician_review"
    # stale version -> conflict
    assert client.put(f"/api/dictations/{d['draft_id']}", json=body, headers=HDR).status_code == 409
    # invalid tooth code -> rejected by validation
    bad = dict(body, expected_version=2)
    bad["findings"] = [dict(findings[0], site=dict(findings[0]["site"], fdi="99"))]
    assert client.put(f"/api/dictations/{d['draft_id']}", json=bad, headers=HDR).status_code == 422
    approved = client.post(f"/api/dictations/{d['draft_id']}/approve", headers=HDR).json()
    assert approved["status"] == "approved" and approved["approved_by"] == "dr-anand"


def test_dictation_in_conversation_links_patient_from_context(convo):
    convo.say("Open Priya Sharma's latest OPG")
    r = convo.say("Lower right third molar is horizontally impacted")
    assert r["action"] == "dictate"
    assert r["draft"]["patient"]["patient_id"] == "P-1001"
    assert r["draft"]["patient_link"] == "session_context"
    assert any("confirm the patient" in x for x in r["draft"]["review_reasons"])
