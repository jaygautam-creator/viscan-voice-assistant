"""Run the end-to-end demo against a live server and write docs/DEMO.md.

    uv run uvicorn app.main:app --port 8000          # in another terminal
    uv run python scripts/demo.py [--base-url http://127.0.0.1:8000]

Part 1 calls the real HTTP API. Part 2 runs in-process with a *scripted*
LLM that deliberately hallucinates, to show the validator rejecting
unsupported claims (no API key needed, fully reproducible).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT: list[str] = []


def md(text: str = "") -> None:
    OUT.append(text)


def compact(resp: dict) -> dict:
    """Keep the fields a reviewer needs; drop bulky ones."""
    keep: dict = {k: resp.get(k) for k in ("status", "action", "message", "interpreter", "latency_ms") if k in resp}
    if resp.get("report"):
        r = resp["report"]
        keep["report"] = f"{r['report_id']} | {r['patient']['display_name']} | {r['modality']} | {r['study_date']}"
    if resp.get("candidates"):
        keep["candidates"] = [
            f"{c['option']}. {c['display_name']} ({c['patient_id']}, b. {c['birth_year']})" for c in resp["candidates"]
        ]
    if resp.get("available_reports"):
        keep["available_reports"] = resp["available_reports"]
    if resp.get("summary"):
        keep["summary"] = {
            "generated_by": resp["summary"]["generated_by"],
            "bullets": [f"{b['text']}  {b['sentence_ids']}" for b in resp["summary"]["bullets"]],
        }
    if resp.get("answer"):
        a = resp["answer"]
        keep["answer"] = {"answerable": a["answerable"], "text": a["text"], "sentence_ids": a["sentence_ids"]}
    if resp.get("sources"):
        keep["sources"] = [
            f"{s['report_id']}#{s['sentence_id']} ({s['section']}): {s['text']}" for s in resp["sources"]
        ]
    if resp.get("draft"):
        keep["draft"] = compact_draft(resp["draft"])
    if resp.get("slots"):
        keep["slots"] = [
            f"{x['option']}. {x['label']} · {x['modality']} · {x['machine_name']} ({x['slot_id']})"
            for x in resp["slots"]
        ]
    if resp.get("booking"):
        bk = resp["booking"]
        keep["booking"] = {k: bk[k] for k in ("booking_id", "status", "modality", "start", "machine_name", "reason")}
    if "context" in resp:
        ctx = resp["context"]
        keep["context"] = {
            "active_patient": (ctx.get("active_patient") or {}).get("patient_id"),
            "active_report_id": ctx.get("active_report_id"),
            "pending_question": ctx.get("pending_question"),
        }
    if resp.get("trace"):
        keep["trace_steps"] = [t["step"] for t in resp["trace"]]
    return keep


def compact_draft(d: dict) -> dict:
    return {
        "draft_id": d["draft_id"],
        "status": d["status"],
        "extractor": d["extractor"],
        "findings": [
            {
                "id": f["finding_id"],
                "site": f["site"]["fdi"] or f["site"]["name"] or f["site"]["kind"],
                "condition": f["condition"],
                "observation": f["observation"],
                "presence": f["presence"],
                "certainty": f["certainty"],
                "evidence": f["evidence"],
                "review_flags": f["review_flags"],
            }
            for f in d["findings"]
        ],
        "recommendations": [r["text"] for r in d["recommendations"]],
        "unparsed_segments": d["unparsed_segments"],
        "rejected_claims": d["rejected_claims"],
        "review_reasons": d["review_reasons"],
    }


def show(title: str, curl: str, resp: dict | list, note: str = "") -> None:
    md(f"### {title}")
    if note:
        md(note)
        md()
    md("```bash")
    md(curl)
    md("```")
    md("```json")
    md(json.dumps(resp, indent=2, ensure_ascii=False))
    md("```")
    md()


class Api:
    def __init__(self, base: str):
        self.base = base
        self.http = httpx.Client(base_url=base, timeout=120)
        self.session: str | None = None

    def chat(
        self, title: str, text: str, clinician: str = "dr-anand", note: str = "", new_session: bool = False
    ) -> dict:
        if new_session:
            self.session = None
        body = {"text": text, "session_id": self.session}
        r = self.http.post("/api/chat", json=body, headers={"X-Clinician-Id": clinician})
        data = r.json()
        if r.status_code == 200:
            self.session = data["session_id"]
        sid = '"$SID"' if body["session_id"] else "null"
        curl = (
            f"curl -s {self.base}/api/chat -H 'X-Clinician-Id: {clinician}' -H 'Content-Type: application/json' \\\n"
            f'  -d \'{{"text": {json.dumps(text)}, "session_id": {sid}}}\''
        )
        show(title, curl, compact(data) if r.status_code == 200 else {"http_status": r.status_code, **data}, note)
        return data


def part1(base: str) -> None:
    api = Api(base)
    health = api.http.get("/api/health").json()
    md("## Part 1 - live HTTP API")
    md()
    show("Health / configuration", f"curl -s {base}/api/health", health)

    md("## A. Report retrieval")
    md()
    api.chat(
        "A1. Exact name: latest OPG (not the older OPG, not the newer CBCT)",
        "Open Priya Sharma's latest OPG report",
        new_session=True,
    )
    api.chat(
        "A2. Ambiguous name -> clarification, no report returned",
        "Open Priya's latest OPG report",
        new_session=True,
        note="Two patients share the first name Priya in this clinic. The assistant asks instead of guessing.",
    )
    api.chat("A3. Answer the clarification -> original request (latest OPG) resumes", "Priya Sharma")
    api.chat(
        "A4. Missing report -> explicit not-found with what is available",
        "Open Arjun Mehta's latest CBCT",
        new_session=True,
    )
    api.chat("A5. Patient from another clinic is invisible", "Open Priya Nair's OPG", new_session=True)
    api.chat(
        "A6. Same request by the Mumbai radiologist (authorised for that clinic)",
        "Open Priya's OPG",
        clinician="dr-kapoor",
        new_session=True,
    )
    api.chat(
        "A7. Receptionist role cannot read clinical reports",
        "Open Priya Sharma's OPG",
        clinician="front-desk-blr",
        new_session=True,
    )
    api.chat(
        "A8. Likely speech-recognition error -> confirm, never auto-open", "Open Pria Sharma's OPG", new_session=True
    )
    api.chat("A9. Confirm", "yes")

    md("## C. Conversational follow-up (+ bonus: report-aware Q&A)")
    md()
    api.chat("C1. 'Summarize her report' resolves to the report in context; evidence attached", "Summarize her report")
    api.chat("C2. Grounded question (bonus)", "Is the wisdom tooth close to the nerve?")
    api.chat("C3. Question the report does not answer", "Is there a root fracture?")
    api.chat("C4. Follow-up switching modality for the same patient", "Now show me her CBCT")
    api.chat("C5. Follow-up with no context -> asks which patient", "Summarize her report", new_session=True)
    api.chat(
        "C6. Pronoun contradicts the patient in context -> asks", "Open Arjun Mehta's latest OPG", new_session=True
    )
    api.chat("C7.", "Summarize her report")

    md("## B. Clinical dictation -> structured draft")
    md()
    api.chat(
        "B1. The example dictation with no patient in context (draft is created, patient not linked)",
        "Lower right third molar is horizontally impacted",
        new_session=True,
    )
    api.chat("B1b. Open a patient first...", "Open Priya Sharma's latest OPG")
    api.chat(
        "B1c. ...then dictate: the draft is linked to the patient in context (flagged for confirmation)",
        "Lower right third molar is horizontally impacted",
    )

    text = (
        "Lower right third molar is horizontally impacted and close to the mandibular canal. "
        "Possible periapical radiolucency on tooth three six. No caries on 47. The other lower wisdom tooth is partially erupted. "
        "The patient was anxious today. Advise CBCT before extraction."
    )
    r = api.http.post(
        "/api/dictations",
        json={"transcript": text, "patient_id": "P-1001", "modality": "OPG"},
        headers={"X-Clinician-Id": "dr-anand"},
    ).json()
    show(
        "B2. Dictation API with uncertainty, negation, an under-specified tooth and an unmappable remark",
        f"curl -s {base}/api/dictations -H 'X-Clinician-Id: dr-anand' -H 'Content-Type: application/json' \\\n"
        f'  -d \'{{"transcript": {json.dumps(text)}, "patient_id": "P-1001", "modality": "OPG"}}\'',
        compact_draft(r),
    )
    md("Readable draft generated from the validated structure:")
    md("```text")
    md(r["draft_text"])
    md("```")
    md()

    blocked = api.http.post(f"/api/dictations/{r['draft_id']}/approve", headers={"X-Clinician-Id": "dr-anand"})
    show(
        "B3. Approval is blocked while a tooth is unresolved",
        f"curl -s -X POST {base}/api/dictations/{r['draft_id']}/approve -H 'X-Clinician-Id: dr-anand'",
        {"http_status": blocked.status_code, **blocked.json()},
    )
    findings = r["findings"]
    for f in findings:
        if f["site"]["kind"] == "tooth_unresolved":
            f["site"] = {
                "kind": "tooth",
                "fdi": "38",
                "universal": "17",
                "name": "lower left third molar",
                "candidates": [],
                "source": "clinician",
            }
    edit = {
        "findings": findings,
        "recommendations": r["recommendations"],
        "modality": "OPG",
        "expected_version": r["version"],
    }
    e_resp = api.http.put(f"/api/dictations/{r['draft_id']}", json=edit, headers={"X-Clinician-Id": "dr-anand"})
    assert e_resp.status_code == 200, e_resp.text
    e = e_resp.json()
    show(
        "B4. Clinician edits the JSON (resolves the tooth to 38)",
        f"curl -s -X PUT {base}/api/dictations/{r['draft_id']} -H 'X-Clinician-Id: dr-anand' -H 'Content-Type: application/json' -d @edited.json",
        {
            "version": e["version"],
            "edited_by_clinician": e["edited_by_clinician"],
            "status": e["status"],
            "findings": [
                f"{f['finding_id']} {f['site']['fdi']} {f['observation']} ({f['certainty']})" for f in e["findings"]
            ],
        },
    )
    a = api.http.post(f"/api/dictations/{r['draft_id']}/approve", headers={"X-Clinician-Id": "dr-anand"}).json()
    show(
        "B5. Approve",
        f"curl -s -X POST {base}/api/dictations/{r['draft_id']}/approve -H 'X-Clinician-Id: dr-anand'",
        {k: a[k] for k in ("draft_id", "status", "approved_by", "approved_at", "version")},
    )

    md("## D. Bonus: scan-booking agent (checks synthetic slots, books only after explicit confirmation)")
    md()
    api.chat("D1. Open a patient", "Open Priya Sharma's latest OPG", new_session=True)
    api.chat(
        "D2. Book the scan the report recommends: modality taken from R1, free slots offered",
        "Book the scan recommended in her report",
    )
    api.chat("D3. Pick a slot -> confirmation requested, nothing booked yet", "the first one")
    api.chat("D4. Explicit yes -> booked", "yes")
    api.chat("D5. Unclear answers never book", "Schedule an OPG for Arjun Mehta on Monday afternoon", new_session=True)
    api.chat("D6. Choose a slot", "the second one")
    api.chat("D7. Not a clear yes -> still waiting, nothing booked", "hmm, let me check with him first")
    api.chat(
        "D8. Non-imaging appointments are out of scope", "Book Priya Sharma for a cleaning on Friday", new_session=True
    )
    b = api.http.get("/api/bookings", headers={"X-Clinician-Id": "dr-anand"}).json()
    show(
        "D9. Bookings in the clinic diary (only the confirmed one)",
        f"curl -s {base}/api/bookings -H 'X-Clinician-Id: dr-anand'",
        [
            {k: x[k] for k in ("booking_id", "modality", "start", "machine_name", "slot_id", "reason")}
            | {"patient": x["patient"]["display_name"]}
            for x in b
        ],
    )

    md("## Voice input (server-side STT)")
    md()
    manifest = json.loads((ROOT / "samples" / "audio" / "manifest.json").read_text())
    api.session = None
    for item in manifest[:3]:
        path = ROOT / "samples" / "audio" / item["file"]
        files = {"audio": (item["file"], path.read_bytes(), "audio/wav")}
        data = {"session_id": api.session} if api.session else {}
        resp = api.http.post("/api/chat/audio", files=files, data=data, headers={"X-Clinician-Id": "dr-anand"}).json()
        api.session = resp["session_id"]
        stt = resp["trace"][0]
        out = {
            "stt": {
                "provider": stt["detail"]["provider"],
                "model": stt["detail"]["model"],
                "transcript": stt["detail"]["transcript"],
                "ms": stt["ms"],
            },
            **compact(resp),
        }
        show(
            f"Audio: {item['file']}",
            f"curl -s {base}/api/chat/audio -H 'X-Clinician-Id: dr-anand' -F audio=@samples/audio/{item['file']}"
            + (" -F session_id=$SID" if data else ""),
            out,
        )
    path = ROOT / "samples" / "audio" / "06_dictation_multi.wav"
    d = api.http.post(
        "/api/dictations/audio",
        files={"audio": (path.name, path.read_bytes(), "audio/wav")},
        data={"patient_id": "P-1001"},
        headers={"X-Clinician-Id": "dr-anand"},
    ).json()
    show(
        "Audio dictation -> draft",
        f"curl -s {base}/api/dictations/audio -H 'X-Clinician-Id: dr-anand' "
        "-F audio=@samples/audio/06_dictation_multi.wav -F patient_id=P-1001",
        {
            "transcript_raw": d["transcript_raw"],
            "transcript_normalized": d["transcript_normalized"],
            **compact_draft(d),
        },
    )


def part2() -> None:
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.llm import JsonResult, LLMError, LLMUsage, ToolCallResult
    from app.main import create_app
    from app.speech.stt import MockSTT

    class ScriptedLLM:
        name = "scripted-hallucinating-llm"

        def __init__(self, tool_calls=(), json_outputs=(), fail=None):
            self.tool_calls, self.json_outputs, self.fail = list(tool_calls), list(json_outputs), fail

        def tool_call(self, **_):
            if self.fail:
                raise LLMError(self.fail)
            n, i = self.tool_calls.pop(0)
            return ToolCallResult(n, i, LLMUsage(model="scripted"))

        def json_output(self, **_):
            if self.fail:
                raise LLMError(self.fail)
            return JsonResult(self.json_outputs.pop(0), LLMUsage(model="scripted"))

    settings = dataclasses.replace(get_settings(), audit_log_path=None)
    md("## Part 2 - guardrails against a misbehaving LLM (in-process, scripted LLM output)")
    md()
    md(
        "These calls use a scripted stand-in for Claude that returns deliberately wrong output, so the "
        "validator's behaviour is reproducible without an API key. The same code path handles real Claude output."
    )
    md()
    transcript = "Lower right third molar is horizontally impacted. Possible periapical radiolucency on 36."
    llm_out = {
        "findings": [
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
            {
                "evidence_quote": "Lower right third molar is horizontally impacted",
                "condition": "pericoronitis",
                "presence": "present",
                "certainty": "definite",
                "site_text": "48",
                "attributes": [],
            },
            {
                "evidence_quote": "Deep caries on 47",
                "condition": "caries",
                "presence": "present",
                "certainty": "definite",
                "site_text": "47",
                "attributes": [],
            },
            {
                "evidence_quote": "horizontally impacted",
                "condition": "impaction",
                "presence": "present",
                "certainty": "definite",
                "site_text": "38",
                "attributes": [],
            },
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
    client = TestClient(create_app(settings=settings, llm=ScriptedLLM(json_outputs=[llm_out]), stt=MockSTT()))
    d = client.post("/api/dictations", json={"transcript": transcript}, headers={"X-Clinician-Id": "dr-anand"}).json()
    md("### G1. Hallucinated findings are rejected; dropped hedge is restored")
    md(f"Transcript: `{transcript}`")
    md()
    md("Scripted LLM output (what the model proposed):")
    md("```json")
    md(json.dumps(llm_out, indent=2))
    md("```")
    md("What reached the draft after validation:")
    md("```json")
    md(json.dumps(compact_draft(d), indent=2))
    md("```")
    md()

    llm = ScriptedLLM(
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
                    {"text": "Pulpal involvement of 47.", "sentence_ids": ["I2"]},
                    {"text": "Extraction of 38 is required.", "sentence_ids": ["R1"]},
                    {"text": "Mild bone loss.", "sentence_ids": ["F99"]},
                ]
            }
        ],
    )
    client = TestClient(create_app(settings=settings, llm=llm, stt=MockSTT()))
    sid = client.post(
        "/api/chat", json={"text": "Open Priya Sharma's latest OPG"}, headers={"X-Clinician-Id": "dr-anand"}
    ).json()["session_id"]
    s = client.post(
        "/api/chat", json={"text": "Summarize her report", "session_id": sid}, headers={"X-Clinician-Id": "dr-anand"}
    ).json()
    md("### G2. Ungrounded summary bullets are dropped")
    md("```json")
    md(
        json.dumps(
            {
                "kept": s["summary"]["bullets"],
                "dropped": s["summary"]["dropped"],
                "generated_by": s["summary"]["generated_by"],
            },
            indent=2,
        )
    )
    md("```")
    md()

    client = TestClient(create_app(settings=settings, llm=ScriptedLLM(fail="timeout"), stt=MockSTT()))
    r = client.post(
        "/api/chat", json={"text": "Open Priya Sharma's latest OPG report"}, headers={"X-Clinician-Id": "dr-anand"}
    ).json()
    md("### G3. LLM outage -> deterministic fallback, request still served")
    md("```json")
    md(
        json.dumps(
            {
                "interpreter": r["interpreter"],
                "status": r["status"],
                "report": r["report"]["report_id"],
                "trace": [t for t in r["trace"] if t["step"] == "interpret"],
            },
            indent=2,
        )
    )
    md("```")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "DEMO.md")
    args = ap.parse_args()
    md("# Demo: documented API calls with real outputs")
    md()
    md(
        f"Generated by `scripts/demo.py` on {datetime.now().strftime('%Y-%m-%d %H:%M')} against `{args.base_url}`. "
        "Responses are trimmed to the relevant fields (full responses also include the pipeline trace). "
        "`$SID` is the `session_id` returned by the previous call."
    )
    md()
    part1(args.base_url)
    part2()
    args.out.write_text("\n".join(OUT) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
