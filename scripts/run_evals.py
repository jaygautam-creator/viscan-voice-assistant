"""Run the conversational eval set against the rules interpreter and (if configured) Claude.

    uv run python scripts/run_evals.py            # rules only (offline)
    uv run python scripts/run_evals.py --llm      # also Claude (needs ANTHROPIC_API_KEY)

Each case is a short conversation; a turn passes only if every expectation
holds (status, action, chosen report, candidates, cited evidence, draft teeth).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.main import create_app  # noqa: E402
from app.speech.stt import MockSTT  # noqa: E402


def check(resp: dict, exp: dict) -> list[str]:
    errs = []
    report_id = (resp.get("report") or {}).get("report_id")
    if resp.get("summary"):
        report_id = resp["summary"]["report_id"]
    for key, want in exp.items():
        if key == "status" and resp["status"] != want:
            errs.append(f"status {resp['status']} != {want}")
        elif key == "action" and resp["action"] != want:
            errs.append(f"action {resp['action']} != {want}")
        elif key == "report_id" and report_id != want:
            errs.append(f"report {report_id} != {want}")
        elif key == "candidates" and sorted(c["patient_id"] for c in resp["candidates"]) != sorted(want):
            errs.append(f"candidates {[c['patient_id'] for c in resp['candidates']]} != {want}")
        elif key == "has_sources" and bool(resp["sources"]) != want:
            errs.append("missing sources")
        elif key == "cites" and not set(want) <= {s["sentence_id"] for s in resp["sources"]}:
            errs.append(f"cites {[s['sentence_id'] for s in resp['sources']]} lacks {want}")
        elif key == "answerable" and (resp.get("answer") or {}).get("answerable") != want:
            errs.append(f"answerable != {want}")
        elif key == "draft_teeth":
            got = sorted({f["site"]["fdi"] for f in (resp.get("draft") or {}).get("findings", []) if f["site"]["fdi"]})
            if got != sorted(want):
                errs.append(f"draft teeth {got} != {want}")
        elif key == "has_slots" and bool(resp.get("slots")) != want:
            errs.append(f"slots offered = {bool(resp.get('slots'))} != {want}")
        elif key == "slot_modality" and not (resp.get("slots") and all(s["modality"] == want for s in resp["slots"])):
            errs.append(f"slot modality != {want}")
        elif key == "booked" and bool(resp.get("booking")) != want:
            errs.append(f"booked = {bool(resp.get('booking'))} != {want}")
        elif key == "uncertain_teeth":
            got = sorted(
                {
                    f["site"]["fdi"]
                    for f in (resp.get("draft") or {}).get("findings", [])
                    if f["certainty"] == "uncertain"
                }
            )
            if got != sorted(want):
                errs.append(f"uncertain teeth {got} != {want}")
    return errs


_LLM_STEPS = ("llm_extract", "llm_summary", "llm_answer")
EVAL_NOW = datetime(2026, 10, 22, 9, 30)  # fixed clinic clock so slot availability is reproducible


def run(engine: str, cases: list[dict], router_effort: str | None = None, model: str | None = None) -> dict:
    settings = dataclasses.replace(get_settings(), audit_log_path=None)
    llm = None
    if engine == "llm":
        if settings.llm_provider == "none":
            raise SystemExit("No LLM configured: set GEMINI_API_KEY or ANTHROPIC_API_KEY (e.g. in .env).")
        if router_effort:
            settings = dataclasses.replace(settings, router_effort=router_effort)
        if model:
            settings = dataclasses.replace(settings, model=model)
        llm = build_llm(settings)
        if llm is None:
            raise SystemExit("LLM client could not be created; check credentials.")
    else:
        settings = dataclasses.replace(settings, llm_provider="none")
    results, latencies, costs = [], [], []
    routed = {"llm": 0, "fallback": 0}
    gen = {"llm": 0, "fallback": 0}
    errors: dict[str, int] = {}
    for case in cases:
        client = TestClient(create_app(settings=settings, llm=llm, stt=MockSTT(), clock=lambda: EVAL_NOW))
        sid, failures, case_fallbacks = None, [], 0
        for i, turn in enumerate(case["turns"]):
            r = client.post(
                "/api/chat", json={"text": turn["say"], "session_id": sid}, headers={"X-Clinician-Id": "dr-anand"}
            )
            body = r.json()
            sid = body.get("session_id")
            latencies.append(body.get("latency_ms", 0))
            if engine == "llm":
                served = body.get("interpreter", "").startswith("llm:")
                routed["llm" if served else "fallback"] += 1
                case_fallbacks += not served
            for t in body.get("trace", []):
                detail = t.get("detail") or {}
                usage = detail.get("usage")
                if usage and usage.get("est_cost_usd"):
                    costs.append(usage["est_cost_usd"])
                if engine == "llm" and (t["step"] in _LLM_STEPS or t["step"] == "interpret") and detail.get("error"):
                    errors[detail["error"]] = errors.get(detail["error"], 0) + 1
                if engine == "llm" and t["step"] in _LLM_STEPS:
                    ok = not detail.get("error")
                    gen["llm" if ok else "fallback"] += 1
                    case_fallbacks += not ok
            errs = check(body, turn["expect"])
            if errs:
                failures.append(f"turn {i + 1} '{turn['say']}': {'; '.join(errs)} [engine={body.get('interpreter')}]")
        results.append({"id": case["id"], "pass": not failures, "failures": failures, "llm_fallbacks": case_fallbacks})
    passed = sum(r["pass"] for r in results)
    pure = [r for r in results if r["llm_fallbacks"] == 0]
    return {
        "engine": engine if engine == "rules" else f"{llm.name} (router effort {settings.router_effort})",
        "passed": passed,
        "total": len(results),
        "llm_only_passed": sum(r["pass"] for r in pure) if engine == "llm" else None,
        "llm_only_total": len(pure) if engine == "llm" else None,
        "router_turns": routed if engine == "llm" else None,
        "generation_calls": gen if engine == "llm" else None,
        "llm_errors": errors,
        "median_turn_ms": round(statistics.median(latencies), 1) if latencies else None,
        "p95_turn_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
        "total_cost_usd": round(sum(costs), 4) if costs else 0.0,
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate the configured LLM (Gemini or Claude)")
    ap.add_argument("--router-effort", help="override router effort, e.g. minimal / low")
    ap.add_argument("--model", help="override the LLM model id")
    ap.add_argument("--cases", help="comma-separated case ids to run (e.g. to fit a free-tier quota)")
    ap.add_argument("--json", type=Path, help="write full results to this file")
    args = ap.parse_args()
    os.environ.setdefault("VISCAN_AUDIT_LOG", "")
    cases = json.loads((ROOT / "evals" / "cases.json").read_text())
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    engines = ["rules"] + (["llm"] if args.llm else [])
    summary = []
    for e in engines:
        res = run(e, cases, args.router_effort, args.model)
        summary.append(res)
        print(
            f"\n== {res['engine']}: {res['passed']}/{res['total']} cases passed | median {res['median_turn_ms']} ms/turn"
            f" | p95 {res['p95_turn_ms']} ms | est. cost ${res['total_cost_usd']}"
        )
        if e == "llm":
            rt, gc = res["router_turns"], res["generation_calls"]
            print(
                f"   served by the LLM: {rt['llm']}/{rt['llm'] + rt['fallback']} routing turns, "
                f"{gc['llm']}/{gc['llm'] + gc['fallback']} generation calls | errors: {res['llm_errors'] or 'none'}"
            )
            print(f"   cases with no fallback at all: {res['llm_only_passed']}/{res['llm_only_total']} passed")
            if rt["fallback"] or gc["fallback"]:
                print(
                    "   WARNING: some turns fell back to the rules engine, so the pass rate above is NOT a pure LLM score."
                )
        for r in res["results"]:
            if not r["pass"]:
                print(f"  FAIL {r['id']}:")
                for f in r["failures"]:
                    print(f"       {f}")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
