"""Orchestrator: interpret -> validate against business rules -> execute -> respond.

The interpreter (LLM or rules) only proposes an Intent. Everything that
decides *what data is shown* happens here, deterministically:
patient resolution, authorisation, report selection, follow-up context,
clarification state and grounding of generated text.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import ValidationError

from app.assistant.sessions import DraftStore, Pending, Session, SessionStore
from app.clinical.dental import fdi_to_universal, tooth_name, valid_fdi
from app.clinical.identity import resolve_patient, select_candidate
from app.config import Settings
from app.features import grounding
from app.features.dictation import DictationService, finalize
from app.features.scheduling import (
    Scheduler,
    SlotUnavailable,
    When,
    is_affirmative,
    parse_when,
    select_slot,
    slot_views,
)
from app.llm import LLM, LLMError
from app.schemas import (
    Answer,
    CandidateView,
    Clinician,
    ContextView,
    DraftEdit,
    DraftReport,
    Patient,
    Report,
    ReportView,
    SourceRef,
    Summary,
    TraceStep,
    TurnResponse,
)
from app.storage.audit import AuditLog
from app.storage.repository import PERMISSIONS, Forbidden, NotFound, Repository, patient_ref, report_view, require
from app.understanding.interpreter import (
    ContextSummary,
    Intent,
    InterpretResult,
    LLMInterpreter,
    RuleInterpreter,
    Target,
)
from app.understanding.normalize import clean_input, detect_modality, normalize_utterance


class InputError(ValueError):
    pass


class Conflict(Exception):
    pass


@dataclass
class Out:
    status: str
    message: str
    report: ReportView | None = None
    summary: Summary | None = None
    answer: Answer | None = None
    draft: DraftReport | None = None
    candidates: list[CandidateView] = field(default_factory=list)
    available_reports: list[dict] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    slots: list[dict] = field(default_factory=list)
    booking: dict | None = None


_BOOKING_KINDS = ("need_modality", "choose_slot", "confirm_booking")
_MODALITY_NAMES = {"OPG": "OPG", "CBCT": "CBCT", "IOPA": "IOPA", "BITEWING": "bitewing", "CEPH": "cephalogram"}
_ARTICLE = {"OPG": "an", "IOPA": "an"}

_FEMALE = re.compile(r"\b(she|her|hers|herself)\b", re.I)
_MALE = re.compile(r"\b(he|him|his|himself)\b", re.I)


def pronoun_gender(text: str) -> tuple[str | None, str | None]:
    if m := _FEMALE.search(text):
        return "F", m.group(0)
    if m := _MALE.search(text):
        return "M", m.group(0)
    return None, None


def _pronoun_for(p: Patient) -> str:
    return {"F": "she/her", "M": "he/him"}.get(p.sex, "they/them")


def _fmt_date(d) -> str:
    return d.strftime("%d %B %Y").lstrip("0")


class Assistant:
    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        llm: LLM | None,
        audit: AuditLog,
        sessions: SessionStore | None = None,
        drafts: DraftStore | None = None,
        scheduler: Scheduler | None = None,
    ):
        self.settings = settings
        self.scheduler = scheduler
        self.repo = repo
        self.llm = llm
        self.audit = audit
        self.sessions = sessions or SessionStore(settings.session_ttl_s)
        self.drafts = drafts or DraftStore()
        self.llm_interpreter = LLMInterpreter(llm, settings.router_effort) if llm else None
        self.dictation = DictationService(llm, settings.extraction_effort)

    # ------------------------------------------------------------ public ---

    def turn(self, clinician: Clinician, text: str, session_id: str | None = None) -> TurnResponse:
        t0 = time.perf_counter()
        cleaned = clean_input(text)
        if not cleaned:
            raise InputError("Empty input.")
        if len(cleaned) > self.settings.max_utterance_chars:
            raise InputError(
                f"Input is {len(cleaned)} characters; the limit for a conversational turn is "
                f"{self.settings.max_utterance_chars}. Use /api/dictations for long dictation."
            )
        session = self.sessions.get_or_create(session_id, clinician.clinician_id)
        session.turns += 1
        norm = normalize_utterance(cleaned)
        trace: list[dict] = []
        interp = self._interpret(norm, session, clinician, trace)
        intent = interp.intent
        try:
            out = self._dispatch(intent, interp, norm, session, clinician, trace)
        except Forbidden as e:
            out = Out("forbidden", str(e))
        except NotFound as e:
            out = Out("not_found", str(e))
        return TurnResponse(
            session_id=session.session_id,
            turn_id=uuid.uuid4().hex[:12],
            input_text=text,
            normalized_text=norm,
            status=out.status,
            action=intent.action,
            message=out.message,
            report=out.report,
            summary=out.summary,
            answer=out.answer,
            draft=out.draft,
            candidates=out.candidates,
            slots=out.slots,
            booking=out.booking,
            available_reports=out.available_reports,
            sources=out.sources,
            context=self._context_view(session, clinician),
            interpreter=interp.engine,
            trace=[TraceStep(step=t["step"], detail=t.get("detail", {}), ms=t.get("ms")) for t in trace],
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def create_dictation(
        self,
        clinician: Clinician,
        transcript: str,
        patient_id: str | None = None,
        modality: str | None = None,
    ) -> DraftReport:
        require(clinician, "dictate")
        cleaned = clean_input(transcript)
        if not cleaned:
            raise InputError("Empty transcript.")
        if len(cleaned) > self.settings.max_transcript_chars:
            raise InputError(f"Transcript exceeds {self.settings.max_transcript_chars} characters.")
        patient = self.repo.get_patient(clinician, patient_id) if patient_id else None
        norm = normalize_utterance(cleaned)
        draft = self.dictation.create_draft(
            transcript_raw=transcript,
            transcript_normalized=norm,
            created_by=clinician.clinician_id,
            created_by_name=clinician.display_name,
            patient=patient_ref(patient) if patient else None,
            patient_link="explicit" if patient else "none",
            modality=modality or detect_modality(norm),
        )
        self.drafts.put(draft)
        self.audit.record(
            clinician_id=clinician.clinician_id,
            action="create_draft",
            outcome="ok",
            draft_id=draft.draft_id,
            patient_id=patient.patient_id if patient else None,
        )
        return draft

    def get_draft(self, clinician: Clinician, draft_id: str) -> DraftReport:
        d = self.drafts.get(draft_id)
        if d is None or d.created_by != clinician.clinician_id:
            raise NotFound(f"No draft {draft_id}.")
        return d

    def update_draft(self, clinician: Clinician, draft_id: str, edit: DraftEdit) -> DraftReport:
        require(clinician, "dictate")
        d = self.get_draft(clinician, draft_id)
        if d.status == "approved":
            raise Conflict("Draft is already approved; create an addendum instead.")
        if edit.expected_version != d.version:
            raise Conflict(f"Draft has changed (version {d.version}); reload before saving.")
        for f in edit.findings:
            if f.site.kind == "tooth" and not (f.site.fdi and valid_fdi(f.site.fdi)):
                raise InputError(f"Finding {f.finding_id}: '{f.site.fdi}' is not a valid FDI tooth code.")
            if f.site.kind == "tooth":  # keep derived fields consistent with the (possibly edited) FDI code
                f.site = f.site.model_copy(
                    update={"universal": fdi_to_universal(f.site.fdi), "name": tooth_name(f.site.fdi), "candidates": []}
                )
        updated = d.model_copy(
            update={
                "findings": edit.findings,
                "recommendations": edit.recommendations,
                "modality": edit.modality or d.modality,
                "version": d.version + 1,
                "edited_by_clinician": True,
            }
        )
        updated = finalize(updated)
        self.drafts.put(updated)
        self.audit.record(clinician_id=clinician.clinician_id, action="edit_draft", outcome="ok", draft_id=draft_id)
        return updated

    def approve_draft(self, clinician: Clinician, draft_id: str) -> DraftReport:
        require(clinician, "approve_draft")
        d = self.get_draft(clinician, draft_id)
        blockers = [
            f"{f.finding_id}: site not fully specified"
            for f in d.findings
            if f.site.kind in ("tooth_unresolved", "unspecified")
        ]
        if not d.patient:
            blockers.append("no patient linked")
        if blockers:
            raise Conflict("Resolve before approving: " + "; ".join(blockers))
        approved = d.model_copy(
            update={
                "status": "approved",
                "review_required": False,
                "approved_by": clinician.clinician_id,
                "approved_at": datetime.now(UTC),
            }
        )
        self.drafts.put(approved)
        self.audit.record(clinician_id=clinician.clinician_id, action="approve_draft", outcome="ok", draft_id=draft_id)
        return approved

    # -------------------------------------------------------- interpret ---

    def _context_summary(self, session: Session, clinician: Clinician) -> ContextSummary:
        ctx = ContextSummary()
        if session.active_patient_id:
            p = self.repo.get_patient(clinician, session.active_patient_id)
            ctx.has_active_patient = True
            ctx.active_patient_pronoun = _pronoun_for(p)
        if session.active_report_id:
            r = self.repo.get_report(clinician, session.active_report_id)
            ctx.active_report = f"{r.modality} {r.study_date.isoformat()}"
        if session.pending:
            ctx.pending_question = {
                "choose_patient": f"The user was asked to choose between {len(session.pending.candidate_ids)} patients with similar names.",
                "confirm_patient": "The user was asked to confirm a patient name (yes/no).",
                "need_patient": "The user was asked which patient they mean.",
                "need_modality": "The user was asked which scan type to book.",
                "choose_slot": f"The user was offered {len(session.pending.data.get('slot_ids', []))} scan appointment slots and asked to pick one.",
                "confirm_booking": "The user was asked to confirm a scan booking (yes/no).",
            }[session.pending.kind]
        return ctx

    def _rules(self, clinician: Clinician) -> RuleInterpreter:
        gaz = set()
        for p in self.repo.accessible_patients(clinician):
            gaz.update({p.first_name.lower(), p.last_name.lower()})
        return RuleInterpreter(gaz)

    def _interpret(self, text: str, session: Session, clinician: Clinician, trace: list) -> InterpretResult:
        ctx = self._context_summary(session, clinician)
        fallback_reason = None
        if self.llm_interpreter:
            t = time.perf_counter()
            try:
                r = self.llm_interpreter.interpret(text, ctx)
                trace.append(
                    {
                        "step": "interpret",
                        "detail": {"engine": r.engine, "intent": r.intent.model_dump(), "usage": r.usage},
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                    }
                )
                return r
            except LLMError as e:
                fallback_reason = e.kind
                trace.append({"step": "interpret", "detail": {"engine": self.llm_interpreter.name, "error": e.kind}})
        t = time.perf_counter()
        r = self._rules(clinician).interpret(text, ctx)
        r.fallback_reason = fallback_reason
        if fallback_reason:
            r.engine = f"rules (fallback: {fallback_reason})"
        trace.append(
            {
                "step": "interpret",
                "detail": {"engine": r.engine, "intent": r.intent.model_dump()},
                "ms": round((time.perf_counter() - t) * 1000, 2),
            }
        )
        return r

    # --------------------------------------------------------- dispatch ---

    def _dispatch(
        self,
        intent: Intent,
        interp: InterpretResult,
        utterance: str,
        session: Session,
        clinician: Clinician,
        trace: list,
    ) -> Out:
        a = intent.action
        if a in ("open_report", "summarize_report", "ask_report"):
            # A new full request supersedes any pending clarification.
            session.pending = None
            target = self._resolve_target(intent, session, clinician, utterance, trace)
            if isinstance(target, Out):
                return target
            return self._run_with_patient(intent, target, session, clinician, trace)
        if a == "book_scan":
            session.pending = None
            require(clinician, "book_scan")
            target = self._resolve_target(intent, session, clinician, utterance, trace)
            if isinstance(target, Out):
                return target
            return self._book(intent, target, session, clinician, trace)
        if a == "select_candidate":
            return self._select(intent, session, clinician, trace, utterance)
        if a == "cancel":
            pending = session.pending
            session.pending = None
            if pending and pending.kind in _BOOKING_KINDS:
                return Out("ok", "Okay, nothing has been booked.")
            return Out("ok", "Okay, cancelled." if pending else "Nothing to cancel.")
        if a == "dictate":
            # Use the clinician's own words, not the router's copy of them.
            text = intent.dictation if interp.engine == "rules" and intent.dictation else utterance
            return self._dictate(text, utterance, session, clinician, trace)
        capabilities = (
            "I can open a patient's report, summarise it, answer questions about it, turn dictated findings "
            "into a draft report, or book an imaging scan."
        )
        return Out("unsupported", f"{intent.reason} {capabilities}" if intent.reason else capabilities)

    # ---------------------------------------------------- patient target ---

    def _candidate_views(self, patients: list[Patient], session: Session) -> list[CandidateView]:
        return [
            CandidateView(
                option=i + 1,
                patient_id=p.patient_id,
                display_name=p.display_name,
                birth_year=p.date_of_birth.year,
                latest_study=self.repo.latest_study_label(p.patient_id),
                is_current_patient=p.patient_id == session.active_patient_id,
            )
            for i, p in enumerate(patients)
        ]

    def _resolve_target(
        self, intent: Intent, session: Session, clinician: Clinician, utterance: str, trace: list
    ) -> Patient | Out:
        tgt = intent.target
        if tgt.reference == "named" and tgt.patient_name:
            res = resolve_patient(tgt.patient_name, self.repo.accessible_patients(clinician))
            trace.append({"step": "resolve_patient", "detail": {"mention": tgt.patient_name, **res.as_trace()}})
            if res.status == "unique":
                return res.patients[0]
            if res.status == "none":
                self.audit.record(clinician_id=clinician.clinician_id, action="resolve_patient", outcome="no_match")
                return Out(
                    "not_found",
                    f"I couldn't find a patient named '{tgt.patient_name}' in your clinic. Please check the name.",
                )
            cands = self._candidate_views(res.patients, session)
            if res.status == "confirm":
                p = res.patients[0]
                q = f"I couldn't find an exact match for '{tgt.patient_name}'. Did you mean {p.display_name}, born {p.date_of_birth.year}?"
                session.pending = Pending("confirm_patient", intent, [p.patient_id], q)
                return Out("needs_clarification", q, candidates=cands)
            listing = "; ".join(
                f"{c.option}) {c.display_name}, born {c.birth_year}"
                + (" (current patient)" if c.is_current_patient else "")
                for c in cands
            )
            q = f"I found {len(cands)} patients matching '{tgt.patient_name}': {listing}. Which one do you mean?"
            session.pending = Pending("choose_patient", intent, [p.patient_id for p in res.patients], q)
            return Out("needs_clarification", q, candidates=cands)

        if session.active_patient_id:
            p = self.repo.get_patient(clinician, session.active_patient_id)
            gender, word = pronoun_gender(utterance)
            if tgt.reference == "context" and gender and p.sex in ("F", "M") and gender != p.sex:
                q = f"You said '{word}', but the current patient is {p.display_name}. Which patient do you mean?"
                session.pending = Pending("need_patient", intent, [], q)
                trace.append({"step": "resolve_patient", "detail": {"status": "pronoun_mismatch", "pronoun": word}})
                return Out("needs_clarification", q)
            trace.append(
                {
                    "step": "resolve_patient",
                    "detail": {"status": "from_context", "patient_id": p.patient_id, "reference": tgt.reference},
                }
            )
            return p
        q = "Which patient do you mean? Please say the patient's name."
        session.pending = Pending("need_patient", intent, [], q)
        trace.append({"step": "resolve_patient", "detail": {"status": "no_context"}})
        return Out("needs_clarification", q)

    def _continue(self, intent: Intent, patient: Patient, session: Session, clinician: Clinician, trace: list) -> Out:
        if intent.action == "book_scan":
            return self._book(intent, patient, session, clinician, trace)
        return self._run_with_patient(intent, patient, session, clinician, trace)

    def _select(self, intent: Intent, session: Session, clinician: Clinician, trace: list, utterance: str = "") -> Out:
        choice = intent.choice or utterance
        p = session.pending
        if p is not None and p.kind in _BOOKING_KINDS:
            return self._select_booking(p, choice, utterance or choice, session, clinician, trace)
        rules = self._rules(clinician)
        if p is None:
            name = rules.extract_name(choice)
            if name:
                return self._dispatch(
                    Intent(action="open_report", target=Target(reference="named", patient_name=name)),
                    InterpretResult(intent=intent, engine="rules"),
                    choice,
                    session,
                    clinician,
                    trace,
                )
            return Out("unsupported", "There's no open question to answer. You can ask me to open a patient's report.")

        if p.kind == "need_patient":
            name = rules.extract_name(choice) or choice.strip(" .?!")
            try:
                new_target = p.intent.target.model_copy(update={"reference": "named", "patient_name": name})
                Target.model_validate(new_target.model_dump())
            except ValidationError:
                return Out("needs_clarification", "Sorry, I didn't catch the name. " + p.question)
            session.pending = None
            new_intent = p.intent.model_copy(update={"target": new_target})
            target = self._resolve_target(new_intent, session, clinician, choice, trace)
            if isinstance(target, Out):
                return target
            return self._continue(new_intent, target, session, clinician, trace)

        candidates = [self.repo.get_patient(clinician, pid) for pid in p.candidate_ids]
        sel = select_candidate(choice, candidates)
        trace.append(
            {
                "step": "select_candidate",
                "detail": {
                    "answer": choice,
                    "selected": sel if isinstance(sel, str) else (sel.patient_id if sel else None),
                },
            }
        )
        if sel == "cancel":
            session.pending = None
            return Out("ok", "Okay, cancelled.")
        if sel is None:
            return Out(
                "needs_clarification",
                "Sorry, I didn't catch which one. " + p.question,
                candidates=self._candidate_views(candidates, session),
            )
        session.pending = None
        return self._continue(p.intent, sel, session, clinician, trace)

    # ------------------------------------------------------- execution ---

    def _pick_report(
        self, intent: Intent, patient: Patient, reports: list[Report], session: Session, trace: list
    ) -> Report | None:
        tgt = intent.target
        unspecified = tgt.modality is None and tgt.which == "latest" and not tgt.report_date
        if unspecified and intent.action != "open_report" and session.active_report_id:
            active = next((r for r in reports if r.report_id == session.active_report_id), None)
            if active:
                trace.append(
                    {
                        "step": "select_report",
                        "detail": {"rule": "active report in conversation", "report_id": active.report_id},
                    }
                )
                return active
        pool = [r for r in reports if tgt.modality is None or r.modality == tgt.modality]
        chosen: Report | None = None
        if pool:
            if tgt.which == "on_date" and tgt.report_date:
                dated = [r for r in pool if r.study_date.isoformat().startswith(tgt.report_date)]
                chosen = dated[-1] if dated else None
            elif tgt.which == "earliest":
                chosen = pool[0]
            elif tgt.which == "previous":
                ids = [r.report_id for r in pool]
                if session.active_report_id in ids and ids.index(session.active_report_id) > 0:
                    chosen = pool[ids.index(session.active_report_id) - 1]
                elif len(pool) >= 2:
                    chosen = pool[-2]
            else:
                chosen = pool[-1]
        trace.append(
            {
                "step": "select_report",
                "detail": {
                    "modality": tgt.modality or "ANY",
                    "which": tgt.which,
                    "date": tgt.report_date,
                    "candidates": [f"{r.report_id} {r.modality} {r.study_date}" for r in pool],
                    "chosen": chosen.report_id if chosen else None,
                },
            }
        )
        return chosen

    def _describe(self, tgt: Target) -> str:
        mod = tgt.modality or ""
        if tgt.which == "on_date" and tgt.report_date:
            return f"{mod} report dated {tgt.report_date}".strip()
        if tgt.which == "previous":
            return f"previous {mod} report".replace("  ", " ")
        if tgt.which == "earliest":
            return f"earliest {mod} report".replace("  ", " ")
        return f"{mod} report".strip() if mod else "report"

    def _run_with_patient(
        self, intent: Intent, patient: Patient, session: Session, clinician: Clinician, trace: list
    ) -> Out:
        try:
            reports = self.repo.reports_for(clinician, patient.patient_id)
        except Forbidden as e:
            self.audit.record(
                clinician_id=clinician.clinician_id,
                action=intent.action,
                outcome="forbidden",
                patient_id=patient.patient_id,
            )
            return Out("forbidden", str(e))
        if session.active_patient_id != patient.patient_id:
            session.active_report_id = None
        session.active_patient_id = patient.patient_id
        report = self._pick_report(intent, patient, reports, session, trace)
        if report is None:
            available = [
                {"report_id": r.report_id, "modality": r.modality, "study_date": r.study_date.isoformat()}
                for r in reports
            ]
            avail_txt = ", ".join(f"{r.modality} ({_fmt_date(r.study_date)})" for r in reversed(reports)) or "none"
            self.audit.record(
                clinician_id=clinician.clinician_id,
                action=intent.action,
                outcome="no_report",
                patient_id=patient.patient_id,
            )
            return Out(
                "not_found",
                f"{patient.display_name} has no {self._describe(intent.target)} on file. Available reports: {avail_txt}.",
                available_reports=available,
            )
        session.active_report_id = report.report_id
        view = report_view(report, patient)
        self.audit.record(
            clinician_id=clinician.clinician_id,
            action=intent.action,
            outcome="ok",
            patient_id=patient.patient_id,
            report_id=report.report_id,
        )
        heading = f"{patient.display_name}'s {report.modality} report from {_fmt_date(report.study_date)}"
        if intent.action == "open_report":
            impression = " ".join(s.text for s in report.impression)
            return Out(
                "ok",
                f"Opening {heading}. Impression: {impression}",
                report=view,
                sources=grounding.sources_for(report, [s.id for s in report.impression]),
            )
        if intent.action == "summarize_report":
            summary = grounding.summarize(report, self.llm, self.settings.router_effort, trace)
            ids = [i for b in summary.bullets for i in b.sentence_ids]
            spoken = " ".join(b.text if b.text.endswith(".") else b.text + "." for b in summary.bullets)
            return Out(
                "ok",
                f"Summary of {heading}: {spoken}",
                report=view,
                summary=summary,
                sources=grounding.sources_for(report, ids),
            )
        question = intent.question or ""
        answer = grounding.answer_question(question, report, self.llm, self.settings.router_effort, trace)
        return Out(
            "ok",
            f"Based on {heading}: {answer.text}",
            report=view,
            answer=answer,
            sources=grounding.sources_for(report, answer.sentence_ids),
        )

    def _dictate(self, text: str, utterance: str, session: Session, clinician: Clinician, trace: list) -> Out:
        require(clinician, "dictate")
        if len(text) > self.settings.max_transcript_chars:
            raise InputError("Dictation too long.")
        patient = self.repo.get_patient(clinician, session.active_patient_id) if session.active_patient_id else None
        steps: list[dict] = []
        draft = self.dictation.create_draft(
            transcript_raw=text,
            transcript_normalized=normalize_utterance(text),
            created_by=clinician.clinician_id,
            created_by_name=clinician.display_name,
            patient=patient_ref(patient) if patient else None,
            patient_link="session_context" if patient else "none",
            modality=detect_modality(text),
            trace=steps,
        )
        trace.extend(steps)
        trace.append(
            {
                "step": "draft",
                "detail": {"draft_id": draft.draft_id, "extractor": draft.extractor, "findings": len(draft.findings)},
            }
        )
        self.drafts.put(draft)
        self.audit.record(
            clinician_id=clinician.clinician_id,
            action="create_draft",
            outcome="ok",
            draft_id=draft.draft_id,
            patient_id=patient.patient_id if patient else None,
        )
        n = len(draft.findings)
        unc = sum(f.certainty == "uncertain" for f in draft.findings)
        extra = []
        if unc:
            extra.append(f"{unc} marked uncertain")
        if draft.unparsed_segments:
            extra.append(f"{len(draft.unparsed_segments)} segment(s) need manual review")
        msg = f"Draft created with {n} finding{'s' if n != 1 else ''}" + (f" ({'; '.join(extra)})" if extra else "")
        msg += ". It needs your review before it can be used."
        return Out("ok", msg, draft=draft)

    # ------------------------------------------------------- scan booking ---

    def _recommended_scan(
        self, patient: Patient, session: Session, clinician: Clinician
    ) -> tuple[str, SourceRef] | None:
        """Scan type recommended in the patient's current (or latest) report, with the sentence as evidence."""
        if "read_reports" not in PERMISSIONS.get(clinician.role, set()):
            return None
        reports = self.repo.reports_for(clinician, patient.patient_id)
        if not reports:
            return None
        report = next((r for r in reports if r.report_id == session.active_report_id), reports[-1])
        for s in report.recommendations:
            mod = detect_modality(s.text)
            if mod:
                return mod, grounding.sources_for(report, [s.id])[0]
        return None

    def _book(self, intent: Intent, patient: Patient, session: Session, clinician: Clinician, trace: list) -> Out:
        require(clinician, "book_scan")
        if self.scheduler is None:
            return Out("unsupported", "Scan booking is not configured.")
        session.active_patient_id = patient.patient_id
        today = self.scheduler.clock().date()
        modality = intent.target.modality
        sources: list[SourceRef] = []
        reason = None
        if modality is None:
            rec = self._recommended_scan(patient, session, clinician)
            if rec is None:
                q = f"Which scan should I book for {patient.display_name}: OPG, CBCT, IOPA, bitewing or cephalogram?"
                session.pending = Pending("need_modality", intent, [patient.patient_id], q)
                trace.append({"step": "booking", "detail": {"stage": "need_modality"}})
                return Out("needs_clarification", q)
            modality, src = rec
            sources = [src]
            reason = f'Recommended in the {src.modality} report of {_fmt_date(src.study_date)}: "{src.text}"'
            trace.append(
                {
                    "step": "booking",
                    "detail": {
                        "stage": "modality_from_report",
                        "modality": modality,
                        "evidence": f"{src.report_id}#{src.sentence_id}",
                    },
                }
            )
        when = parse_when(intent.when_text or "", today)
        slots = self.scheduler.find(patient.clinic_id, modality, when)
        trace.append(
            {
                "step": "check_slots",
                "detail": {
                    "modality": modality,
                    "requested": when.describe(),
                    "clinic": patient.clinic_id,
                    "found": [s.slot_id for s in slots],
                },
            }
        )
        name = _MODALITY_NAMES.get(modality, modality)
        if slots:
            intro = f"Free {name} slots for {patient.display_name}" + ("" if when.empty else f" {when.describe()}")
        else:
            slots = self.scheduler.find(patient.clinic_id, modality, When())
            if not slots:
                return Out(
                    "not_found",
                    f"There are no free {name} slots in the next {self.scheduler.days_ahead} days.",
                    sources=sources,
                )
            intro = f"No free {name} slots {when.describe()}. The earliest available are"
        prefix = f"{patient.display_name}'s report recommends {_ARTICLE.get(modality, 'a')} {name}. " if reason else ""
        data = {
            "slot_ids": [s.slot_id for s in slots],
            "modality": modality,
            "patient_id": patient.patient_id,
            "reason": reason,
        }
        exact = (
            when.at
            and slots[0].start.time() == when.at
            and (not when.date_from or slots[0].start.date() == when.date_from)
        )
        if exact or (len(slots) == 1 and when.at):
            return self._ask_confirmation(intent, slots[0], patient, data, session, sources, prefix)
        listing = "; ".join(f"{i}) {s.label}" for i, s in enumerate(slots, 1))
        q = f"{prefix}{intro}: {listing}. Which one should I book?"
        session.pending = Pending("choose_slot", intent, [patient.patient_id], q, data)
        return Out(
            "needs_clarification",
            q,
            slots=[v.model_dump(mode="json") for v in slot_views(slots, modality)],
            sources=sources,
        )

    def _ask_confirmation(
        self,
        intent: Intent,
        slot,
        patient: Patient,
        data: dict,
        session: Session,
        sources: list[SourceRef] | None = None,
        prefix: str = "",
    ) -> Out:
        name = _MODALITY_NAMES.get(data["modality"], data["modality"])
        q = (
            f"{prefix}Please confirm: book {_ARTICLE.get(data['modality'], 'a')} {name} for {patient.display_name} ({patient.patient_id}) on "
            f"{slot.start:%A %d %B} at {slot.start:%H:%M} ({slot.machine_name}). Say yes to book or no to cancel."
        )
        session.pending = Pending("confirm_booking", intent, [patient.patient_id], q, {**data, "slot_id": slot.slot_id})
        view = slot_views([slot], data["modality"])[0].model_dump(mode="json")
        return Out("needs_confirmation", q, slots=[view], sources=sources or [])

    def _select_booking(
        self, p: Pending, choice: str, utterance: str, session: Session, clinician: Clinician, trace: list
    ) -> Out:
        patient = self.repo.get_patient(clinician, p.candidate_ids[0])
        today = self.scheduler.clock().date()
        if p.kind == "need_modality":
            mod = detect_modality(choice)
            if mod is None:
                return Out("needs_clarification", "Sorry, which scan type? " + p.question)
            session.pending = None
            new_intent = p.intent.model_copy(update={"target": p.intent.target.model_copy(update={"modality": mod})})
            return self._book(new_intent, patient, session, clinician, trace)

        if p.kind == "choose_slot":
            slots = [s for s in (self.scheduler.get(i) for i in p.data["slot_ids"]) if s]
            sel = select_slot(choice, slots, today)
            trace.append(
                {"step": "select_slot", "detail": {"answer": choice, "selected": sel.slot_id if sel else None}}
            )
            if sel is None:
                return Out(
                    "needs_clarification",
                    "Sorry, which slot? Say 'the first one', a time, or 'cancel'. " + p.question,
                    slots=[v.model_dump(mode="json") for v in slot_views(slots, p.data["modality"])],
                )
            return self._ask_confirmation(p.intent, sel, patient, p.data, session)

        # confirm_booking: only an explicit yes in the clinician's own words books the slot.
        affirmative = is_affirmative(utterance)
        trace.append({"step": "confirm_booking", "detail": {"utterance": utterance, "explicit_yes": affirmative}})
        if not affirmative:
            if re.match(r"^\s*(no|nope|cancel|don'?t|stop|never ?mind)\b", utterance, re.I):
                session.pending = None
                return Out("ok", "Okay, nothing has been booked.")
            slot = self.scheduler.get(p.data["slot_id"])
            return Out(
                "needs_confirmation",
                "I need a clear yes or no before booking. " + p.question,
                slots=[slot_views([slot], p.data["modality"])[0].model_dump(mode="json")] if slot else [],
            )
        session.pending = None
        try:
            booking = self.scheduler.book(
                p.data["slot_id"],
                patient_ref(patient),
                p.data["modality"],
                clinician.clinician_id,
                p.data.get("reason"),
            )
        except SlotUnavailable as e:
            alts = self.scheduler.find(patient.clinic_id, p.data["modality"], When())
            trace.append({"step": "book", "detail": {"error": str(e)}})
            if not alts:
                return Out("not_found", f"{e} No other slots are free.")
            q = f"{e} Other free slots: " + "; ".join(f"{i}) {s.label}" for i, s in enumerate(alts, 1)) + ". Which one?"
            session.pending = Pending(
                "choose_slot", p.intent, [patient.patient_id], q, {**p.data, "slot_ids": [s.slot_id for s in alts]}
            )
            return Out(
                "needs_clarification",
                q,
                slots=[v.model_dump(mode="json") for v in slot_views(alts, p.data["modality"])],
            )
        self.audit.record(
            clinician_id=clinician.clinician_id,
            action="book_scan",
            outcome="ok",
            patient_id=patient.patient_id,
            booking_id=booking.booking_id,
            slot_id=booking.slot_id,
        )
        trace.append({"step": "book", "detail": {"booking_id": booking.booking_id, "slot_id": booking.slot_id}})
        name = _MODALITY_NAMES.get(booking.modality, booking.modality)
        return Out(
            "ok",
            f"Booked: {name} for {patient.display_name} on {booking.start:%A %d %B} at {booking.start:%H:%M} "
            f"({booking.machine_name}). Booking ID {booking.booking_id}.",
            booking=booking.model_dump(mode="json"),
        )

    def _context_view(self, session: Session, clinician: Clinician) -> ContextView:
        cv = ContextView(active_report_id=session.active_report_id)
        if session.active_patient_id:
            cv.active_patient = patient_ref(self.repo.get_patient(clinician, session.active_patient_id))
        if session.pending:
            cv.pending_question = session.pending.question
        return cv
