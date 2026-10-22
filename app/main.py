"""FastAPI application: HTTP API + a minimal web UI for the demo."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.assistant.orchestrator import Assistant, Conflict, InputError
from app.config import ROOT, Settings, get_settings
from app.features.scheduling import Scheduler, When, slot_views
from app.llm import LLM, CircuitBreakerLLM, build_llm
from app.schemas import ChatRequest, Clinician, DictationRequest, DraftEdit, DraftReport, TraceStep, TurnResponse
from app.speech.stt import STTError, STTProvider, Transcript, build_stt, validate_audio
from app.storage.audit import AuditLog
from app.storage.repository import Forbidden, NotFound, Repository, patient_ref, report_view, require

log = logging.getLogger("viscan")


def create_app(
    settings: Settings | None = None,
    llm: LLM | None | str = "auto",
    stt: STTProvider | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    repo = Repository(settings.data_dir)
    model = build_llm(settings) if llm == "auto" else llm
    scheduler = Scheduler(settings.data_dir / "scan_machines.json", clock=clock or datetime.now)
    assistant = Assistant(settings, repo, model, AuditLog(settings.audit_log_path), scheduler=scheduler)
    stt_provider = stt or build_stt(settings)

    app = FastAPI(
        title="Viscan voice dental assistant (PoC)",
        version="0.1.0",
        description="Synthetic data only. Report retrieval, dictation-to-draft and grounded follow-ups.",
    )
    app.state.assistant = assistant
    app.state.settings = settings

    @app.middleware("http")
    async def _no_stale_ui(request: Request, call_next):
        # The UI is plain static files; make browsers revalidate so edits show up immediately.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    # ---------------------------------------------------------- auth ---
    def current_clinician(x_clinician_id: str | None = Header(default=None)) -> Clinician:
        """Mock authentication: the caller names a clinician via X-Clinician-Id.

        In production this would be the subject of a verified OIDC/JWT token.
        """
        cid = x_clinician_id or settings.default_clinician
        c = repo.clinician(cid)
        if c is None:
            raise HTTPException(401, "Unknown clinician.")
        return c

    # ------------------------------------------------- error handling ---
    @app.exception_handler(NotFound)
    async def _nf(_: Request, e: NotFound):
        return JSONResponse({"detail": str(e)}, status_code=404)

    @app.exception_handler(Forbidden)
    async def _fb(_: Request, e: Forbidden):
        return JSONResponse({"detail": str(e)}, status_code=403)

    @app.exception_handler(InputError)
    async def _ie(_: Request, e: InputError):
        return JSONResponse({"detail": str(e)}, status_code=422)

    @app.exception_handler(Conflict)
    async def _cf(_: Request, e: Conflict):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(STTError)
    async def _stt(_: Request, e: STTError):
        return JSONResponse({"detail": str(e)}, status_code=e.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        ref = uuid.uuid4().hex[:8]
        log.exception("unhandled error ref=%s", ref)
        return JSONResponse({"detail": f"Internal error (ref {ref})."}, status_code=500)

    async def _read_audio(audio: UploadFile) -> tuple[bytes, str]:
        data = await audio.read(settings.max_audio_bytes + 1)
        name = audio.filename or "audio.wav"
        validate_audio(data, name, settings.max_audio_bytes)
        return data, name

    # ------------------------------------------------------- routes ---
    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "interpreter": assistant.llm_interpreter.name if assistant.llm_interpreter else "rules",
            "llm_model": settings.model if model else None,
            "stt_provider": stt_provider.name,
            "clinic_time": scheduler.clock().isoformat(timespec="minutes"),
            "llm_status": _llm_status(),
        }

    def _llm_status() -> str:
        if model is None:
            return "offline (rules mode)"
        if isinstance(model, CircuitBreakerLLM):
            import time as _t

            remaining = model.open_until - _t.time()
            if remaining > 0:
                return (
                    f"paused after {model.last_kind}; rules fallback for {int(remaining // 60)}m {int(remaining % 60)}s"
                )
        return "ready"

    @app.post("/api/chat", response_model=TurnResponse)
    def chat(body: ChatRequest, clinician: Clinician = Depends(current_clinician)) -> TurnResponse:
        return assistant.turn(clinician, body.text, body.session_id)

    @app.post("/api/chat/audio", response_model=TurnResponse)
    async def chat_audio(
        audio: UploadFile = File(...),
        session_id: str | None = Form(default=None),
        clinician: Clinician = Depends(current_clinician),
    ) -> TurnResponse:
        data, name = await _read_audio(audio)
        tr = stt_provider.transcribe(data, name)
        resp = assistant.turn(clinician, tr.text, session_id)
        resp.trace.insert(
            0,
            TraceStep(
                step="stt", detail=tr.model_dump(exclude={"text"}) | {"transcript": tr.text}, ms=round(tr.latency_ms, 1)
            ),
        )
        return resp

    @app.post("/api/transcribe", response_model=Transcript)
    async def transcribe(
        audio: UploadFile = File(...), clinician: Clinician = Depends(current_clinician)
    ) -> Transcript:
        data, name = await _read_audio(audio)
        return stt_provider.transcribe(data, name)

    @app.post("/api/dictations", response_model=DraftReport, status_code=201)
    def create_dictation(body: DictationRequest, clinician: Clinician = Depends(current_clinician)) -> DraftReport:
        return assistant.create_dictation(clinician, body.transcript, body.patient_id, body.modality)

    @app.post("/api/dictations/audio", response_model=DraftReport, status_code=201)
    async def create_dictation_audio(
        audio: UploadFile = File(...),
        patient_id: str | None = Form(default=None),
        modality: str | None = Form(default=None),
        clinician: Clinician = Depends(current_clinician),
    ) -> DraftReport:
        data, name = await _read_audio(audio)
        tr = stt_provider.transcribe(data, name)
        body = DictationRequest(transcript=tr.text, patient_id=patient_id, modality=modality)
        return assistant.create_dictation(clinician, body.transcript, body.patient_id, body.modality)

    @app.get("/api/dictations/{draft_id}", response_model=DraftReport)
    def get_dictation(draft_id: str, clinician: Clinician = Depends(current_clinician)) -> DraftReport:
        return assistant.get_draft(clinician, draft_id)

    @app.put("/api/dictations/{draft_id}", response_model=DraftReport)
    def edit_dictation(
        draft_id: str, body: DraftEdit, clinician: Clinician = Depends(current_clinician)
    ) -> DraftReport:
        return assistant.update_draft(clinician, draft_id, body)

    @app.post("/api/dictations/{draft_id}/approve", response_model=DraftReport)
    def approve_dictation(draft_id: str, clinician: Clinician = Depends(current_clinician)) -> DraftReport:
        return assistant.approve_draft(clinician, draft_id)

    @app.get("/api/patients")
    def list_patients(clinician: Clinician = Depends(current_clinician)) -> list[dict]:
        return [
            {**patient_ref(p).model_dump(), "latest_study": repo.latest_study_label(p.patient_id)}
            for p in repo.accessible_patients(clinician)
        ]

    @app.get("/api/patients/{patient_id}/reports")
    def list_reports(patient_id: str, clinician: Clinician = Depends(current_clinician)) -> list[dict]:
        return [
            {"report_id": r.report_id, "modality": r.modality, "study_date": r.study_date.isoformat()}
            for r in repo.reports_for(clinician, patient_id)
        ]

    @app.get("/api/reports/{report_id}")
    def get_report(report_id: str, clinician: Clinician = Depends(current_clinician)):
        r = repo.get_report(clinician, report_id)
        return report_view(r, repo.get_patient(clinician, r.patient_id))

    @app.get("/api/slots")
    def list_slots(
        modality: str, day: date_cls | None = None, clinician: Clinician = Depends(current_clinician)
    ) -> list[dict]:
        """Free slots in the caller's clinic (read-only; booking goes through the confirm flow in /api/chat)."""
        require(clinician, "book_scan")
        when = When(date_from=day, date_to=day) if day else When()
        slots = [s for cid in clinician.clinic_ids for s in scheduler.find(cid, modality.upper(), when, limit=50)]
        return [v.model_dump(mode="json") for v in slot_views(slots, modality.upper())]

    @app.get("/api/bookings")
    def list_bookings(clinician: Clinician = Depends(current_clinician)) -> list[dict]:
        require(clinician, "book_scan")
        return [b.model_dump(mode="json") for cid in clinician.clinic_ids for b in scheduler.bookings(cid)]

    @app.get("/api/clinicians")
    def clinicians() -> list[dict]:
        """Demo only: lets the UI switch the mock identity."""
        return [c.model_dump() for c in (repo.clinician(i) for i in ("dr-anand", "dr-kapoor", "front-desk-blr")) if c]

    static = ROOT / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")
    samples = ROOT / "samples" / "audio"
    if samples.is_dir():
        app.mount("/samples", StaticFiles(directory=samples), name="samples")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(static) / "index.html")

    return app


app = create_app()
