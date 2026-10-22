"""Domain and API models. Everything that crosses a boundary is validated here."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Modality = Literal["OPG", "CBCT", "IOPA", "BITEWING", "CEPH"]
Role = Literal["dentist", "radiologist", "receptionist"]


# ----------------------------------------------------------------- data ---


class Sentence(BaseModel):
    id: str
    text: str


class Report(BaseModel):
    report_id: str
    patient_id: str
    modality: Modality
    study_date: date
    status: Literal["final", "preliminary"]
    reported_by: str
    clinical_indication: str
    findings: list[Sentence]
    impression: list[Sentence]
    recommendations: list[Sentence]

    def sentences(self) -> list[tuple[str, Sentence]]:
        return (
            [("findings", s) for s in self.findings]
            + [("impression", s) for s in self.impression]
            + [("recommendations", s) for s in self.recommendations]
        )

    def sentence(self, sid: str) -> tuple[str, Sentence] | None:
        return next(((sec, s) for sec, s in self.sentences() if s.id == sid), None)


class Patient(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    sex: Literal["F", "M", "X"]
    date_of_birth: date
    clinic_id: str

    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class Clinician(BaseModel):
    clinician_id: str
    display_name: str
    role: Role
    clinic_ids: list[str]


# ---------------------------------------------------------------- views ---


class PatientRef(BaseModel):
    patient_id: str
    display_name: str
    birth_year: int
    sex: str


class CandidateView(BaseModel):
    option: int
    patient_id: str
    display_name: str
    birth_year: int
    latest_study: str | None = None
    is_current_patient: bool = False


class SourceRef(BaseModel):
    report_id: str
    modality: str
    study_date: date
    section: str
    sentence_id: str
    text: str


class ReportView(BaseModel):
    report_id: str
    patient: PatientRef
    modality: str
    study_date: date
    status: str
    reported_by: str
    clinical_indication: str
    findings: list[Sentence]
    impression: list[Sentence]
    recommendations: list[Sentence]
    tooth_map: dict[str, list[str]] = {}  # FDI tooth -> sentence ids mentioning it


class SummaryBullet(BaseModel):
    text: str
    sentence_ids: list[str]


class DroppedClaim(BaseModel):
    text: str
    reason: str


class Summary(BaseModel):
    report_id: str
    bullets: list[SummaryBullet]
    generated_by: str
    dropped: list[DroppedClaim] = []


class Answer(BaseModel):
    question: str
    answerable: bool
    text: str
    sentence_ids: list[str]
    generated_by: str
    dropped: list[DroppedClaim] = []


class TraceStep(BaseModel):
    step: str
    detail: dict = {}
    ms: float | None = None


# ------------------------------------------------------------ dictation ---


class Attribute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=80)


class Site(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tooth", "tooth_unresolved", "region", "unspecified"]
    fdi: str | None = None
    universal: str | None = None
    name: str | None = None
    candidates: list[str] = []
    source: Literal["dictated", "carried_over", "clinician", "none"] = "dictated"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str
    condition: str
    condition_label: str
    observation: str = Field(max_length=300)
    presence: Literal["present", "absent"]
    certainty: Literal["definite", "uncertain"]
    hedge_terms: list[str] = []
    site: Site
    attributes: list[Attribute] = []
    evidence: str = Field(max_length=1000)
    evidence_span: tuple[int, int] | None = None
    review_flags: list[str] = []
    origin: Literal["rules", "llm", "clinician"]


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=500)
    evidence: str = Field(max_length=1000)
    origin: Literal["rules", "llm", "clinician"]


class RejectedClaim(BaseModel):
    claim: str
    reason: str
    origin: str = "llm"


class DraftReport(BaseModel):
    draft_id: str
    version: int = 1
    status: Literal["pending_clinician_review", "approved"] = "pending_clinician_review"
    review_required: bool = True
    patient: PatientRef | None = None
    patient_link: Literal["explicit", "session_context", "none"] = "none"
    modality: str | None = None
    created_at: datetime
    created_by: str
    created_by_name: str | None = None
    transcript_raw: str
    transcript_normalized: str
    findings: list[Finding]
    recommendations: list[Recommendation] = []
    unparsed_segments: list[str] = []
    rejected_claims: list[RejectedClaim] = []
    review_reasons: list[str] = []
    draft_text: str = ""
    extractor: str
    edited_by_clinician: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None


class DraftEdit(BaseModel):
    """Body of PUT /api/dictations/{id}: the clinician's edited structured fields."""

    model_config = ConfigDict(extra="forbid")
    findings: list[Finding] = Field(max_length=50)
    recommendations: list[Recommendation] = Field(default=[], max_length=20)
    modality: Modality | None = None
    expected_version: int


# ------------------------------------------------------------ API I/O ---


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=5000)
    session_id: str | None = Field(default=None, max_length=64)


class DictationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transcript: str = Field(min_length=1, max_length=20000)
    patient_id: str | None = Field(default=None, max_length=32)
    modality: Modality | None = None
    session_id: str | None = Field(default=None, max_length=64)


Status = Literal["ok", "needs_clarification", "needs_confirmation", "not_found", "forbidden", "unsupported", "error"]


class ContextView(BaseModel):
    active_patient: PatientRef | None = None
    active_report_id: str | None = None
    pending_question: str | None = None


class TurnResponse(BaseModel):
    session_id: str
    turn_id: str
    input_text: str
    normalized_text: str
    status: Status
    action: str | None = None
    message: str
    report: ReportView | None = None
    summary: Summary | None = None
    answer: Answer | None = None
    draft: DraftReport | None = None
    candidates: list[CandidateView] = []
    slots: list[dict] = []
    booking: dict | None = None
    available_reports: list[dict] = []
    sources: list[SourceRef] = []
    context: ContextView
    interpreter: str
    trace: list[TraceStep] = []
    latency_ms: float = 0.0
