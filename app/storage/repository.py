"""Mock patient/report store with clinic-scoped authorisation.

Every read takes the calling clinician. Patients outside the clinician's
clinics are invisible (treated as "not found", so their existence is not
revealed); role permissions gate clinical content.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas import Clinician, Patient, PatientRef, Report, ReportView

PERMISSIONS: dict[str, set[str]] = {
    "dentist": {"read_reports", "dictate", "approve_draft", "book_scan"},
    "radiologist": {"read_reports", "dictate", "approve_draft", "book_scan"},
    "receptionist": {"read_demographics", "book_scan"},
}


class NotFound(Exception):
    pass


class Forbidden(Exception):
    pass


def require(clinician: Clinician, permission: str) -> None:
    if permission not in PERMISSIONS.get(clinician.role, set()):
        raise Forbidden(f"Role '{clinician.role}' is not allowed to {permission.replace('_', ' ')}.")


def patient_ref(p: Patient) -> PatientRef:
    return PatientRef(patient_id=p.patient_id, display_name=p.display_name, birth_year=p.date_of_birth.year, sex=p.sex)


def _tooth_map(r: Report) -> dict[str, list[str]]:
    from app.clinical.dental import parse_sites

    out: dict[str, list[str]] = {}
    for _, s in r.sentences():
        for code in parse_sites(s.text).fdi_codes:
            out.setdefault(code, []).append(s.id)
    return out


def report_view(r: Report, p: Patient) -> ReportView:
    return ReportView(
        tooth_map=_tooth_map(r),
        report_id=r.report_id,
        patient=patient_ref(p),
        modality=r.modality,
        study_date=r.study_date,
        status=r.status,
        reported_by=r.reported_by,
        clinical_indication=r.clinical_indication,
        findings=r.findings,
        impression=r.impression,
        recommendations=r.recommendations,
    )


class Repository:
    def __init__(self, data_dir: Path):
        self._clinicians = {
            c["clinician_id"]: Clinician(**c) for c in json.loads((data_dir / "clinicians.json").read_text())
        }
        self._patients = {p["patient_id"]: Patient(**p) for p in json.loads((data_dir / "patients.json").read_text())}
        reports = [Report(**r) for r in json.loads((data_dir / "reports.json").read_text())]
        self._reports = {r.report_id: r for r in reports}
        for r in reports:
            if r.patient_id not in self._patients:
                raise ValueError(f"Report {r.report_id} references unknown patient {r.patient_id}")

    # -- identity ---------------------------------------------------------
    def clinician(self, clinician_id: str) -> Clinician | None:
        return self._clinicians.get(clinician_id)

    # -- patients ---------------------------------------------------------
    def accessible_patients(self, clinician: Clinician) -> list[Patient]:
        return [p for p in self._patients.values() if p.clinic_id in clinician.clinic_ids]

    def get_patient(self, clinician: Clinician, patient_id: str) -> Patient:
        p = self._patients.get(patient_id)
        if p is None or p.clinic_id not in clinician.clinic_ids:
            raise NotFound(f"No patient {patient_id} in your clinics.")
        return p

    # -- reports ----------------------------------------------------------
    def reports_for(self, clinician: Clinician, patient_id: str) -> list[Report]:
        require(clinician, "read_reports")
        self.get_patient(clinician, patient_id)
        return sorted(
            (r for r in self._reports.values() if r.patient_id == patient_id),
            key=lambda r: (r.study_date, r.report_id),
        )

    def get_report(self, clinician: Clinician, report_id: str) -> Report:
        require(clinician, "read_reports")
        r = self._reports.get(report_id)
        if r is None:
            raise NotFound(f"No report {report_id}.")
        self.get_patient(clinician, r.patient_id)  # clinic check; raises NotFound
        return r

    def latest_study_label(self, patient_id: str) -> str | None:
        rs = [r for r in self._reports.values() if r.patient_id == patient_id]
        if not rs:
            return None
        r = max(rs, key=lambda x: x.study_date)
        return f"{r.modality} {r.study_date.isoformat()}"
