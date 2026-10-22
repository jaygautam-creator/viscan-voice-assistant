"""In-memory conversation state (per clinician, with TTL).

Production would use an encrypted store (e.g. Redis with TTL) keyed by an
authenticated session; the shape of the state would stay the same.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from app.schemas import DraftReport
from app.storage.repository import Forbidden
from app.understanding.interpreter import Intent


@dataclass
class Pending:
    kind: Literal[
        "choose_patient", "confirm_patient", "need_patient", "need_modality", "choose_slot", "confirm_booking"
    ]
    intent: Intent
    candidate_ids: list[str]
    question: str
    data: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    clinician_id: str
    created_at: float
    last_seen: float
    active_patient_id: str | None = None
    active_report_id: str | None = None
    pending: Pending | None = None
    turns: int = 0


class SessionStore:
    def __init__(self, ttl_s: int = 1800):
        self.ttl_s = ttl_s
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None, clinician_id: str) -> Session:
        now = time.time()
        with self._lock:
            self._expire(now)
            s = self._sessions.get(session_id) if session_id else None
            if s is not None and s.clinician_id != clinician_id:
                raise Forbidden("This session belongs to another user.")
            if s is None:
                s = Session(secrets.token_urlsafe(12), clinician_id, now, now)
                self._sessions[s.session_id] = s
            s.last_seen = now
            return s

    def _expire(self, now: float) -> None:
        dead = [k for k, s in self._sessions.items() if now - s.last_seen > self.ttl_s]
        for k in dead:
            del self._sessions[k]


class DraftStore:
    def __init__(self) -> None:
        self._drafts: dict[str, DraftReport] = {}
        self._lock = threading.Lock()

    def put(self, d: DraftReport) -> None:
        with self._lock:
            self._drafts[d.draft_id] = d

    def get(self, draft_id: str) -> DraftReport | None:
        return self._drafts.get(draft_id)
