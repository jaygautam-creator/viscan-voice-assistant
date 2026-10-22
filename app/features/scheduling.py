"""Scan-booking agent support: synthetic slots, "when" parsing, safe booking.

The agent pattern is propose -> confirm -> commit:
  * code (not the LLM) searches the diary and proposes real, free slots;
  * nothing is booked until the clinician explicitly confirms one slot;
  * at commit time the slot is re-checked (it may have gone in the meantime).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

from pydantic import BaseModel

from app.schemas import PatientRef


class SlotUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Machine:
    machine_id: str
    name: str
    clinic_id: str
    modalities: tuple[str, ...]
    slot_minutes: int


class Slot(BaseModel):
    slot_id: str
    clinic_id: str
    machine_id: str
    machine_name: str
    modalities: list[str]
    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        return f"{self.start.strftime('%a %d %b')}, {self.start.strftime('%H:%M')}"


class SlotView(BaseModel):
    option: int
    slot_id: str
    label: str
    start: datetime
    end: datetime
    machine_id: str
    machine_name: str
    modality: str


class Booking(BaseModel):
    booking_id: str
    status: str = "confirmed"
    patient: PatientRef
    modality: str
    slot_id: str
    start: datetime
    end: datetime
    machine_id: str
    machine_name: str
    clinic_id: str
    booked_by: str
    booked_at: datetime
    reason: str | None = None


# ------------------------------------------------------------- "when" parsing ---

_WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WD = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
_TIME = r"(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?"
_PARTS = {
    "morning": (time(9), time(12)),
    "afternoon": (time(12), time(17)),
    "evening": (time(15), time(17)),
    "lunch": (time(12), time(14)),
}


@dataclass
class When:
    date_from: date | None = None
    date_to: date | None = None
    after: time | None = None
    before: time | None = None
    at: time | None = None
    phrases: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.date_from or self.after or self.before or self.at)

    def describe(self) -> str:
        bits = []
        if self.date_from and self.date_to and self.date_to != self.date_from:
            bits.append(f"{self.date_from:%a %d %b} to {self.date_to:%a %d %b}")
        elif self.date_from:
            bits.append(f"on {self.date_from:%a %d %b}")
        if self.at:
            bits.append(f"at {self.at:%H:%M}")
        else:
            if self.after and self.before:
                bits.append(f"between {self.after:%H:%M} and {self.before:%H:%M}")
            elif self.after:
                bits.append(f"after {self.after:%H:%M}")
            elif self.before:
                bits.append(f"before {self.before:%H:%M}")
        return " ".join(bits) or "at the earliest availability"


def _to_time(h: str, m: str | None, ampm: str | None) -> time | None:
    hour, minute = int(h), int(m or 0)
    if minute > 59:
        return None
    if ampm:
        ap = ampm.replace(".", "").lower()
        if hour > 12 or hour == 0:
            return None
        if ap == "pm" and hour != 12:
            hour += 12
        if ap == "am" and hour == 12:
            hour = 0
    elif 1 <= hour <= 7:  # clinic hours: "at 3" means 3 pm
        hour += 12
    if hour > 23:
        return None
    return time(hour, minute)


def parse_when(text: str, today: date) -> When:
    t = text.lower()
    w = When()

    # ---- day
    if re.search(r"\bday after tomorrow\b", t):
        w.date_from = today + timedelta(days=2)
    elif re.search(r"\btomorrow\b", t):
        w.date_from = today + timedelta(days=1)
    elif re.search(r"\btoday\b", t):
        w.date_from = today
    elif m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        w.date_from = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    elif m := re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})\b", t):
        w.date_from = _future(today, int(m.group(1)), _MONTHS[m.group(2)])
    elif m := re.search(rf"\b({_MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t):
        w.date_from = _future(today, int(m.group(2)), _MONTHS[m.group(1)])
    elif m := re.search(r"\b(\d{1,2})/(\d{1,2})\b", t):  # Indian dd/mm
        w.date_from = _future(today, int(m.group(1)), int(m.group(2)))
    elif m := re.search(rf"\b(next|this|coming)?\s*({_WD})\b", t):
        wd = _WEEKDAYS[m.group(2)]
        ahead = (wd - today.weekday()) % 7 or 7
        w.date_from = today + timedelta(days=ahead)
    elif re.search(r"\bnext week\b", t):
        start = today + timedelta(days=7 - today.weekday())
        w.date_from, w.date_to = start, start + timedelta(days=5)
    elif re.search(r"\bthis week\b", t):
        w.date_from, w.date_to = today, today + timedelta(days=max(0, 5 - today.weekday()))
    if w.date_from and not w.date_to:
        w.date_to = w.date_from

    # ---- time of day
    if m := re.search(rf"\bafter\s+{_TIME}", t):
        w.after = _to_time(*m.groups())
    if m := re.search(rf"\bbefore\s+{_TIME}", t):
        w.before = _to_time(*m.groups())
    if re.search(r"\bbefore\s+(noon|lunch)\b", t):
        w.before = time(12)
    if re.search(r"\bafter\s+(noon|lunch)\b", t):
        w.after = time(13) if "lunch" in t else time(12)
    if not (w.after or w.before):
        for part, (a, b) in _PARTS.items():
            if re.search(rf"\b{part}\b", t):
                w.after, w.before = a, b
                break
        if re.search(r"\b(at|around|by|@)\s+noon\b", t):
            w.at = time(12)
        elif (m := re.search(rf"\b(?:at|around|@)\s+{_TIME}", t)) or (
            m := re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", t)
        ):
            w.at = _to_time(*m.groups())
        elif m := re.search(r"\b(\d{1,2})[:.](\d{2})\b", t):
            w.at = _to_time(m.group(1), m.group(2), None)
    return w


def _future(today: date, day: int, month: int) -> date | None:
    try:
        d = date(today.year, month, day)
    except ValueError:
        return None
    return d if d >= today else date(today.year + 1, month, day)


# ------------------------------------------------------------------ diary ---


class Scheduler:
    def __init__(self, config_path: Path, clock: Callable[[], datetime] = datetime.now):
        cfg = json.loads(config_path.read_text())
        self.clock = clock
        self.machines = [
            Machine(m["machine_id"], m["name"], m["clinic_id"], tuple(m["modalities"]), m["slot_minutes"])
            for m in cfg["machines"]
        ]
        self.weekdays = set(cfg["working_weekdays"])
        self.sessions = [(time.fromisoformat(a), time.fromisoformat(b)) for a, b in cfg["sessions"]]
        self.days_ahead = cfg["days_ahead"]
        self.prebooked_percent = cfg["prebooked_percent"]
        self._slots: dict[str, Slot] = {}
        self._days: set[date] = set()
        self._bookings: dict[str, Booking] = {}  # slot_id -> booking
        self._lock = threading.Lock()

    # -- generation --------------------------------------------------------
    def _ensure_day(self, day: date) -> None:
        if day in self._days:
            return
        self._days.add(day)
        if day.weekday() not in self.weekdays:
            return
        for m in self.machines:
            for start_t, end_t in self.sessions:
                cur = datetime.combine(day, start_t)
                end = datetime.combine(day, end_t)
                while cur + timedelta(minutes=m.slot_minutes) <= end:
                    sid = f"{m.machine_id}-{cur:%Y%m%d-%H%M}"
                    self._slots[sid] = Slot(
                        slot_id=sid,
                        clinic_id=m.clinic_id,
                        machine_id=m.machine_id,
                        machine_name=m.name,
                        modalities=list(m.modalities),
                        start=cur,
                        end=cur + timedelta(minutes=m.slot_minutes),
                    )
                    cur += timedelta(minutes=m.slot_minutes)

    def _window(self) -> list[date]:
        today = self.clock().date()
        days = [today + timedelta(days=i) for i in range(self.days_ahead + 1)]
        for d in days:
            self._ensure_day(d)
        return days

    def _prebooked(self, slot_id: str) -> bool:
        return int(hashlib.sha1(slot_id.encode()).hexdigest()[:8], 16) % 100 < self.prebooked_percent

    def is_free(self, slot: Slot) -> bool:
        return slot.start > self.clock() and not self._prebooked(slot.slot_id) and slot.slot_id not in self._bookings

    # -- queries -------------------------------------------------------------
    def find(self, clinic_id: str, modality: str, when: When, limit: int = 3) -> list[Slot]:
        days = self._window()
        lo, hi = days[0], days[-1]
        d_from = max(when.date_from or lo, lo)
        d_to = min(when.date_to or hi, hi)
        out = []
        for s in sorted(self._slots.values(), key=lambda s: (s.start, s.machine_id)):
            if s.clinic_id != clinic_id or modality not in s.modalities or not self.is_free(s):
                continue
            if not (d_from <= s.start.date() <= d_to):
                continue
            st = s.start.time()
            if when.after and st < when.after:
                continue
            if when.before and st >= when.before:
                continue
            out.append(s)
        if when.at:
            target = when.at.hour * 60 + when.at.minute
            out.sort(key=lambda s: (s.start.date(), abs(s.start.hour * 60 + s.start.minute - target)))
            out = [s for s in out if abs(s.start.hour * 60 + s.start.minute - target) <= 90]
        return out[:limit]

    def get(self, slot_id: str) -> Slot | None:
        self._window()
        return self._slots.get(slot_id)

    def bookings(self, clinic_id: str) -> list[Booking]:
        return sorted((b for b in self._bookings.values() if b.clinic_id == clinic_id), key=lambda b: b.start)

    # -- commit --------------------------------------------------------------
    def book(
        self, slot_id: str, patient: PatientRef, modality: str, booked_by: str, reason: str | None = None
    ) -> Booking:
        with self._lock:
            slot = self.get(slot_id)
            if slot is None:
                raise SlotUnavailable("That slot no longer exists.")
            if modality not in slot.modalities:
                raise SlotUnavailable(f"{slot.machine_name} cannot perform {modality}.")
            if not self.is_free(slot):
                raise SlotUnavailable(f"The {slot.label} slot has just been taken.")
            b = Booking(
                booking_id=f"BK-{uuid.uuid4().hex[:6].upper()}",
                patient=patient,
                modality=modality,
                slot_id=slot.slot_id,
                start=slot.start,
                end=slot.end,
                machine_id=slot.machine_id,
                machine_name=slot.machine_name,
                clinic_id=slot.clinic_id,
                booked_by=booked_by,
                booked_at=self.clock(),
                reason=reason,
            )
            self._bookings[slot.slot_id] = b
            return b


# ----------------------------------------------------------- slot choice ---

_ORDINAL = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
_CARDINAL = {"one": 0, "1": 0, "two": 1, "2": 1, "three": 2, "3": 2}
AFFIRMATIVE = re.compile(
    r"^\s*(yes|yeah|yep|yup|confirm(?:ed)?|correct|book it|go ahead|please (?:do|book)|sure|ok(?:ay)?|do it|"
    r"that'?s (?:right|fine|correct|good)|sounds good)\b",
    re.I,
)
_NEGATION = re.compile(r"\b(no|not|don'?t|dont|wait|cancel|hold on|stop|never ?mind|instead)\b", re.I)


def is_affirmative(text: str) -> bool:
    return bool(AFFIRMATIVE.match(text)) and not _NEGATION.search(text)


def select_slot(text: str, slots: list[Slot], today: date) -> Slot | None:
    low = text.lower()
    toks = re.findall(r"[a-z0-9]+", low)
    if re.search(r"\b(earliest|soonest|first available)\b", low):
        return slots[0] if slots else None
    for i, t in enumerate(toks):
        if t == "last" and slots:
            return slots[-1]
        if t in _ORDINAL and _ORDINAL[t] < len(slots):
            return slots[_ORDINAL[t]]
        cardinal_ok = len(toks) == 1 or (i > 0 and toks[i - 1] in ("option", "number", "slot"))
        if t in _CARDINAL and _CARDINAL[t] < len(slots) and cardinal_ok:
            return slots[_CARDINAL[t]]
    w = parse_when(text, today)
    if w.at:
        hits = [s for s in slots if s.start.time() == w.at and (not w.date_from or s.start.date() == w.date_from)]
        if len(hits) == 1:
            return hits[0]
    if w.date_from and not w.at:
        hits = [s for s in slots if s.start.date() == w.date_from]
        if len(hits) == 1:
            return hits[0]
    return None


def slot_views(slots: list[Slot], modality: str) -> list[SlotView]:
    return [
        SlotView(
            option=i + 1,
            slot_id=s.slot_id,
            label=s.label,
            start=s.start,
            end=s.end,
            machine_id=s.machine_id,
            machine_name=s.machine_name,
            modality=modality,
        )
        for i, s in enumerate(slots)
    ]
