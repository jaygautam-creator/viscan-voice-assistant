"""Deterministic patient identity resolution.

Safety rules (business rules, never delegated to the LLM):
  * A report is opened only for a *unique, exact* name match.
  * Several exact matches -> ask the clinician to choose (ambiguous).
  * Only a fuzzy match (likely STT error, e.g. "Pria") -> ask to confirm.
  * The search space is the clinician's own clinics only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rapidfuzz import fuzz

from app.schemas import Patient

_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "master", "patient", "the", "named", "called"}
FUZZY_THRESHOLD = 80


def name_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z'\-]*", text.lower()) if t not in _HONORIFICS]


def _patient_names(p: Patient) -> list[str]:
    return [p.first_name.lower(), p.last_name.lower()]


@dataclass
class Resolution:
    status: Literal["unique", "ambiguous", "confirm", "none"]
    patients: list[Patient] = field(default_factory=list)
    method: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    def as_trace(self) -> dict:
        return {
            "status": self.status,
            "method": self.method,
            "candidates": [p.patient_id for p in self.patients],
            "scores": self.scores,
        }


def _token_matches(tok: str, name: str) -> bool:
    if len(tok) == 1:  # an initial: "Priya S"
        return name.startswith(tok)
    return tok == name


def resolve_patient(mention: str, patients: list[Patient]) -> Resolution:
    toks = name_tokens(mention)
    if not toks:
        return Resolution("none", method="empty_mention")

    exact = [p for p in patients if all(any(_token_matches(t, n) for n in _patient_names(p)) for t in toks)]
    if exact:
        exact.sort(key=lambda p: (p.last_name, p.first_name))
        return Resolution(
            "unique" if len(exact) == 1 else "ambiguous",
            exact,
            "exact",
            {p.patient_id: 100.0 for p in exact},
        )

    scored: list[tuple[Patient, float]] = []
    for p in patients:
        names = _patient_names(p)
        per_token = [max(fuzz.ratio(t, n) for n in names) for t in toks if len(t) > 1]
        if per_token and min(per_token) >= FUZZY_THRESHOLD:
            scored.append((p, sum(per_token) / len(per_token)))
    if not scored:
        return Resolution("none", method="no_match")
    scored.sort(key=lambda x: -x[1])
    pats = [p for p, _ in scored]
    scores = {p.patient_id: round(s, 1) for p, s in scored}
    # Never auto-open on a fuzzy match: one candidate -> confirm, several -> choose.
    return Resolution("confirm" if len(pats) == 1 else "ambiguous", pats, "fuzzy", scores)


_ORDINALS = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "1": 0,
    "second": 1,
    "2nd": 1,
    "two": 1,
    "2": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
    "3": 2,
    "fourth": 3,
    "4th": 3,
    "four": 3,
    "4": 3,
}
_CARDINALS = {"one", "two", "three", "four", "1", "2", "3", "4"}
_YES = re.compile(
    r"^\s*(yes|yeah|yep|yup|correct|right|that'?s (?:right|correct|her|him)|confirm(?:ed)?|sure|ok(?:ay)?)\b", re.I
)
_CANCEL = re.compile(r"^\s*(no|nope|cancel|never ?mind|neither|none|stop|forget it)\b", re.I)


def select_candidate(text: str, candidates: list[Patient]) -> Patient | Literal["cancel"] | None:
    """Interpret the clinician's answer to "which patient did you mean?"."""
    low = text.lower().strip()
    toks = re.findall(r"[a-z0-9\-]+", low)

    # 1) A distinguishing name token ("Shah", "Priya Shah").
    name_hits = {
        p.patient_id
        for p in candidates
        for t in toks
        if len(t) > 1 and t in _patient_names(p) and sum(t in _patient_names(c) for c in candidates) == 1
    }
    if len(name_hits) == 1:
        return next(p for p in candidates if p.patient_id in name_hits)

    # 2) Patient ID ("P-1002" / "1002").
    for p in candidates:
        if p.patient_id.lower() in low or p.patient_id.split("-")[-1] in toks:
            return p

    # 3) Birth year ("the one born in 1985").
    years = [int(t) for t in toks if re.fullmatch(r"(19|20)\d{2}", t)]
    if years:
        hits = [p for p in candidates if p.date_of_birth.year in years]
        if len(hits) == 1:
            return hits[0]

    # 4) Ordinals ("the second one", "the last one") always count; cardinals ("one", "2")
    #    only alone or after "option"/"number", so "the tall one" is not read as option 1.
    for i, t in enumerate(toks):
        if t == "last":
            return candidates[-1]
        if t in _ORDINALS and _ORDINALS[t] < len(candidates):
            cardinal = t in _CARDINALS
            if not cardinal or len(toks) == 1 or (i > 0 and toks[i - 1] in ("option", "number", "no")):
                return candidates[_ORDINALS[t]]

    # 5) Yes / no.
    if _CANCEL.match(low):
        return "cancel"
    if len(candidates) == 1 and _YES.match(low):
        return candidates[0]
    return None
