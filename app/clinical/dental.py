"""Deterministic dental anatomy parsing.

Turns phrases like "lower right third molar", "tooth 48" or "both lower
wisdom teeth" into FDI tooth codes (with Universal numbering for display).
The LLM may *propose* a site, but the code here re-derives it from the
dictated words, so a hallucinated tooth number cannot reach the draft.

Numbers are read as FDI (ISO 3950), the default in Indian dental practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# FDI quadrant by (arch, side). Deciduous quadrants are +4.
_QUADRANT = {("upper", "right"): 1, ("upper", "left"): 2, ("lower", "left"): 3, ("lower", "right"): 4}
_QUADRANT_NAME = {
    1: ("upper", "right"),
    2: ("upper", "left"),
    3: ("lower", "left"),
    4: ("lower", "right"),
    5: ("upper", "right"),
    6: ("upper", "left"),
    7: ("lower", "left"),
    8: ("lower", "right"),
}
PERMANENT_NAMES = {
    1: "central incisor",
    2: "lateral incisor",
    3: "canine",
    4: "first premolar",
    5: "second premolar",
    6: "first molar",
    7: "second molar",
    8: "third molar",
}
DECIDUOUS_NAMES = {1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first molar", 5: "second molar"}

# Spoken tooth type -> (position, plural_form_regex). Order matters: longest first.
_TOOTH_TYPES: list[tuple[str, int]] = [
    (r"central incisors?", 1),
    (r"lateral incisors?", 2),
    (r"canines?|cuspids?|eye ?teeth|eye ?tooth", 3),
    (r"first (?:pre-?molars?|bicuspids?)", 4),
    (r"second (?:pre-?molars?|bicuspids?)", 5),
    (r"first molars?", 6),
    (r"second molars?", 7),
    (r"third molars?|wisdom (?:tooth|teeth)", 8),
]
_TOOTH_TYPE_RE = re.compile(
    r"\b(?P<dec>(?:deciduous|primary|baby|milk)\s+)?(?P<type>" + "|".join(f"(?:{p})" for p, _ in _TOOTH_TYPES) + r")\b",
    re.I,
)
_ARCH_WORDS = {
    "upper": "upper",
    "maxillary": "upper",
    "maxilla": "upper",
    "top": "upper",
    "lower": "lower",
    "mandibular": "lower",
    "mandible": "lower",
    "bottom": "lower",
}
_ARCH_RE = re.compile(r"\b(upper|maxillary|maxilla|top|lower|mandibular|mandible|bottom)\b", re.I)
_SIDE_RE = re.compile(r"\b(right|left)\b", re.I)
_BOTH_RE = re.compile(r"\b(both|bilateral(?:ly)?|either side|each side)\b", re.I)

_PALMER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_PALMER_RE = re.compile(
    r"\b(?:(?P<arch>upper|lower|maxillary|mandibular|top|bottom)\s+(?P<side>right|left)\s+"
    r"|(?P<q>ur|ul|ll|lr)\s?)(?P<pos>one|two|three|four|five|six|seven|eight|[1-8])s?\b(?!\s*(?:mm|cm|%))",
    re.I,
)

# Two-digit numbers that are measurements, grades, ages etc. are not teeth.
_NUMBER_RE = re.compile(r"(?<![\d.])(\d{2})(?![\d.])")
_MONTH_WORDS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_NOT_TOOTH_BEFORE = re.compile(
    rf"(grade|class|type|stage|measuring|approximately|approx\.?|about|around|size|aged?|over|under|than|"
    rf"{_MONTH_WORDS})\s*$",
    re.I,
)
_TOOTH_CUE_BEFORE = re.compile(r"(tooth|teeth|#|number|no\.?)\s*$", re.I)
_NOT_TOOTH_AFTER = re.compile(
    rf"^\s*(mm|millimet\w*|cm|centimet\w*|%|percent|years?|yrs?|months?|days?|weeks?|degrees?|°|hours?|minutes?|"
    rf"(?:st|nd|rd|th)\b|(?:of\s+)?(?:{_MONTH_WORDS})\b)",
    re.I,
)


def valid_fdi(code: str) -> bool:
    if len(code) != 2 or not code.isdigit():
        return False
    q, p = int(code[0]), int(code[1])
    if 1 <= q <= 4:
        return 1 <= p <= 8
    if 5 <= q <= 8:
        return 1 <= p <= 5
    return False


def fdi_to_universal(code: str) -> str | None:
    """Universal/ADA numbering (1-32 permanent, A-T deciduous) for display."""
    if not valid_fdi(code):
        return None
    q, p = int(code[0]), int(code[1])
    if q <= 4:
        return str({1: 9 - p, 2: 8 + p, 3: 25 - p, 4: 24 + p}[q])
    letters = "ABCDEFGHIJKLMNOPQRST"
    idx = {5: 5 - p, 6: 4 + p, 7: 15 - p, 8: 14 + p}[q]
    return letters[idx]


def tooth_name(code: str) -> str:
    q, p = int(code[0]), int(code[1])
    arch, side = _QUADRANT_NAME[q]
    if q >= 5:
        return f"{arch} {side} deciduous {DECIDUOUS_NAMES[p]}"
    return f"{arch} {side} {PERMANENT_NAMES[p]}"


@dataclass(frozen=True)
class ToothRef:
    fdi: str
    start: int
    end: int
    source: str  # "number" | "name"

    @property
    def universal(self) -> str | None:
        return fdi_to_universal(self.fdi)

    @property
    def name(self) -> str:
        return tooth_name(self.fdi)


@dataclass(frozen=True)
class PartialToothRef:
    """A tooth type was dictated but the arch and/or side was not."""

    phrase: str
    candidates: tuple[str, ...]
    missing: tuple[str, ...]
    start: int
    end: int


@dataclass(frozen=True)
class RegionRef:
    region: str
    side: str | None
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"{self.side} {self.region}" if self.side else self.region


@dataclass
class SiteParse:
    teeth: list[ToothRef] = field(default_factory=list)
    partial: list[PartialToothRef] = field(default_factory=list)
    regions: list[RegionRef] = field(default_factory=list)

    @property
    def fdi_codes(self) -> list[str]:
        seen: list[str] = []
        for t in self.teeth:
            if t.fdi not in seen:
                seen.append(t.fdi)
        return seen

    @property
    def empty(self) -> bool:
        return not (self.teeth or self.partial or self.regions)


_REGIONS: list[tuple[str, str]] = [
    (
        "inferior alveolar canal",
        r"inferior alveolar (?:canal|nerve)|mandibular canal|ian canal|nerve canal|alveolar nerve",
    ),
    ("mental foramen", r"mental foramen"),
    ("maxillary sinus", r"maxillary sinus(?:es)?|maxillary antrum|\bsinus(?:es)?\b|\bantrum\b"),
    ("temporomandibular joint", r"temporomandibular joints?|\btmj\b|condyles?|condylar"),
    (
        "mandibular anterior region",
        r"mandibular anterior(?: region)?|lower anterior(?: region| teeth)?|lower front teeth|mandibular incisors",
    ),
    (
        "maxillary anterior region",
        r"maxillary anterior(?: region)?|upper anterior(?: region| teeth)?|upper front teeth|maxillary incisors",
    ),
    ("posterior mandible", r"posterior mandible|mandibular posterior(?: region)?|lower posterior(?: region| teeth)?"),
    ("posterior maxilla", r"posterior maxilla|maxillary posterior(?: region)?|upper posterior(?: region| teeth)?"),
    ("both arches", r"both arches|full mouth|generali[sz]ed"),
    ("airway", r"\bairway\b"),
]
_REGION_RES = [(name, re.compile(pat, re.I)) for name, pat in _REGIONS]


def _nearest_word(pattern: re.Pattern, text: str, lo: int, hi: int) -> re.Match | None:
    """Last match of pattern inside text[lo:hi] (closest to the tooth phrase)."""
    last = None
    for m in pattern.finditer(text, lo, hi):
        last = m
    return last


def parse_sites(text: str) -> SiteParse:
    out = SiteParse()
    taken: list[tuple[int, int]] = []

    # 1) Named teeth: "lower right third molar", "upper left deciduous canine",
    #    "third molar on the lower right", "both lower wisdom teeth".
    type_matches = list(_TOOTH_TYPE_RE.finditer(text))
    prev_end = 0
    for i, m in enumerate(type_matches):
        position = next(pos for pat, pos in _TOOTH_TYPES if re.fullmatch(pat, m.group("type"), re.I))
        deciduous = bool(m.group("dec"))
        # Arch/side words may sit up to ~40 chars before, or right after ("on the lower right").
        lo = max(prev_end, m.start() - 40)
        next_start = type_matches[i + 1].start() if i + 1 < len(type_matches) else len(text)
        hi_after = min(next_start, m.end() + 25)
        before = text[lo : m.start()]
        after = text[m.end() : hi_after]
        after_ok = re.match(r"^\s*(?:on|of|in)\s+the\s+", after, re.I)
        arch_m = _nearest_word(_ARCH_RE, text, lo, m.start())
        side_m = _nearest_word(_SIDE_RE, text, lo, m.start())
        span_start = min([x.start() for x in (arch_m, side_m) if x] + [m.start()])
        span_end = m.end()
        if after_ok:
            window = text[m.end() : hi_after]
            if not arch_m and (am := _ARCH_RE.search(window)):
                arch_m = am
                span_end = max(span_end, m.end() + am.end())
            if not side_m and (sm := _SIDE_RE.search(window)):
                side_m = sm
                span_end = max(span_end, m.end() + sm.end())
        prev_end = m.end()
        arch = _ARCH_WORDS[arch_m.group(1).lower()] if arch_m else None
        side = side_m.group(1).lower() if side_m else None
        both = bool(_BOTH_RE.search(before))
        max_pos = 5 if deciduous else 8
        if position > max_pos:
            continue
        offset = 4 if deciduous else 0
        arches = [arch] if arch else (["upper", "lower"])
        sides = [side] if side else (["right", "left"])
        codes = [f"{_QUADRANT[(a, s)] + offset}{position}" for a in arches for s in sides]
        if arch and side:
            out.teeth.append(ToothRef(codes[0], span_start, span_end, "name"))
        elif both and (arch or side):
            for c in codes:
                out.teeth.append(ToothRef(c, span_start, span_end, "name"))
        else:
            missing = tuple(x for x, v in (("arch", arch), ("side", side)) if v is None)
            out.partial.append(PartialToothRef(text[span_start:span_end], tuple(codes), missing, span_start, span_end))
        taken.append((span_start, span_end))

    # 1b) Palmer-style speech: "lower right eight", "upper left 3", "UL6".
    for m in _PALMER_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        if m.group("q"):
            q = {"ur": 1, "ul": 2, "ll": 3, "lr": 4}[m.group("q").lower()]
        else:
            q = _QUADRANT[(_ARCH_WORDS[m.group("arch").lower()], m.group("side").lower())]
        pos = m.group("pos").lower()
        pos_n = int(pos) if pos.isdigit() else _PALMER_WORDS[pos]
        out.teeth.append(ToothRef(f"{q}{pos_n}", m.start(), m.end(), "palmer"))
        taken.append((m.start(), m.end()))

    # 2) Numbered teeth (FDI): "tooth 48", "on 36", "38 and 48".
    for m in _NUMBER_RE.finditer(text):
        code = m.group(1)
        if not valid_fdi(code):
            continue
        before = text[max(0, m.start() - 25) : m.start()]
        after = text[m.end() : m.end() + 15]
        if _NOT_TOOTH_AFTER.match(after):
            continue
        if _NOT_TOOTH_BEFORE.search(before) and not _TOOTH_CUE_BEFORE.search(before):
            continue
        out.teeth.append(ToothRef(code, m.start(), m.end(), "number"))

    # 3) Anatomical regions.
    for name, pat in _REGION_RES:
        for m in pat.finditer(text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            side_m = _nearest_word(_SIDE_RE, text, max(0, m.start() - 20), m.start())
            out.regions.append(RegionRef(name, side_m.group(1).lower() if side_m else None, m.start(), m.end()))
            taken.append((m.start(), m.end()))

    out.teeth.sort(key=lambda t: t.start)
    return out


# Phrases a clinician might use in a *question* about a tooth group.
TOOTH_GROUPS = {
    "wisdom": ["18", "28", "38", "48"],
    "third molar": ["18", "28", "38", "48"],
}
