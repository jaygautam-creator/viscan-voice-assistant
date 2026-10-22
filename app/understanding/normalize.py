"""Text normalisation shared by every input path (typed text and STT output).

Speech-to-text output is messy: "O P G", "tooth four eight", "3rd molar".
Normalising it once, deterministically, means the interpreters and the
validators all see the same string, and evidence quotes can be checked
against it character for character.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_ACRONYMS = [
    (re.compile(r"\bo\.?\s?p\.?\s?g\b\.?", re.I), "OPG"),
    (re.compile(r"\bc\.?\s?b\.?\s?c\.?\s?t\b\.?", re.I), "CBCT"),
    (re.compile(r"\bi\.?\s?o\.?\s?p\.?\s?a\b\.?", re.I), "IOPA"),
    (re.compile(r"\bt\.?\s?m\.?\s?j\b\.?", re.I), "TMJ"),
    (re.compile(r"\bp\.?\s?d\.?\s?l\b\.?", re.I), "PDL"),
    (re.compile(r"\br\.\s?c\.\s?t\b\.?", re.I), "RCT"),
]
_ORDINALS = [
    (re.compile(r"\b1st\b", re.I), "first"),
    (re.compile(r"\b2nd\b", re.I), "second"),
    (re.compile(r"\b3rd\b", re.I), "third"),
]

_DIGITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_TEENS = {
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80}

_D = "|".join(_DIGITS)
_SPOKEN_NUM = (
    rf"(?:(?:{'|'.join(_TENS)})(?:[\s-]+(?:{_D}))?"  # forty eight / forty-eight
    rf"|(?:{'|'.join(_TEENS)})"  # eleven .. eighteen
    rf"|(?:{_D})[\s-]+(?:{_D}))"  # four eight (FDI read digit by digit)
)
# STT often renders "four eight" as "4-8" or "4 8".
_SPLIT_DIGITS = re.compile(r"\b(tooth|teeth|number|#)(\s+)([1-8])[\s\-]([1-8])\b", re.I)
_TOOTH_KEYWORD_NUM = re.compile(rf"\b(tooth|teeth|number|#)(\s+)({_SPOKEN_NUM})\b", re.I)
_LIST_NUM = re.compile(rf"(\b\d{{2}}\s*(?:,|and|,\s*and)\s+)({_SPOKEN_NUM})\b", re.I)


def _spoken_to_int(phrase: str) -> int | None:
    words = re.split(r"[\s-]+", phrase.lower().strip())
    if len(words) == 1:
        w = words[0]
        return _TEENS.get(w) or _TENS.get(w)
    if len(words) == 2:
        a, b = words
        if a in _TENS and b in _DIGITS:
            return _TENS[a] + _DIGITS[b]
        if a in _DIGITS and b in _DIGITS:
            return _DIGITS[a] * 10 + _DIGITS[b]
    return None


def _replace_spoken_teeth(text: str) -> str:
    def kw(m: re.Match) -> str:
        n = _spoken_to_int(m.group(3))
        return f"{m.group(1)}{m.group(2)}{n}" if n else m.group(0)

    def lst(m: re.Match) -> str:
        n = _spoken_to_int(m.group(2))
        return f"{m.group(1)}{n}" if n else m.group(0)

    text = _SPLIT_DIGITS.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}", text)
    text = _TOOTH_KEYWORD_NUM.sub(kw, text)
    for _ in range(6):  # tooth lists: "teeth 38 and four eight"
        new = _LIST_NUM.sub(lst, text)
        if new == text:
            break
        text = new
    return text


def clean_input(text: str) -> str:
    """Unicode-normalise, strip control characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def normalize_utterance(text: str) -> str:
    """Deterministic clean-up for typed or transcribed speech."""
    text = clean_input(text)
    for pat, rep in _ACRONYMS:
        text = pat.sub(rep, text)
    for pat, rep in _ORDINALS:
        text = pat.sub(rep, text)
    text = _replace_spoken_teeth(text)
    return text


# --------------------------------------------------------------------------
# Report-request vocabulary (modality, which report, dates)
# --------------------------------------------------------------------------

MODALITIES = ("OPG", "CBCT", "IOPA", "BITEWING", "CEPH")

_MODALITY_PATTERNS = {
    "OPG": re.compile(r"\b(opg|orthopantomo\w*|panoramic|pano|panorex|dpt)\b", re.I),
    "CBCT": re.compile(r"\b(cbct|cone[- ]beam|3-?d scan|ct scan)\b", re.I),
    "IOPA": re.compile(
        r"\b(iopa|intra-?oral periapical|periapical (?:x-?ray|radiograph|film|view|image)|pa (?:x-?ray|film|view))\b",
        re.I,
    ),
    "BITEWING": re.compile(r"\b(bite[- ]?wings?|bwx)\b", re.I),
    "CEPH": re.compile(r"\b(ceph|cephalo\w*)\b", re.I),
}


def detect_modality(text: str) -> str | None:
    for mod, pat in _MODALITY_PATTERNS.items():
        if pat.search(text):
            return mod
    return None


_LATEST = re.compile(r"\b(latest|most recent|newest|recent|current|last)\b", re.I)
_EARLIEST = re.compile(
    r"\b(earliest|oldest|baseline|original)\b"
    r"|\bfirst\s+(?:\w+\s+){0,2}?(report|scan|x-?ray|opg|cbct|iopa|bitewing|ceph|radiograph|study)\b",
    re.I,
)
_PREVIOUS = re.compile(r"\b(previous|prior|older|one before|last but one|penultimate|second[- ]latest)\b", re.I)

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)
_MDY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_MY = re.compile(rf"\b(?:in|from|of|dated)?\s*({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)


def detect_date(text: str) -> str | None:
    """Return 'YYYY-MM-DD' or 'YYYY-MM' if the text names a date."""
    if m := _ISO_DATE.search(text):
        return m.group(0)
    if m := _DMY.search(text):
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    if m := _MDY.search(text):
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    if m := _MY.search(text):
        return f"{int(m.group(2)):04d}-{_MONTHS[m.group(1).lower()]:02d}"
    return None


def detect_selector(text: str) -> tuple[str, str | None]:
    """Which report the user wants: latest / earliest / previous / on_date."""
    date = detect_date(text)
    if date:
        return "on_date", date
    if _PREVIOUS.search(text):
        return "previous", None
    if _EARLIEST.search(text):
        return "earliest", None
    return "latest", None


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
