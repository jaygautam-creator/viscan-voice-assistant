"""Controlled clinical vocabulary used for extraction *and* for grounding checks.

Every condition the system can put in a draft has a key, a display label and
the surface terms that must appear in the dictation for that condition to be
considered "said". The LLM picks from these keys (or "other"); the validator
then checks that one of the condition's terms really occurs in the quoted
evidence. That check is what stops invented diagnoses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    pattern: str  # regex of surface terms (word-bounded at use)


CONDITIONS: list[Condition] = [
    Condition("impaction", "Impaction", r"impacted|impaction"),
    Condition("partial_eruption", "Partially erupted / unerupted", r"partially erupted|unerupted|not erupted|erupting"),
    Condition("caries", "Caries", r"caries|carious|decay(?:ed)?|cavit(?:y|ies|ation)"),
    Condition(
        "periapical_lesion",
        "Periapical lesion",
        r"periapical (?:radiolucenc(?:y|ies)|lesions?|pathology|abscess|granuloma|rarefaction|infection|changes?)"
        r"|apical periodontitis|pa lesion|(?:apical|periradicular) radiolucenc(?:y|ies)|radiolucenc(?:y|ies) at the (?:root )?apex",
    ),
    Condition(
        "bone_loss",
        "Alveolar bone loss",
        r"bone loss|loss of (?:alveolar )?bone|bone defects?|vertical defects?|crestal bone loss",
    ),
    Condition("furcation", "Furcation involvement", r"furcation(?: involvement| radiolucency| defect)?"),
    Condition(
        "widened_pdl",
        "Widened PDL space",
        r"widen(?:ed|ing)(?: of the)? (?:periodontal ligament|pdl)(?: space)?|pdl widening",
    ),
    Condition("calculus", "Calculus", r"calculus|tartar"),
    Condition("missing_tooth", "Missing tooth", r"missing|edentulous|extracted|absent"),
    Condition(
        "root_canal_treated",
        "Root canal treated",
        r"root canal (?:treated|treatment|filling|filled)|endodontically treated|\brct\b|obturat(?:ed|ion)|root[- ]filled",
    ),
    Condition("restoration", "Restoration", r"restorations?|restored|fillings?|amalgam|composite|inlay|onlay"),
    Condition(
        "crown_prosthesis",
        "Crown (prosthetic)",
        r"(?:full[- ]coverage|prosthetic|pfm|zirconia|ceramic|metal|porcelain) crown|crowned",
    ),
    Condition("implant", "Implant", r"implants?"),
    Condition("fracture", "Fracture", r"fractur(?:e|ed|es)|cracked|crack"),
    Condition("root_resorption", "Root resorption", r"(?:root )?resorption|resorbed"),
    Condition(
        "ian_proximity",
        "Relation to inferior alveolar canal",
        r"(?:close|near|direct|intimate)\s+(?:proximity|relation(?:ship)?|contact)"
        r"|(?:abutting|touching|contacting|close to|near|approaching|encroaching on|in contact with|superimposed on|overlapping)"
        r"\s+the\s+(?:inferior alveolar|mandibular|ian|nerve)",
    ),
    Condition("pericoronitis", "Pericoronitis", r"pericoronitis"),
    Condition("cyst", "Cyst / cystic lesion", r"cysts?|cystic"),
    Condition("abscess", "Abscess", r"abscess"),
    Condition("sinus_finding", "Sinus finding", r"mucosal thickening|sinusitis|opacif(?:ied|ication)|polyps?"),
    Condition(
        "tipping_drift",
        "Tipping / drifting",
        r"tipp(?:ing|ed)|drift(?:ed|ing)|supra-?erupt(?:ed|ion)|over-?erupt(?:ed|ion)|extruded",
    ),
    Condition("retained_deciduous", "Retained deciduous tooth", r"retained"),
    Condition("supernumerary", "Supernumerary tooth", r"supernumerary|mesiodens"),
    Condition("rotation", "Rotation", r"rotated|rotation"),
    Condition("crowding", "Crowding", r"crowding|crowded"),
    Condition("tooth_wear", "Tooth wear", r"attrition|abrasion|erosion|tooth wear"),
    Condition("mobility", "Mobility", r"mobile|mobility"),
    Condition("radiolucency", "Radiolucency (unspecified)", r"radiolucen(?:cy|cies|t)"),
    Condition("radiopacity", "Radiopacity", r"radiopa(?:city|cities|que)"),
]
CONDITION_BY_KEY = {c.key: c for c in CONDITIONS}
_CONDITION_RES = [(c, re.compile(rf"\b(?:{c.pattern})\b", re.I)) for c in CONDITIONS]

NORMAL_KEY = "normal_finding"
NORMAL_LABEL = "Normal / no abnormality"
_NORMAL_RE = re.compile(
    r"\b(within normal limits|wnl|normal|unremarkable|appears? clear|are clear|is clear|intact|healthy|adequate|no abnormality)\b",
    re.I,
)

HEDGE_RE = re.compile(
    r"\b(possible|possibly|probable|probably|likely|suspected|suspect|suspicious|suggestive of|suggests?|query|"
    r"questionable|equivocal|doubtful|may|might|could|appears?|seems?|cannot be (?:excluded|ruled out)|"
    r"can'?t (?:exclude|rule out)|rule out|r/o|not sure|unsure|i think|maybe|perhaps|consistent with|indeterminate|"
    r"borderline|cannot be excluded)\b|\?",
    re.I,
)
# Hedge phrases that contain a negation word must not be read as negation.
_HEDGED_NEGATION = re.compile(r"cannot be (?:excluded|ruled out)|can'?t (?:exclude|rule out)|not sure|rule out", re.I)
NEGATION_RE = re.compile(r"\b(no|not|without|negative for|free of|absence of|ruled out|nil|none|neither|nor)\b", re.I)
RECOMMENDATION_RE = re.compile(
    r"\b(advise[ds]?|advised|recommend(?:ed|s|ation)?|refer(?:ral|red)?|suggest(?:ed)?|plan(?:ned)? (?:for|to)|"
    r"consider|schedule|review in|follow[- ]?up|to be extracted|needs? (?:extraction|rct|treatment)|"
    r"should be|indicated)\b",
    re.I,
)

ATTRIBUTE_PATTERNS: dict[str, re.Pattern] = {
    "orientation": re.compile(
        r"\b(horizontal(?:ly)?|vertical(?:ly)?|mesio-?angular(?:ly)?|disto-?angular(?:ly)?|buccal(?:ly)?|"
        r"lingual(?:ly)?|palatal(?:ly)?|transverse(?:ly)?|inverted)\b",
        re.I,
    ),
    "severity": re.compile(
        r"\b(mild|moderate|severe|deep|shallow|superficial|early|advanced|gross|extensive|minimal)\b", re.I
    ),
    "surface": re.compile(
        r"\b(mesial|distal|occlusal|incisal|labial|cervical|proximal|interproximal|mesio-?occlusal|disto-?occlusal|mod)\b",
        re.I,
    ),
    "extent": re.compile(r"\b(generali[sz]ed|locali[sz]ed|bilateral|unilateral|multiple|isolated)\b", re.I),
    "size": re.compile(r"\b(\d+(?:\.\d+)?\s?(?:mm|millimet(?:er|re)s?|cm))\b", re.I),
    "grade": re.compile(r"\b((?:grade|class|stage)\s+(?:i{1,3}|iv|[1-4]))\b", re.I),
}
ATTRIBUTE_NAMES = list(ATTRIBUTE_PATTERNS)


@dataclass(frozen=True)
class TermMatch:
    key: str
    start: int
    end: int
    text: str


def find_conditions(text: str) -> list[TermMatch]:
    """All condition mentions, longest-match-wins on overlaps."""
    raw: list[TermMatch] = []
    for cond, rx in _CONDITION_RES:
        for m in rx.finditer(text):
            raw.append(TermMatch(cond.key, m.start(), m.end(), m.group(0)))
    raw.sort(key=lambda t: (-(t.end - t.start), t.start))
    kept: list[TermMatch] = []
    for t in raw:
        if any(not (t.end <= k.start or t.start >= k.end) for k in kept):
            continue
        kept.append(t)
    # "absent" is a negation cue when another condition is present ("caries is absent").
    if len(kept) > 1:
        kept = [k for k in kept if not (k.key == "missing_tooth" and k.text.lower() == "absent")]
    # A generic radiolucency inside a more specific phrase was already dropped by overlap.
    kept.sort(key=lambda t: t.start)
    return kept


def condition_supported(key: str, evidence: str) -> bool:
    if key == NORMAL_KEY:
        return bool(_NORMAL_RE.search(evidence))
    cond = CONDITION_BY_KEY.get(key)
    if cond is None:
        return False
    return re.search(rf"\b(?:{cond.pattern})\b", evidence, re.I) is not None


def has_normal_cue(text: str) -> bool:
    return bool(_NORMAL_RE.search(text))


def hedge_terms(text: str) -> list[str]:
    return [m.group(0) for m in HEDGE_RE.finditer(text)]


def is_negated(text: str) -> bool:
    """True when the clause denies a finding ("no caries", "caries is absent").

    Negation words that are part of a condition term ("not erupted") or of a
    hedge ("cannot be excluded"), and "tooth no. 48", are ignored.
    """
    s = _HEDGED_NEGATION.sub(" ", text)
    s = re.sub(r"\bno\.?\s+(?=\d)", " ", s, flags=re.I)
    chars = list(s)
    for c in find_conditions(s):
        chars[c.start : c.end] = " " * (c.end - c.start)
    s = "".join(chars)
    # "absent" survives blanking only when it negates another condition.
    if re.search(r"\babsent\b", s, re.I):
        return True
    return bool(NEGATION_RE.search(s))


def extract_attributes(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name, rx in ATTRIBUTE_PATTERNS.items():
        for m in rx.finditer(text):
            val = m.group(1).lower()
            if (name, val) not in out:
                out.append((name, val))
    return out


def label(key: str) -> str:
    if key == NORMAL_KEY:
        return NORMAL_LABEL
    if key == "other":
        return "Unclassified observation"
    return CONDITION_BY_KEY[key].label


# Lay / colloquial words a clinician may use in a *question*, mapped to keys.
QUESTION_SYNONYMS: dict[str, list[str]] = {
    "ian_proximity": [r"nerve", r"canal", r"ian", r"paresthesia", r"numbness"],
    "periapical_lesion": [r"infection", r"infected", r"apical", r"periapical", r"lesion", r"abscess", r"radiolucen\w*"],
    "bone_loss": [r"bone", r"gum disease", r"periodont\w*", r"perio"],
    "caries": [r"cavit\w*", r"decay", r"caries", r"carious"],
    "impaction": [r"impact\w*", r"wisdom", r"third molar"],
    "root_canal_treated": [r"root canal", r"rct", r"endo\w*"],
    "missing_tooth": [r"missing", r"absent", r"gap", r"edentulous"],
    "restoration": [r"filling", r"restor\w*", r"amalgam", r"composite"],
    "fracture": [r"fractur\w*", r"crack\w*", r"broken"],
    "sinus_finding": [r"sinus\w*"],
    "root_resorption": [r"resorption", r"resorb\w*"],
    "calculus": [r"calculus", r"tartar", r"scaling"],
    "furcation": [r"furcation"],
}
