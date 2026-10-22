"""Unit tests for the deterministic parsers the validators rely on."""

import pytest

from app.clinical.dental import fdi_to_universal, parse_sites
from app.clinical.lexicon import is_negated
from app.understanding.interpreter import ContextSummary, RuleInterpreter
from app.understanding.normalize import detect_modality, detect_selector, normalize_utterance


@pytest.mark.parametrize(
    "phrase, fdi",
    [
        ("lower right third molar", "48"),
        ("upper left central incisor", "21"),
        ("mandibular left first molar", "36"),
        ("maxillary right canine", "13"),
        ("third molar on the lower left", "38"),
        ("upper right deciduous canine", "53"),
        ("tooth 46", "46"),
        ("lower right eight", "48"),  # Palmer-style speech
        ("UL6", "26"),
    ],
)
def test_tooth_names_to_fdi(phrase, fdi):
    assert parse_sites(phrase).fdi_codes == [fdi]


def test_universal_numbering():
    assert [fdi_to_universal(c) for c in ("18", "11", "21", "28", "38", "31", "41", "48")] == [
        "1",
        "8",
        "9",
        "16",
        "17",
        "24",
        "25",
        "32",
    ]


def test_numbers_that_are_not_teeth():
    assert parse_sites("lesion measuring approximately 14 mm").fdi_codes == []
    assert parse_sites("seen on 18 September 2026").fdi_codes == []
    assert parse_sites("grade 21 lesion").fdi_codes == []
    assert parse_sites("caries on 15 and 16").fdi_codes == ["15", "16"]


def test_partial_and_bilateral_sites():
    sp = parse_sites("lower third molar")
    assert sp.fdi_codes == [] and set(sp.partial[0].candidates) == {"38", "48"}
    assert set(parse_sites("both lower wisdom teeth").fdi_codes) == {"38", "48"}


def test_stt_normalisation():
    assert normalize_utterance("open the o p g for tooth four eight") == "open the OPG for tooth 48"
    assert normalize_utterance("teeth three eight and forty-eight") == "teeth 38 and 48"
    assert normalize_utterance("radiolucency on tooth 3-6") == "radiolucency on tooth 36"
    assert detect_modality("show the cone beam scan") == "CBCT"
    assert detect_selector("previous OPG") == ("previous", None)
    assert detect_selector("OPG from 18 September 2026") == ("on_date", "2026-09-18")


def test_negation_detection():
    assert is_negated("No caries on 47")
    assert is_negated("periapical pathology is not seen")
    assert not is_negated("pulpal involvement cannot be excluded")
    assert not is_negated("tooth 48 is not erupted")  # part of the condition term
    assert not is_negated("tooth no. 48 is impacted")


def test_rule_interpreter_intents():
    r = RuleInterpreter({"priya", "sharma", "shah"})
    ctx = ContextSummary()
    assert r.interpret("Open Priya's latest OPG report", ctx).intent.action == "open_report"
    assert r.interpret("Summarize her report", ctx).intent.target.reference == "context"
    assert r.interpret("Is the wisdom tooth close to the nerve?", ctx).intent.action == "ask_report"
    assert r.interpret("Lower right third molar is horizontally impacted", ctx).intent.action == "dictate"
    assert r.interpret("Book a cleaning for Friday", ctx).intent.action == "unsupported"
    i = r.interpret("Open Priyas latest OPG", ctx).intent  # STT dropped the apostrophe
    assert i.target.patient_name == "Priya" and i.target.modality == "OPG"
