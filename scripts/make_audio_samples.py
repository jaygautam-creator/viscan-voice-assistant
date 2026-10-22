"""Generate the sample voice clips (macOS `say` + `afconvert`) and the mock-STT manifest.

The manifest maps each clip's SHA-256 to the sentence that was spoken, which
is what the deterministic mock STT returns. Run: uv run python scripts/make_audio_samples.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples" / "audio"

# (file stem, text spoken by TTS, transcript a perfect STT would return)
CLIPS = [
    ("01_open_priya_opg", "Open Priya's latest O. P. G. report.", "Open Priya's latest OPG report."),
    ("02_choose_priya_sharma", "Priya Sharma.", "Priya Sharma."),
    ("03_summarize_her_report", "Summarize her report.", "Summarize her report."),
    ("04_question_nerve", "Is the wisdom tooth close to the nerve?", "Is the wisdom tooth close to the nerve?"),
    (
        "05_dictation_single",
        "Lower right third molar is horizontally impacted.",
        "Lower right third molar is horizontally impacted.",
    ),
    (
        "06_dictation_multi",
        "Lower right third molar is horizontally impacted and close to the mandibular canal. "
        "Possible periapical radiolucency on tooth three six. No caries on forty seven.",
        "Lower right third molar is horizontally impacted and close to the mandibular canal. "
        "Possible periapical radiolucency on tooth three six. No caries on forty seven.",
    ),
]


def main() -> None:
    if not shutil.which("say") or not shutil.which("afconvert"):
        raise SystemExit("This generator needs macOS `say` and `afconvert`.")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for stem, spoken, transcript in CLIPS:
        aiff = OUT / f"{stem}.aiff"
        wav = OUT / f"{stem}.wav"
        subprocess.run(["say", "-r", "170", "-o", str(aiff), spoken], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
        aiff.unlink()
        manifest.append(
            {"file": wav.name, "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "transcript": transcript}
        )
        print("wrote", wav.name)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote manifest.json")


if __name__ == "__main__":
    main()
