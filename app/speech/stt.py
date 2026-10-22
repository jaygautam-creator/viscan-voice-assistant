"""Speech-to-text providers behind one interface.

* mock   - deterministic: recognises the bundled sample clips by SHA-256.
           Lets the audio path run in CI with no model or API key.
* local  - faster-whisper on CPU (pip extra `stt-local`); audio stays on the machine.
* openai - OpenAI transcription API (pip extra `stt-openai`).

All providers receive a short dental vocabulary prompt to bias recognition
of terms such as "OPG", "periapical" and "mesioangular".
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from app.config import ROOT, Settings

DENTAL_PROMPT = (
    "Dental radiology dictation. OPG, CBCT, IOPA, bitewing, cephalogram. Lower right third molar, "
    "horizontally impacted, mesioangular, distoangular, periapical radiolucency, inferior alveolar canal, "
    "caries, furcation, root canal treated, alveolar bone loss. FDI tooth numbers such as 48, 36, 21."
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".oga", ".flac", ".aiff", ".aif"}


class STTError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class Transcript(BaseModel):
    text: str
    provider: str
    model: str | None = None
    latency_ms: float
    audio_sha256: str


class STTProvider(Protocol):
    name: str

    def transcribe(self, audio: bytes, filename: str) -> Transcript: ...


def _sha(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()


class MockSTT:
    name = "mock"

    def __init__(self, manifest: Path = ROOT / "samples" / "audio" / "manifest.json"):
        self.by_hash: dict[str, str] = {}
        if manifest.is_file():
            for item in json.loads(manifest.read_text()):
                self.by_hash[item["sha256"]] = item["transcript"]

    def transcribe(self, audio: bytes, filename: str) -> Transcript:
        t0 = time.perf_counter()
        digest = _sha(audio)
        text = self.by_hash.get(digest)
        if text is None:
            raise STTError(
                "Mock STT only recognises the bundled sample clips. Set VISCAN_STT_PROVIDER=local "
                "(faster-whisper) or openai for real transcription, or type the text.",
                status_code=422,
            )
        return Transcript(
            text=text,
            provider=self.name,
            model="sample-manifest",
            latency_ms=(time.perf_counter() - t0) * 1000,
            audio_sha256=digest,
        )


class LocalWhisperSTT:
    name = "local"

    def __init__(self, model_size: str):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise STTError("faster-whisper is not installed (uv sync --extra stt-local).", 500) from e
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    @staticmethod
    def _decode_16k_mono(path: str):
        """Decode any container (wav/webm/mp3/m4a...) to 16 kHz mono float32 with PyAV."""
        import av
        import numpy as np

        chunks = []
        with av.open(path) as container:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
            chunks.extend(f.to_ndarray() for f in resampler.resample(None))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=1).flatten().astype(np.float32) / 32768.0

    def transcribe(self, audio: bytes, filename: str) -> Transcript:
        model = self._load()
        t0 = time.perf_counter()
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(audio)
            f.flush()
            try:
                samples = self._decode_16k_mono(f.name)
                segments, _info = model.transcribe(
                    samples, language="en", initial_prompt=DENTAL_PROMPT, vad_filter=True, beam_size=5
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:  # decoding errors, corrupt audio
                raise STTError(f"Could not transcribe audio: {type(e).__name__}: {e}", 422) from e
        if not text:
            raise STTError("No speech detected in the audio.", 422)
        return Transcript(
            text=text,
            provider=self.name,
            model=self.model_size,
            latency_ms=(time.perf_counter() - t0) * 1000,
            audio_sha256=_sha(audio),
        )


class OpenAISTT:
    name = "openai"

    def __init__(self, model: str):
        self.model = model

    def transcribe(self, audio: bytes, filename: str) -> Transcript:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise STTError("openai package not installed (uv sync --extra stt-openai).", 500) from e
        t0 = time.perf_counter()
        try:
            resp = OpenAI(timeout=30).audio.transcriptions.create(
                model=self.model, file=(filename, audio), prompt=DENTAL_PROMPT, language="en"
            )
        except Exception as e:
            raise STTError(f"Transcription service error: {type(e).__name__}") from e
        return Transcript(
            text=resp.text.strip(),
            provider=self.name,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
            audio_sha256=_sha(audio),
        )


def build_stt(settings: Settings) -> STTProvider:
    if settings.stt_provider == "local":
        return LocalWhisperSTT(settings.whisper_model)
    if settings.stt_provider == "openai":
        return OpenAISTT(settings.openai_stt_model)
    return MockSTT()


def validate_audio(audio: bytes, filename: str, max_bytes: int) -> None:
    if not audio:
        raise STTError("Empty audio upload.", 422)
    if len(audio) > max_bytes:
        raise STTError(f"Audio exceeds {max_bytes // (1024 * 1024)} MB.", 413)
    if Path(filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        raise STTError(f"Unsupported audio type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", 415)
