"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "gemini": "gemini-3.8-flash", "none": ""}


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "anthropic" | "gemini" | "none"
    model: str
    gemini_api_key: str | None
    router_effort: str
    extraction_effort: str
    llm_timeout_s: float
    llm_max_retries: int
    use_server_fallbacks: bool
    stt_provider: str  # "mock" | "local" | "openai"
    whisper_model: str
    openai_stt_model: str
    data_dir: Path
    default_clinician: str
    max_utterance_chars: int
    max_transcript_chars: int
    max_audio_bytes: int
    session_ttl_s: int
    audit_log_path: Path | None
    booking_days_ahead: int = 14


@lru_cache
def get_settings() -> Settings:
    _load_dotenv(ROOT / ".env")
    env = os.environ.get
    gemini_key = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY") or None
    provider = env("VISCAN_LLM_PROVIDER", "auto").lower()
    if provider == "auto":
        if env("ANTHROPIC_API_KEY") or env("ANTHROPIC_AUTH_TOKEN"):
            provider = "anthropic"
        elif gemini_key:
            provider = "gemini"
        else:
            provider = "none"
    audit = env("VISCAN_AUDIT_LOG", str(ROOT / "logs" / "audit.jsonl"))
    return Settings(
        llm_provider=provider,
        model=env("VISCAN_MODEL") or DEFAULT_MODELS.get(provider, ""),
        gemini_api_key=gemini_key,
        router_effort=env("VISCAN_ROUTER_EFFORT", "low"),
        extraction_effort=env("VISCAN_EXTRACTION_EFFORT", "medium"),
        llm_timeout_s=float(env("VISCAN_LLM_TIMEOUT_S", "30")),
        llm_max_retries=int(env("VISCAN_LLM_MAX_RETRIES", "1")),
        use_server_fallbacks=env("VISCAN_LLM_FALLBACKS", "true").lower() == "true",
        stt_provider=env("VISCAN_STT_PROVIDER", "mock").lower(),
        whisper_model=env("VISCAN_WHISPER_MODEL", "base.en"),
        openai_stt_model=env("VISCAN_OPENAI_STT_MODEL", "gpt-4o-transcribe"),
        data_dir=Path(env("VISCAN_DATA_DIR", str(ROOT / "data"))),
        default_clinician=env("VISCAN_DEMO_CLINICIAN", "dr-anand"),
        max_utterance_chars=int(env("VISCAN_MAX_UTTERANCE_CHARS", "1000")),
        max_transcript_chars=int(env("VISCAN_MAX_TRANSCRIPT_CHARS", "5000")),
        max_audio_bytes=int(env("VISCAN_MAX_AUDIO_BYTES", str(10 * 1024 * 1024))),
        session_ttl_s=int(env("VISCAN_SESSION_TTL_S", "1800")),
        audit_log_path=Path(audit) if audit else None,
    )
