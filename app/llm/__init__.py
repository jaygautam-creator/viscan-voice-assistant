"""LLM providers behind one small interface (see base.py).

`build_llm(settings)` returns a Gemini or Claude client wrapped in a circuit
breaker, or None for the offline rules mode.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.base import LLM, PRICING, JsonResult, LLMError, LLMUsage, ToolCallResult
from app.llm.breaker import CircuitBreakerLLM
from app.llm.claude import ClaudeLLM
from app.llm.gemini import GeminiLLM

log = logging.getLogger("viscan.llm")

__all__ = [
    "LLM",
    "LLMError",
    "LLMUsage",
    "ToolCallResult",
    "JsonResult",
    "PRICING",
    "ClaudeLLM",
    "GeminiLLM",
    "CircuitBreakerLLM",
    "build_llm",
]


def build_llm(settings: Settings) -> LLM | None:
    try:
        if settings.llm_provider == "anthropic":
            return CircuitBreakerLLM(ClaudeLLM(settings))
        if settings.llm_provider == "gemini":
            return CircuitBreakerLLM(GeminiLLM(settings))
    except Exception as e:  # missing credentials / SDK etc. -> run in rules mode
        log.warning("LLM disabled: %s", e)
    return None
