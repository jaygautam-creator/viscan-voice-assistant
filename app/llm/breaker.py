"""Circuit breaker in front of any provider."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from app.llm.base import LLM, JsonResult, LLMError, ToolCallResult

log = logging.getLogger("viscan.llm")


class CircuitBreakerLLM:
    """Stops calling the provider for a while after rate-limit/auth failures.

    Without it every turn would pay a failed round trip before falling back.
    While open, calls fail fast with LLMError("circuit_open") and the
    orchestrator uses the deterministic path.
    """

    TRIP_KINDS = {"rate_limited", "auth"}

    def __init__(self, inner: LLM, default_cooldown_s: float = 60.0, max_cooldown_s: float = 900.0):
        self.inner = inner
        self.name = inner.name
        self.default_cooldown_s = default_cooldown_s
        self.max_cooldown_s = max_cooldown_s
        self.open_until = 0.0
        self.last_kind: str | None = None
        self._lock = threading.Lock()

    def _guard(self) -> None:
        remaining = self.open_until - time.time()
        if remaining > 0:
            raise LLMError("circuit_open", f"LLM paused after '{self.last_kind}', retry in {remaining:.0f}s")

    def _trip(self, e: LLMError) -> None:
        if e.kind in self.TRIP_KINDS:
            cooldown = min(e.retry_after_s or self.default_cooldown_s, self.max_cooldown_s)
            with self._lock:
                self.open_until = time.time() + cooldown
                self.last_kind = e.kind
            log.warning("LLM circuit open for %.0fs after %s", cooldown, e.kind)

    def tool_call(self, **kwargs: Any) -> ToolCallResult:
        self._guard()
        try:
            return self.inner.tool_call(**kwargs)
        except LLMError as e:
            self._trip(e)
            raise

    def json_output(self, **kwargs: Any) -> JsonResult:
        self._guard()
        try:
            return self.inner.json_output(**kwargs)
        except LLMError as e:
            self._trip(e)
            raise
