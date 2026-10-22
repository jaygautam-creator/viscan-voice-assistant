"""Provider-neutral LLM contract.

The rest of the app sees two calls, `tool_call` and `json_output`, and one
exception type, `LLMError`. Every provider failure (timeout, rate limit,
refusal, malformed output) becomes an `LLMError`, so callers can fall back to
the deterministic path instead of failing the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# USD per million tokens (input, output). Used only for the per-turn cost estimate in traces.
# Gemini output prices include thinking tokens. Gemini 3.x Flash prices are the 2026 rates
# (they double from 2027-01-01 per ai.google.dev/gemini-api/docs/pricing).
PRICING = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}


class LLMError(Exception):
    def __init__(self, kind: str, message: str = "", retry_after_s: float | None = None):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind
        self.retry_after_s = retry_after_s


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    request_id: str | None = None

    def cost_usd(self) -> float | None:
        price = PRICING.get(self.model)
        if not price:
            return None
        pin, pout = price
        billed_in = self.input_tokens + 1.25 * self.cache_write_tokens + 0.1 * self.cache_read_tokens
        return round((billed_in * pin + self.output_tokens * pout) / 1_000_000, 6)

    def as_trace(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "est_cost_usd": self.cost_usd(),
            "request_id": self.request_id,
        }


@dataclass
class ToolCallResult:
    name: str
    input: dict[str, Any]
    usage: LLMUsage = field(default_factory=LLMUsage)


@dataclass
class JsonResult:
    data: dict[str, Any]
    usage: LLMUsage = field(default_factory=LLMUsage)


class LLM(Protocol):
    name: str

    def tool_call(
        self, *, system: str, user: str, tools: list[dict], effort: str, max_tokens: int = 4096
    ) -> ToolCallResult: ...

    def json_output(
        self, *, system: str, user: str, schema: dict, effort: str, max_tokens: int = 16000
    ) -> JsonResult: ...
