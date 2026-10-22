"""Claude (Anthropic) adapter: strict tool calling + structured outputs, refusal fallback enabled."""

from __future__ import annotations

import json
import time
from typing import Any

from app.config import Settings
from app.llm.base import JsonResult, LLMError, LLMUsage, ToolCallResult


class ClaudeLLM:
    """Claude via the official SDK, with server-side refusal fallback enabled."""

    def __init__(self, settings: Settings):
        import anthropic  # imported lazily so the offline mode has no hard dependency at import time

        self._anthropic = anthropic
        self.settings = settings
        self.model = settings.model
        self.name = f"anthropic:{settings.model}"
        self.client = anthropic.Anthropic(timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries)

    def _create(self, **kwargs: Any):
        a = self._anthropic
        oc = kwargs.get("output_config")
        if oc and oc.get("effort") == "minimal":  # Claude's lowest effort level is "low"
            kwargs["output_config"] = {**oc, "effort": "low"}
        if self.settings.use_server_fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        t0 = time.perf_counter()
        try:
            resp = self.client.beta.messages.create(model=self.model, **kwargs)
        except a.APITimeoutError as e:
            raise LLMError("timeout", str(e)) from e
        except a.RateLimitError as e:
            raise LLMError("rate_limited", str(e)) from e
        except a.AuthenticationError as e:
            raise LLMError("auth", "Anthropic credentials rejected") from e
        except a.BadRequestError as e:
            raise LLMError("bad_request", e.message) from e
        except a.APIStatusError as e:
            raise LLMError(f"api_{e.status_code}", e.message) from e
        except a.APIConnectionError as e:
            raise LLMError("connection", str(e)) from e
        latency = (time.perf_counter() - t0) * 1000
        if resp.stop_reason == "refusal":
            raise LLMError("refusal", "model declined the request")
        if resp.stop_reason == "max_tokens":
            raise LLMError("truncated", "output hit max_tokens")
        u = resp.usage
        usage = LLMUsage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            latency_ms=latency,
            model=self.model,
            request_id=getattr(resp, "_request_id", None),
        )
        return resp, usage

    def _system(self, system: str) -> list[dict]:
        # Stable prefix (tools + system) is marked for prompt caching.
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def tool_call(self, *, system, user, tools, effort, max_tokens=4096) -> ToolCallResult:
        resp, usage = self._create(
            max_tokens=max_tokens,
            system=self._system(system),
            tools=tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": user}],
        )
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        if block is None:
            raise LLMError("no_tool_call", "model answered without calling a tool")
        tool_input = block.input if isinstance(block.input, dict) else json.loads(block.input)
        return ToolCallResult(block.name, tool_input, usage)

    def json_output(self, *, system, user, schema, effort, max_tokens=16000) -> JsonResult:
        resp, usage = self._create(
            max_tokens=max_tokens,
            system=self._system(system),
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            raise LLMError("empty_output")
        try:
            return JsonResult(json.loads(text), usage)
        except json.JSONDecodeError as e:
            raise LLMError("invalid_json", str(e)) from e
