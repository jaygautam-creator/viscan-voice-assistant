"""Gemini (Google) adapter: forced function calling + JSON-schema output, thinking levels per route."""

from __future__ import annotations

import json
import time
from typing import Any

from app.config import Settings
from app.llm.base import JsonResult, LLMError, LLMUsage, ToolCallResult


class GeminiLLM:
    """Gemini via the official google-genai SDK.

    Same contract as ClaudeLLM: tools are passed as JSON-schema function
    declarations with forced function calling (mode=ANY); JSON outputs use
    response_json_schema. Effort maps to Gemini thinking levels.
    """

    _LEVELS = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}
    _BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}

    def __init__(self, settings: Settings, client: Any = None):
        from google import genai
        from google.genai import errors, types

        self._types, self._errors = types, errors
        self.settings = settings
        self.model = settings.model
        self.name = f"gemini:{settings.model}"
        self.client = client or genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_s * 1000),
                # 429s are not retried here: a quota error rarely clears in seconds, so the
                # circuit breaker takes over instead (see CircuitBreakerLLM).
                retry_options=types.HttpRetryOptions(
                    attempts=settings.llm_max_retries + 1, http_status_codes=[408, 500, 502, 503, 504]
                ),
            ),
        )

    def _generate(self, *, system: str, user: str, effort: str, max_tokens: int, **cfg: Any):
        types, errors = self._types, self._errors
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self._LEVELS.get(effort, "LOW")),
            **cfg,
        )
        t0 = time.perf_counter()
        try:
            resp = self.client.models.generate_content(model=self.model, contents=user, config=config)
        except errors.ClientError as e:
            kind = {429: "rate_limited", 401: "auth", 403: "auth", 400: "bad_request"}.get(e.code, f"api_{e.code}")
            raise LLMError(kind, str(e)[:300], _gemini_retry_after(e)) from e
        except errors.ServerError as e:
            raise LLMError(f"api_{e.code}", str(e)[:300]) from e
        except errors.APIError as e:
            raise LLMError(f"api_{getattr(e, 'code', 'error')}", str(e)[:300]) from e
        except Exception as e:  # httpx timeouts / connection errors
            kind = "timeout" if "timeout" in type(e).__name__.lower() else "connection"
            raise LLMError(kind, f"{type(e).__name__}: {e}"[:300]) from e
        latency = (time.perf_counter() - t0) * 1000
        if not resp.candidates:
            raise LLMError("refusal", f"prompt blocked: {getattr(resp.prompt_feedback, 'block_reason', None)}")
        finish = getattr(resp.candidates[0].finish_reason, "value", str(resp.candidates[0].finish_reason))
        if finish in self._BLOCKED:
            raise LLMError("refusal", f"finish_reason={finish}")
        if finish == "MAX_TOKENS":
            raise LLMError("truncated", "output hit max_output_tokens")
        if finish in ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"):
            raise LLMError("invalid_tool_args", f"finish_reason={finish}")
        u = resp.usage_metadata
        prompt = (u.prompt_token_count or 0) if u else 0
        cached = (u.cached_content_token_count or 0) if u else 0
        usage = LLMUsage(
            input_tokens=prompt - cached,
            output_tokens=((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0,
            cache_read_tokens=cached,
            latency_ms=latency,
            model=self.model,
            request_id=getattr(resp, "response_id", None),
        )
        return resp, usage

    def tool_call(self, *, system, user, tools, effort, max_tokens=4096) -> ToolCallResult:
        types = self._types
        decls = [
            types.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"]
            )
            for t in tools
        ]
        resp, usage = self._generate(
            system=system,
            user=user,
            effort=effort,
            max_tokens=max_tokens,
            tools=[types.Tool(function_declarations=decls)],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        calls = resp.function_calls or []
        if not calls:
            raise LLMError("no_tool_call", "model answered without calling a tool")
        return ToolCallResult(calls[0].name, dict(calls[0].args or {}), usage)

    def json_output(self, *, system, user, schema, effort, max_tokens=16000) -> JsonResult:
        resp, usage = self._generate(
            system=system,
            user=user,
            effort=effort,
            max_tokens=max_tokens,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        if not resp.text:
            raise LLMError("empty_output")
        try:
            return JsonResult(json.loads(resp.text), usage)
        except json.JSONDecodeError as e:
            raise LLMError("invalid_json", str(e)) from e


def _gemini_retry_after(e: Any) -> float | None:
    """Read google.rpc.RetryInfo.retryDelay (e.g. "60746s") from a Gemini error."""
    try:
        for d in (e.details or {}).get("error", {}).get("details", []):
            if d.get("@type", "").endswith("RetryInfo") and d.get("retryDelay", "").endswith("s"):
                return float(d["retryDelay"][:-1])
    except Exception:
        pass
    return None
