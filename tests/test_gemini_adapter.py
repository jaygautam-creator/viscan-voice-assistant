"""Gemini adapter and circuit breaker, with a stubbed google-genai client (no network)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from conftest import FakeLLM
from google.genai import errors, types

from app.config import get_settings
from app.features.dictation import extraction_schema
from app.llm import CircuitBreakerLLM, GeminiLLM, LLMError
from app.understanding.interpreter import ROUTER_TOOLS


class StubModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _resp(finish="STOP", function_calls=None, text=None):
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=types.FinishReason(finish))],
        function_calls=function_calls,
        text=text,
        prompt_feedback=None,
        response_id="resp_1",
        usage_metadata=SimpleNamespace(
            prompt_token_count=1900, cached_content_token_count=1000, candidates_token_count=50, thoughts_token_count=30
        ),
    )


def _llm(outcome) -> tuple[GeminiLLM, StubModels]:
    settings = dataclasses.replace(get_settings(), llm_provider="gemini", model="gemini-3.8-flash", gemini_api_key="x")
    models = StubModels(outcome)
    return GeminiLLM(settings, client=SimpleNamespace(models=models)), models


def test_tool_call_uses_forced_function_calling_and_parses_args():
    call = SimpleNamespace(name="open_report", args={"patient_reference": "named", "patient_name": "Priya"})
    llm, models = _llm(_resp(function_calls=[call]))
    res = llm.tool_call(system="SYS", user="U", tools=ROUTER_TOOLS, effort="low")
    assert res.name == "open_report" and res.input["patient_name"] == "Priya"
    cfg = models.calls[0]["config"]
    assert models.calls[0]["model"] == "gemini-3.8-flash"
    assert cfg.tool_config.function_calling_config.mode.value == "ANY"
    assert cfg.thinking_config.thinking_level.value == "LOW"
    assert {d.name for d in cfg.tools[0].function_declarations} == {t["name"] for t in ROUTER_TOOLS}
    # cached tokens are billed at the cache rate, thinking tokens as output
    assert res.usage.input_tokens == 900 and res.usage.cache_read_tokens == 1000 and res.usage.output_tokens == 80
    assert res.usage.cost_usd() is not None


def test_json_output_uses_response_schema():
    llm, models = _llm(_resp(text='{"findings": [], "recommendations": [], "unmapped_segments": []}'))
    res = llm.json_output(system="S", user="U", schema=extraction_schema(), effort="medium")
    assert res.data["findings"] == []
    cfg = models.calls[0]["config"]
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema["type"] == "object"


@pytest.mark.parametrize(
    "outcome, kind",
    [
        (_resp(finish="SAFETY"), "refusal"),
        (_resp(finish="MAX_TOKENS"), "truncated"),
        (_resp(finish="MALFORMED_FUNCTION_CALL"), "invalid_tool_args"),
        (_resp(function_calls=[]), "no_tool_call"),
        (
            errors.ClientError(
                429,
                {
                    "error": {
                        "code": 429,
                        "message": "quota",
                        "status": "RESOURCE_EXHAUSTED",
                        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "120s"}],
                    }
                },
            ),
            "rate_limited",
        ),
        (
            errors.ServerError(503, {"error": {"code": 503, "message": "unavailable", "status": "UNAVAILABLE"}}),
            "api_503",
        ),
    ],
)
def test_failures_map_to_llm_errors(outcome, kind):
    llm, _ = _llm(outcome)
    with pytest.raises(LLMError) as e:
        llm.tool_call(system="S", user="U", tools=ROUTER_TOOLS, effort="low")
    assert e.value.kind == kind
    if kind == "rate_limited":
        assert e.value.retry_after_s == 120.0


def test_circuit_breaker_fails_fast_after_rate_limit():
    inner = FakeLLM(fail="rate_limited")
    breaker = CircuitBreakerLLM(inner)
    with pytest.raises(LLMError) as e:
        breaker.tool_call(system="S", user="U", tools=[], effort="low")
    assert e.value.kind == "rate_limited"
    with pytest.raises(LLMError) as e:
        breaker.json_output(system="S", user="U", schema={}, effort="low")
    assert e.value.kind == "circuit_open"
    assert len(inner.calls) == 1  # the provider was not called again
