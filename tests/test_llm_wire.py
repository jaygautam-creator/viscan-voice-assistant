"""Contract test for the Anthropic integration without an API key.

The real SDK is pointed at a local HTTP server that records the request and
returns canned Messages API responses. This verifies what we send (model,
strict tools, beta header, fallbacks, effort, structured output schema,
prompt caching) and that we parse tool_use / JSON / refusal responses.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import get_settings
from app.features.dictation import extraction_schema
from app.llm import ClaudeLLM, LLMError
from app.understanding.interpreter import ROUTER_TOOLS


class _Server:
    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                payload = json.dumps(outer.responses.pop(0)).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("request-id", "req_test123")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def _msg(content: list[dict], stop_reason: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 40,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 0,
        },
    }


@pytest.fixture
def wire(monkeypatch):
    srv = _Server()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", srv.url)
    settings = dataclasses.replace(
        get_settings(), llm_provider="anthropic", model="claude-opus-5-5", llm_max_retries=0, llm_timeout_s=5
    )
    yield srv, ClaudeLLM(settings)
    srv.httpd.shutdown()


def test_tool_call_request_and_parsing(wire):
    srv, llm = wire
    srv.responses.append(
        _msg(
            [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "open_report",
                    "input": {
                        "patient_reference": "named",
                        "patient_name": "Priya",
                        "modality": "OPG",
                        "which": "latest",
                        "report_date": None,
                    },
                },
            ],
            "tool_use",
        )
    )
    res = llm.tool_call(system="SYS", user="<utterance>Open Priya's OPG</utterance>", tools=ROUTER_TOOLS, effort="low")
    assert res.name == "open_report" and res.input["patient_name"] == "Priya"
    assert res.usage.cache_read_tokens == 1000 and res.usage.request_id == "req_test123"
    assert res.usage.cost_usd() is not None

    req = srv.requests[0]
    body = req["body"]
    assert req["path"].startswith("/v1/messages")
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "low"}
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert all(t["strict"] is True and t["input_schema"]["additionalProperties"] is False for t in body["tools"])


def test_json_output_request_and_parsing(wire):
    srv, llm = wire
    srv.responses.append(
        _msg(
            [{"type": "text", "text": json.dumps({"findings": [], "recommendations": [], "unmapped_segments": []})}],
            "end_turn",
        )
    )
    res = llm.json_output(system="SYS", user="transcript", schema=extraction_schema(), effort="medium")
    assert res.data == {"findings": [], "recommendations": [], "unmapped_segments": []}
    fmt = srv.requests[0]["body"]["output_config"]
    assert fmt["effort"] == "medium" and fmt["format"]["type"] == "json_schema"


def test_refusal_and_missing_tool_call_raise_llm_error(wire):
    srv, llm = wire
    srv.responses.append(_msg([{"type": "text", "text": ""}], "refusal"))
    with pytest.raises(LLMError) as e:
        llm.tool_call(system="S", user="U", tools=ROUTER_TOOLS, effort="low")
    assert e.value.kind == "refusal"
    srv.responses.append(_msg([{"type": "text", "text": "I think you mean Priya."}], "end_turn"))
    with pytest.raises(LLMError) as e:
        llm.tool_call(system="S", user="U", tools=ROUTER_TOOLS, effort="low")
    assert e.value.kind == "no_tool_call"
