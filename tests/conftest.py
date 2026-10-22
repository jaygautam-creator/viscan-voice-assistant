from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import JsonResult, LLMError, LLMUsage, ToolCallResult  # noqa: E402
from app.main import create_app  # noqa: E402
from app.speech.stt import MockSTT  # noqa: E402


class FakeLLM:
    """Scripted stand-in for Claude. Returns queued tool calls / JSON payloads,
    or raises LLMError(fail) to simulate outages."""

    name = "fake-llm"

    def __init__(
        self,
        tool_calls: list[tuple[str, dict]] | None = None,
        json_outputs: list[dict | Callable[[str], dict]] | None = None,
        fail: str | None = None,
    ):
        self.tool_calls = list(tool_calls or [])
        self.json_outputs = list(json_outputs or [])
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def tool_call(
        self, *, system: str, user: str, tools: list[dict], effort: str, max_tokens: int = 4096
    ) -> ToolCallResult:
        self.calls.append(("tool", user))
        if self.fail:
            raise LLMError(self.fail)
        if not self.tool_calls:
            raise LLMError("no_tool_call")
        name, inp = self.tool_calls.pop(0)
        return ToolCallResult(name, inp, LLMUsage(model="fake"))

    def json_output(self, *, system: str, user: str, schema: dict, effort: str, max_tokens: int = 16000) -> JsonResult:
        self.calls.append(("json", user))
        if self.fail:
            raise LLMError(self.fail)
        if not self.json_outputs:
            raise LLMError("empty_output")
        out: Any = self.json_outputs.pop(0)
        return JsonResult(out(user) if callable(out) else out, LLMUsage(model="fake"))


@pytest.fixture
def settings(tmp_path):
    base = get_settings()
    return dataclasses.replace(base, llm_provider="none", audit_log_path=tmp_path / "audit.jsonl", stt_provider="mock")


FIXED_NOW = datetime(2026, 10, 22, 9, 30)  # Thursday, clinic local time


def make_client(settings, llm=None) -> TestClient:
    return TestClient(create_app(settings=settings, llm=llm, stt=MockSTT(), clock=lambda: FIXED_NOW))


@pytest.fixture
def client(settings) -> TestClient:
    return make_client(settings)


class Conversation:
    """Helper that keeps the session id across turns."""

    def __init__(self, client: TestClient, clinician: str = "dr-anand"):
        self.client = client
        self.clinician = clinician
        self.session_id: str | None = None

    def say(self, text: str) -> dict:
        r = self.client.post(
            "/api/chat", json={"text": text, "session_id": self.session_id}, headers={"X-Clinician-Id": self.clinician}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        self.session_id = body["session_id"]
        return body


@pytest.fixture
def convo(client) -> Conversation:
    return Conversation(client)
