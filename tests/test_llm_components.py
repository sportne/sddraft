"""Tests for LLM abstraction, factory, mock, and Gemini adapter behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from sddraft.domain.errors import LLMError, ValidationError
from sddraft.domain.models import LLMConfig, QueryAnswer
from sddraft.llm import gemini as gemini_module
from sddraft.llm.base import (
    StructuredGenerationRequest,
    validate_json_text,
    validate_payload,
)
from sddraft.llm.factory import create_llm_client
from sddraft.llm.gemini import GeminiLLMClient
from sddraft.llm.mock import MockLLMClient


class TinyModel(BaseModel):
    value: int


class MiscModel(BaseModel):
    text: str
    count: int
    enabled: bool
    tags: list[str]
    meta: dict[str, str]


def test_validate_payload_and_json_helpers() -> None:
    parsed = validate_payload(TinyModel, {"value": 7})
    assert isinstance(parsed, TinyModel)
    assert parsed.value == 7

    parsed_json = validate_json_text(TinyModel, '{"value": 9}')
    assert parsed_json.value == 9

    with pytest.raises(ValidationError):
        validate_payload(TinyModel, {"bad": 1})

    with pytest.raises(ValidationError):
        validate_json_text(TinyModel, '{"bad": 1}')

    with pytest.raises(ValidationError):
        validate_json_text(TinyModel, "not-json")


def test_factory_and_mock_payload_paths() -> None:
    config = LLMConfig(provider="mock", model_name="mock-sddraft", temperature=0.2)
    client = create_llm_client(config)
    assert isinstance(client, MockLLMClient)

    with pytest.raises(LLMError):
        create_llm_client(config, provider="unsupported")

    mock = MockLLMClient()
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=MiscModel,
        model_name="mock",
    )
    resp = mock.generate_structured(req)
    parsed = MiscModel.model_validate_json(resp.raw_text)
    assert parsed.text == "TBD"
    assert parsed.count == 0
    assert parsed.enabled is False
    assert parsed.tags == []
    assert parsed.meta == {}


@dataclass
class _FakePart:
    text: str


@dataclass
class _FakeContent:
    parts: list[_FakePart]


@dataclass
class _FakeCandidate:
    content: _FakeContent | None


@dataclass
class _FakeResponse:
    candidates: list[_FakeCandidate]


class _FakeModelAPI:
    def __init__(self, response: _FakeResponse | Exception):
        self._response = response

    def generate_content(self, **kwargs):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeClient:
    def __init__(self, *, api_key: str, response: _FakeResponse | Exception):
        self.api_key = api_key
        self.models = _FakeModelAPI(response)


class _FakeGenAI:
    def __init__(self, response: _FakeResponse | Exception):
        self._response = response

    def Client(self, *, api_key: str):
        return _FakeClient(api_key=api_key, response=self._response)


class _FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_gemini_init_and_generate_paths(monkeypatch) -> None:
    monkeypatch.setattr(gemini_module, "genai", None)
    monkeypatch.setattr(gemini_module, "GenerateContentConfig", None)
    with pytest.raises(LLMError):
        GeminiLLMClient(model_name="gemini")

    good_response = _FakeResponse(
        candidates=[
            _FakeCandidate(content=_FakeContent(parts=[_FakePart('{"value": 1}')]))
        ]
    )
    monkeypatch.setattr(gemini_module, "genai", _FakeGenAI(good_response))
    monkeypatch.setattr(gemini_module, "GenerateContentConfig", _FakeConfig)

    with pytest.raises(LLMError):
        GeminiLLMClient(model_name="gemini", api_key=None)

    client = GeminiLLMClient(model_name="gemini", api_key="abc")
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="gemini",
    )
    resp = client.generate_structured(req)
    assert TinyModel.model_validate(resp.content.model_dump()).value == 1


def test_gemini_error_branches(monkeypatch) -> None:
    no_candidates = _FakeResponse(candidates=[])
    monkeypatch.setattr(gemini_module, "genai", _FakeGenAI(no_candidates))
    monkeypatch.setattr(gemini_module, "GenerateContentConfig", _FakeConfig)
    client = GeminiLLMClient(model_name="gemini", api_key="abc")

    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="gemini",
    )

    with pytest.raises(LLMError):
        client.generate_structured(req)

    no_parts = _FakeResponse(
        candidates=[_FakeCandidate(content=_FakeContent(parts=[]))]
    )
    monkeypatch.setattr(gemini_module, "genai", _FakeGenAI(no_parts))
    client = GeminiLLMClient(model_name="gemini", api_key="abc")
    with pytest.raises(LLMError):
        client.generate_structured(req)

    provider_error = RuntimeError("provider down")
    monkeypatch.setattr(gemini_module, "genai", _FakeGenAI(provider_error))
    client = GeminiLLMClient(model_name="gemini", api_key="abc")
    with pytest.raises(LLMError):
        client.generate_structured(req)


def test_mock_canned_query_answer() -> None:
    canned = {
        "QueryAnswer": {
            "answer": "Known",
            "citations": [],
            "uncertainty": [],
            "missing_information": [],
            "confidence": 0.9,
        }
    }
    mock = MockLLMClient(canned=canned)
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=QueryAnswer,
        model_name="mock",
    )
    resp = mock.generate_structured(req)
    parsed = QueryAnswer.model_validate(resp.content.model_dump())
    assert parsed.answer == "Known"
