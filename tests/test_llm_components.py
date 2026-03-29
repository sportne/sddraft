"""Tests for LLM abstraction, factory, mock, and Gemini adapter behavior."""

from __future__ import annotations

import io
import urllib.error
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from engllm.domain.errors import LLMError, ValidationError
from engllm.domain.models import LLMConfig, QueryAnswer
from engllm.llm import anthropic as anthropic_module
from engllm.llm import gemini as gemini_module
from engllm.llm import grok as grok_module
from engllm.llm import ollama as ollama_module
from engllm.llm import openai as openai_module
from engllm.llm.anthropic import AnthropicLLMClient
from engllm.llm.base import (
    StructuredGenerationRequest,
    validate_json_text,
    validate_payload,
)
from engllm.llm.factory import create_llm_client
from engllm.llm.gemini import GeminiLLMClient
from engllm.llm.grok import GrokLLMClient
from engllm.llm.mock import MockLLMClient
from engllm.llm.ollama import OllamaLLMClient
from engllm.llm.openai import OpenAILLMClient


class TinyModel(BaseModel):
    value: int


class TinyAltModel(BaseModel):
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
    config = LLMConfig(provider="mock", model_name="mock-engllm", temperature=0.2)
    client = create_llm_client(config)
    assert isinstance(client, MockLLMClient)
    ollama_client = create_llm_client(config, provider="ollama", model_name="qwen")
    assert isinstance(ollama_client, OllamaLLMClient)

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


def test_factory_returns_openai_client(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    class _FakeOpenAIResponses:
        def parse(self, **_kwargs):
            raise AssertionError("parse should not be called in the factory test")

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.responses = _FakeOpenAIResponses()

    monkeypatch.setattr(openai_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "abc")
    config = LLMConfig(provider="mock", model_name="mock-engllm", temperature=0.2)
    client = create_llm_client(config, provider="openai", model_name="gpt-4.1-mini")
    assert isinstance(client, OpenAILLMClient)
    assert created[0]["api_key"] == "abc"


def test_factory_returns_anthropic_client(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    class _FakeAnthropicMessages:
        def create(self, **_kwargs):
            raise AssertionError("create should not be called in the factory test")

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.messages = _FakeAnthropicMessages()

    monkeypatch.setattr(anthropic_module, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abc")
    config = LLMConfig(provider="mock", model_name="mock-engllm", temperature=0.2)
    client = create_llm_client(
        config, provider="anthropic", model_name="claude-3-5-sonnet-latest"
    )
    assert isinstance(client, AnthropicLLMClient)
    assert created[0]["api_key"] == "abc"


def test_factory_returns_grok_client(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    class _FakeGrokCompletions:
        def parse(self, **_kwargs):
            raise AssertionError("parse should not be called in the factory test")

    class _FakeGrokBeta:
        def __init__(self) -> None:
            self.chat = type(
                "_FakeChat",
                (),
                {"completions": _FakeGrokCompletions()},
            )()

    class _FakeGrokOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.beta = _FakeGrokBeta()

    monkeypatch.setattr(grok_module, "OpenAI", _FakeGrokOpenAI)
    monkeypatch.setenv("XAI_API_KEY", "abc")
    config = LLMConfig(provider="mock", model_name="mock-engllm", temperature=0.2)
    client = create_llm_client(config, provider="grok", model_name="grok-4-fast")
    assert isinstance(client, GrokLLMClient)
    assert created[0]["api_key"] == "abc"


@dataclass
class _FakeOpenAIMessage:
    type: str
    refusal: str | None = None


@dataclass
class _FakeOpenAIOutput:
    content: list[_FakeOpenAIMessage] | None


@dataclass
class _FakeOpenAIResponse:
    output_parsed: object | None
    output_text: str = ""
    output: list[_FakeOpenAIOutput] | None = None


class _FakeOpenAIResponsesAPI:
    def __init__(self, response: _FakeOpenAIResponse | Exception):
        self._response = response
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeOpenAIClient:
    def __init__(
        self,
        *,
        response: _FakeOpenAIResponse | Exception,
        created_clients: list[dict[str, object]],
        **kwargs,
    ) -> None:
        created_clients.append(kwargs)
        self.responses = _FakeOpenAIResponsesAPI(response)


class _FakeOpenAIConstructor:
    def __init__(self, response: _FakeOpenAIResponse | Exception) -> None:
        self.response = response
        self.created_clients: list[dict[str, object]] = []
        self.instances: list[_FakeOpenAIClient] = []

    def __call__(self, **kwargs):
        client = _FakeOpenAIClient(
            response=self.response,
            created_clients=self.created_clients,
            **kwargs,
        )
        self.instances.append(client)
        return client


@dataclass
class _FakeGrokMessage:
    parsed: object | None
    content: object = ""
    refusal: str | None = None


@dataclass
class _FakeGrokChoice:
    message: _FakeGrokMessage | None


@dataclass
class _FakeGrokResponse:
    choices: list[_FakeGrokChoice] | None


class _FakeGrokCompletionsAPI:
    def __init__(self, response: _FakeGrokResponse | Exception):
        self._response = response
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeGrokClient:
    def __init__(
        self,
        *,
        response: _FakeGrokResponse | Exception,
        created_clients: list[dict[str, object]],
        **kwargs,
    ) -> None:
        created_clients.append(kwargs)
        completions = _FakeGrokCompletionsAPI(response)
        chat = type("_FakeChat", (), {"completions": completions})()
        self.beta = type("_FakeBeta", (), {"chat": chat})()


class _FakeGrokConstructor:
    def __init__(self, response: _FakeGrokResponse | Exception) -> None:
        self.response = response
        self.created_clients: list[dict[str, object]] = []
        self.instances: list[_FakeGrokClient] = []

    def __call__(self, **kwargs):
        client = _FakeGrokClient(
            response=self.response,
            created_clients=self.created_clients,
            **kwargs,
        )
        self.instances.append(client)
        return client


@dataclass
class _FakeAnthropicContentBlock:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict[str, object] | None = None


@dataclass
class _FakeAnthropicResponse:
    content: list[_FakeAnthropicContentBlock] | None


class _FakeAnthropicMessagesAPI:
    def __init__(self, response: _FakeAnthropicResponse | Exception):
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeAnthropicClient:
    def __init__(
        self,
        *,
        response: _FakeAnthropicResponse | Exception,
        created_clients: list[dict[str, object]],
        **kwargs,
    ) -> None:
        created_clients.append(kwargs)
        self.messages = _FakeAnthropicMessagesAPI(response)


class _FakeAnthropicConstructor:
    def __init__(self, response: _FakeAnthropicResponse | Exception) -> None:
        self.response = response
        self.created_clients: list[dict[str, object]] = []
        self.instances: list[_FakeAnthropicClient] = []

    def __call__(self, **kwargs):
        client = _FakeAnthropicClient(
            response=self.response,
            created_clients=self.created_clients,
            **kwargs,
        )
        self.instances.append(client)
        return client


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


def test_anthropic_init_and_generate_paths(monkeypatch) -> None:
    monkeypatch.setattr(anthropic_module, "Anthropic", None)
    with pytest.raises(LLMError, match="dependencies are unavailable"):
        AnthropicLLMClient(model_name="claude-3-5-sonnet-latest", api_key="abc")

    fake_constructor = _FakeAnthropicConstructor(
        _FakeAnthropicResponse(
            content=[
                _FakeAnthropicContentBlock(
                    type="tool_use",
                    name="emit_structured_response",
                    input={"value": 5},
                )
            ]
        )
    )
    monkeypatch.setattr(anthropic_module, "Anthropic", fake_constructor)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not configured"):
        AnthropicLLMClient(model_name="claude-3-5-sonnet-latest")

    client = AnthropicLLMClient(
        model_name="claude-3-5-sonnet-latest",
        api_key="abc",
        base_url="https://example.invalid",
        timeout_seconds=12.0,
    )
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="claude-3-5-haiku-latest",
        temperature=0.42,
    )
    resp = client.generate_structured(req)

    assert fake_constructor.created_clients[0]["api_key"] == "abc"
    assert fake_constructor.created_clients[0]["base_url"] == "https://example.invalid"
    assert fake_constructor.created_clients[0]["timeout"] == 12.0

    create_call = fake_constructor.instances[0].messages.calls[0]
    assert create_call["model"] == "claude-3-5-haiku-latest"
    assert create_call["system"] == "sys"
    assert create_call["temperature"] == 0.42
    assert create_call["max_tokens"] == 4096
    assert create_call["messages"] == [{"role": "user", "content": "user"}]
    assert create_call["tool_choice"] == {
        "type": "tool",
        "name": "emit_structured_response",
    }
    assert create_call["tools"][0]["name"] == "emit_structured_response"
    assert create_call["tools"][0]["input_schema"]["type"] == "object"

    parsed = TinyModel.model_validate(resp.content.model_dump())
    assert parsed.value == 5
    assert resp.raw_text == '{"value": 5}'


def test_anthropic_error_branches(monkeypatch) -> None:
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="claude-3-5-sonnet-latest",
    )

    provider_error = _FakeAnthropicConstructor(RuntimeError("provider down"))
    monkeypatch.setattr(anthropic_module, "Anthropic", provider_error)
    client = AnthropicLLMClient(model_name="claude-3-5-sonnet-latest", api_key="abc")
    with pytest.raises(LLMError, match="Anthropic request failed: provider down"):
        client.generate_structured(req)

    missing_content = _FakeAnthropicConstructor(_FakeAnthropicResponse(content=None))
    monkeypatch.setattr(anthropic_module, "Anthropic", missing_content)
    client = AnthropicLLMClient(model_name="claude-3-5-sonnet-latest", api_key="abc")
    with pytest.raises(LLMError, match="missing content blocks"):
        client.generate_structured(req)

    text_instead_of_tool = _FakeAnthropicConstructor(
        _FakeAnthropicResponse(
            content=[_FakeAnthropicContentBlock(type="text", text="I refuse.")]
        )
    )
    monkeypatch.setattr(anthropic_module, "Anthropic", text_instead_of_tool)
    client = AnthropicLLMClient(model_name="claude-3-5-sonnet-latest", api_key="abc")
    with pytest.raises(
        LLMError, match="returned text instead of the required structured tool call"
    ):
        client.generate_structured(req)

    bad_tool_input = _FakeAnthropicConstructor(
        _FakeAnthropicResponse(
            content=[
                _FakeAnthropicContentBlock(
                    type="tool_use",
                    name="emit_structured_response",
                    input=None,
                )
            ]
        )
    )
    monkeypatch.setattr(anthropic_module, "Anthropic", bad_tool_input)
    client = AnthropicLLMClient(model_name="claude-3-5-sonnet-latest", api_key="abc")
    with pytest.raises(LLMError, match="tool_use block missing structured input"):
        client.generate_structured(req)

    missing_tool_output = _FakeAnthropicConstructor(
        _FakeAnthropicResponse(
            content=[
                _FakeAnthropicContentBlock(
                    type="tool_use",
                    name="other_tool",
                    input={"value": 1},
                )
            ]
        )
    )
    monkeypatch.setattr(anthropic_module, "Anthropic", missing_tool_output)
    client = AnthropicLLMClient(model_name="claude-3-5-sonnet-latest", api_key="abc")
    with pytest.raises(LLMError, match="missing structured tool output"):
        client.generate_structured(req)


def test_openai_init_and_generate_paths(monkeypatch) -> None:
    monkeypatch.setattr(openai_module, "OpenAI", None)
    with pytest.raises(LLMError, match="dependencies are unavailable"):
        OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")

    fake_constructor = _FakeOpenAIConstructor(
        _FakeOpenAIResponse(
            output_parsed={"value": 7},
            output_text='{"value":7}',
        )
    )
    monkeypatch.setattr(openai_module, "OpenAI", fake_constructor)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY is not configured"):
        OpenAILLMClient(model_name="gpt-4.1-mini")

    client = OpenAILLMClient(
        model_name="gpt-4.1-mini",
        api_key="abc",
        base_url="https://example.invalid/v1",
        timeout_seconds=12.0,
    )
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="gpt-4.1",
        temperature=0.42,
    )
    resp = client.generate_structured(req)

    assert fake_constructor.created_clients[0]["api_key"] == "abc"
    assert (
        fake_constructor.created_clients[0]["base_url"] == "https://example.invalid/v1"
    )
    assert fake_constructor.created_clients[0]["timeout"] == 12.0

    parse_call = fake_constructor.instances[0].responses.calls[0]
    assert parse_call["model"] == "gpt-4.1"
    assert parse_call["temperature"] == 0.42
    assert parse_call["text_format"] is TinyModel
    assert parse_call["input"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]

    parsed = TinyModel.model_validate(resp.content.model_dump())
    assert parsed.value == 7
    assert resp.raw_text == '{"value":7}'


def test_openai_error_branches(monkeypatch) -> None:
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="gpt-4.1-mini",
    )

    provider_error = _FakeOpenAIConstructor(RuntimeError("provider down"))
    monkeypatch.setattr(openai_module, "OpenAI", provider_error)
    client = OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")
    with pytest.raises(LLMError, match="OpenAI request failed: provider down"):
        client.generate_structured(req)

    refusal_response = _FakeOpenAIConstructor(
        _FakeOpenAIResponse(
            output_parsed=None,
            output=[
                _FakeOpenAIOutput(
                    content=[_FakeOpenAIMessage(type="refusal", refusal="safety")]
                )
            ],
        )
    )
    monkeypatch.setattr(openai_module, "OpenAI", refusal_response)
    client = OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")
    with pytest.raises(LLMError, match="returned a refusal: safety"):
        client.generate_structured(req)

    missing_content = _FakeOpenAIConstructor(
        _FakeOpenAIResponse(output_parsed=None, output_text="")
    )
    monkeypatch.setattr(openai_module, "OpenAI", missing_content)
    client = OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")
    with pytest.raises(LLMError, match="missing structured parsed content"):
        client.generate_structured(req)


def test_openai_helper_branches(monkeypatch) -> None:
    direct_model = _FakeOpenAIConstructor(
        _FakeOpenAIResponse(output_parsed=TinyModel(value=9), output_text=123)
    )
    monkeypatch.setattr(openai_module, "OpenAI", direct_model)
    client = OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="gpt-4.1-mini",
    )
    resp = client.generate_structured(req)
    assert TinyModel.model_validate(resp.content.model_dump()).value == 9
    assert resp.raw_text == '{"value":9}'

    alternate_model = _FakeOpenAIConstructor(
        _FakeOpenAIResponse(output_parsed=TinyAltModel(value=11))
    )
    monkeypatch.setattr(openai_module, "OpenAI", alternate_model)
    client = OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")
    resp = client.generate_structured(req)
    assert TinyModel.model_validate(resp.content.model_dump()).value == 11

    json_string = _FakeOpenAIConstructor(
        _FakeOpenAIResponse(output_parsed='{"value": 13}')
    )
    monkeypatch.setattr(openai_module, "OpenAI", json_string)
    client = OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")
    resp = client.generate_structured(req)
    assert TinyModel.model_validate(resp.content.model_dump()).value == 13

    unsupported_shape = _FakeOpenAIConstructor(
        _FakeOpenAIResponse(
            output_parsed=None,
            output=[_FakeOpenAIOutput(content=[]), _FakeOpenAIOutput(content=None)],
        )
    )
    monkeypatch.setattr(openai_module, "OpenAI", unsupported_shape)
    client = OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")
    with pytest.raises(LLMError, match="missing structured parsed content"):
        client.generate_structured(req)

    invalid_parsed_shape = _FakeOpenAIConstructor(_FakeOpenAIResponse(output_parsed=17))
    monkeypatch.setattr(openai_module, "OpenAI", invalid_parsed_shape)
    client = OpenAILLMClient(model_name="gpt-4.1-mini", api_key="abc")
    with pytest.raises(LLMError, match="missing structured parsed content"):
        client.generate_structured(req)


def test_grok_init_and_generate_paths(monkeypatch) -> None:
    monkeypatch.setattr(grok_module, "OpenAI", None)
    with pytest.raises(LLMError, match="dependencies are unavailable"):
        GrokLLMClient(model_name="grok-4-fast", api_key="abc")

    fake_constructor = _FakeGrokConstructor(
        _FakeGrokResponse(
            choices=[
                _FakeGrokChoice(
                    message=_FakeGrokMessage(
                        parsed={"value": 17},
                        content='{"value":17}',
                    )
                )
            ]
        )
    )
    monkeypatch.setattr(grok_module, "OpenAI", fake_constructor)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="XAI_API_KEY is not configured"):
        GrokLLMClient(model_name="grok-4-fast")

    client = GrokLLMClient(
        model_name="grok-4-fast",
        api_key="abc",
        base_url="https://example.invalid/v1",
        timeout_seconds=12.0,
    )
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="grok-4-1-fast",
        temperature=0.42,
    )
    resp = client.generate_structured(req)

    assert fake_constructor.created_clients[0]["api_key"] == "abc"
    assert (
        fake_constructor.created_clients[0]["base_url"] == "https://example.invalid/v1"
    )
    assert fake_constructor.created_clients[0]["timeout"] == 12.0

    parse_call = fake_constructor.instances[0].beta.chat.completions.calls[0]
    assert parse_call["model"] == "grok-4-1-fast"
    assert parse_call["temperature"] == 0.42
    assert parse_call["response_format"] is TinyModel
    assert parse_call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]

    parsed = TinyModel.model_validate(resp.content.model_dump())
    assert parsed.value == 17
    assert resp.raw_text == '{"value":17}'


def test_grok_error_branches(monkeypatch) -> None:
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="grok-4-fast",
    )

    provider_error = _FakeGrokConstructor(RuntimeError("provider down"))
    monkeypatch.setattr(grok_module, "OpenAI", provider_error)
    client = GrokLLMClient(model_name="grok-4-fast", api_key="abc")
    with pytest.raises(LLMError, match="Grok request failed: provider down"):
        client.generate_structured(req)

    no_choices = _FakeGrokConstructor(_FakeGrokResponse(choices=[]))
    monkeypatch.setattr(grok_module, "OpenAI", no_choices)
    client = GrokLLMClient(model_name="grok-4-fast", api_key="abc")
    with pytest.raises(LLMError, match="returned no choices"):
        client.generate_structured(req)

    missing_message = _FakeGrokConstructor(
        _FakeGrokResponse(choices=[_FakeGrokChoice(message=None)])
    )
    monkeypatch.setattr(grok_module, "OpenAI", missing_message)
    client = GrokLLMClient(model_name="grok-4-fast", api_key="abc")
    with pytest.raises(LLMError, match="missing parsed message"):
        client.generate_structured(req)

    refusal_response = _FakeGrokConstructor(
        _FakeGrokResponse(
            choices=[
                _FakeGrokChoice(message=_FakeGrokMessage(parsed=None, refusal="safety"))
            ]
        )
    )
    monkeypatch.setattr(grok_module, "OpenAI", refusal_response)
    client = GrokLLMClient(model_name="grok-4-fast", api_key="abc")
    with pytest.raises(LLMError, match="returned a refusal: safety"):
        client.generate_structured(req)

    missing_parsed = _FakeGrokConstructor(
        _FakeGrokResponse(
            choices=[_FakeGrokChoice(message=_FakeGrokMessage(parsed=None))]
        )
    )
    monkeypatch.setattr(grok_module, "OpenAI", missing_parsed)
    client = GrokLLMClient(model_name="grok-4-fast", api_key="abc")
    with pytest.raises(LLMError, match="missing structured parsed content"):
        client.generate_structured(req)

    parsed_model = _FakeGrokConstructor(
        _FakeGrokResponse(
            choices=[
                _FakeGrokChoice(
                    message=_FakeGrokMessage(parsed=TinyModel(value=19), content=123)
                )
            ]
        )
    )
    monkeypatch.setattr(grok_module, "OpenAI", parsed_model)
    client = GrokLLMClient(model_name="grok-4-fast", api_key="abc")
    resp = client.generate_structured(req)
    assert TinyModel.model_validate(resp.content.model_dump()).value == 19
    assert resp.raw_text == '{"value":19}'

    parsed_json = _FakeGrokConstructor(
        _FakeGrokResponse(
            choices=[
                _FakeGrokChoice(
                    message=_FakeGrokMessage(parsed='{"value": 23}', content=[])
                )
            ]
        )
    )
    monkeypatch.setattr(grok_module, "OpenAI", parsed_json)
    client = GrokLLMClient(model_name="grok-4-fast", api_key="abc")
    resp = client.generate_structured(req)
    assert TinyModel.model_validate(resp.content.model_dump()).value == 23

    unsupported_shape = _FakeGrokConstructor(
        _FakeGrokResponse(
            choices=[_FakeGrokChoice(message=_FakeGrokMessage(parsed=17))]
        )
    )
    monkeypatch.setattr(grok_module, "OpenAI", unsupported_shape)
    client = GrokLLMClient(model_name="grok-4-fast", api_key="abc")
    with pytest.raises(LLMError, match="missing structured parsed content"):
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


class _FakeHTTPResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.status = status

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_ollama_generate_structured_success(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        return _FakeHTTPResponse('{"message": {"content": "{\\"value\\": 5}"}}')

    monkeypatch.setattr(ollama_module.request_lib, "urlopen", _fake_urlopen)
    client = OllamaLLMClient(model_name="qwen", base_url="http://localhost:11434/api")

    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="qwen2.5:14b-instruct-q4_K_M",
        temperature=0.33,
    )
    resp = client.generate_structured(req)

    assert captured["url"] == "http://localhost:11434/api/chat"
    request_body = captured["body"]
    assert isinstance(request_body, str)
    assert '"stream": false' in request_body
    assert '"temperature": 0.33' in request_body
    assert "qwen2.5:14b-instruct-q4_K_M" in request_body

    parsed = TinyModel.model_validate(resp.content.model_dump())
    assert parsed.value == 5


def test_ollama_connection_and_http_error_branches(monkeypatch) -> None:
    client = OllamaLLMClient(model_name="qwen")
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="qwen",
    )

    def _raise_unreachable(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ollama_module.request_lib, "urlopen", _raise_unreachable)
    with pytest.raises(LLMError, match="Cannot connect to Ollama"):
        client.generate_structured(req)

    def _raise_http_error(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b"backend unavailable"),
        )

    monkeypatch.setattr(ollama_module.request_lib, "urlopen", _raise_http_error)
    with pytest.raises(LLMError, match="status 503"):
        client.generate_structured(req)


def test_ollama_malformed_and_missing_content_paths(monkeypatch) -> None:
    client = OllamaLLMClient(model_name="qwen")
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="qwen",
    )

    monkeypatch.setattr(
        ollama_module.request_lib,
        "urlopen",
        lambda *_args, **_kwargs: _FakeHTTPResponse("not-json"),
    )
    with pytest.raises(LLMError, match="malformed JSON response"):
        client.generate_structured(req)

    monkeypatch.setattr(
        ollama_module.request_lib,
        "urlopen",
        lambda *_args, **_kwargs: _FakeHTTPResponse('{"message": {}}'),
    )
    with pytest.raises(LLMError, match="missing 'message.content'"):
        client.generate_structured(req)


def test_ollama_invalid_model_payload_raises_validation(monkeypatch) -> None:
    client = OllamaLLMClient(model_name="qwen")
    req = StructuredGenerationRequest(
        system_prompt="sys",
        user_prompt="user",
        response_model=TinyModel,
        model_name="qwen",
    )

    monkeypatch.setattr(
        ollama_module.request_lib,
        "urlopen",
        lambda *_args, **_kwargs: _FakeHTTPResponse(
            '{"message": {"content": "not-json"}}'
        ),
    )
    with pytest.raises(ValidationError):
        client.generate_structured(req)
