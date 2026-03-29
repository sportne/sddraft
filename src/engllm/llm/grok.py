"""Grok/xAI provider implementation (isolated in llm module)."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from engllm.domain.errors import LLMError
from engllm.llm.base import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    validate_json_text,
    validate_payload,
)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"

_openai_sdk: Any = None
try:
    from openai import OpenAI as _OpenAIClient

    _openai_sdk = _OpenAIClient
except Exception:  # pragma: no cover - optional dependency
    pass

OpenAI: Any = _openai_sdk


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        item_type = getattr(item, "type", None)
        if item_type == "output_text":
            text_value = getattr(item, "text", None)
            if isinstance(text_value, str):
                text_parts.append(text_value)
    return "".join(text_parts)


def _coerce_parsed_content(
    response_model: type[BaseModel],
    parsed_content: Any,
) -> BaseModel:
    if isinstance(parsed_content, response_model):
        return parsed_content
    if isinstance(parsed_content, BaseModel):
        return response_model.model_validate(parsed_content.model_dump())
    if isinstance(parsed_content, dict):
        return validate_payload(response_model, parsed_content)
    if isinstance(parsed_content, str):
        return validate_json_text(response_model, parsed_content)
    raise LLMError("Grok response missing structured parsed content")


class GrokLLMClient:
    """Structured Grok/xAI adapter behind the provider-neutral interface."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if OpenAI is None:
            raise LLMError(
                "Grok provider dependencies are unavailable. "
                "Install engllm[grok] to use this provider."
            )

        api_token = api_key or os.getenv("XAI_API_KEY")
        if not api_token:
            raise LLMError("XAI_API_KEY is not configured")

        self._model_name = model_name
        self._client = OpenAI(
            api_key=api_token,
            base_url=base_url or os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL,
            timeout=timeout_seconds,
        )

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> StructuredGenerationResponse:
        """Send one schema-constrained request to Grok and validate output."""

        try:
            response = self._client.beta.chat.completions.parse(
                model=request.model_name or self._model_name,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                response_format=request.response_model,
                temperature=request.temperature,
            )
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise LLMError(f"Grok request failed: {exc}") from exc

        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise LLMError("Grok returned no choices")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMError("Grok response missing parsed message")

        parsed_content = getattr(message, "parsed", None)
        if parsed_content is None:
            refusal_message = getattr(message, "refusal", None)
            if isinstance(refusal_message, str) and refusal_message.strip():
                raise LLMError(f"Grok returned a refusal: {refusal_message.strip()}")
            raise LLMError("Grok response missing structured parsed content")

        parsed = _coerce_parsed_content(request.response_model, parsed_content)
        raw_text = _extract_message_text(message) or parsed.model_dump_json()
        return StructuredGenerationResponse(
            content=parsed,
            raw_text=raw_text,
            model_name=request.model_name or self._model_name,
        )
