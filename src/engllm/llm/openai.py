"""OpenAI provider implementation (isolated in llm module)."""

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

_openai_sdk: Any = None
try:
    from openai import OpenAI as _OpenAIClient

    _openai_sdk = _OpenAIClient
except Exception:  # pragma: no cover - optional dependency
    pass

OpenAI: Any = _openai_sdk


def _extract_output_text(response: Any) -> str:
    raw_text = getattr(response, "output_text", "")
    if isinstance(raw_text, str):
        return raw_text
    return ""


def _extract_refusal_message(response: Any) -> str | None:
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return None

    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for entry in content:
            if getattr(entry, "type", None) == "refusal":
                refusal_text = getattr(entry, "refusal", None)
                if isinstance(refusal_text, str) and refusal_text.strip():
                    return refusal_text.strip()
    return None


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
    raise LLMError("OpenAI response missing structured parsed content")


class OpenAILLMClient:
    """Structured OpenAI adapter behind the provider-neutral interface."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if OpenAI is None:
            raise LLMError(
                "OpenAI provider dependencies are unavailable. "
                "Install engllm[openai] to use this provider."
            )

        api_token = api_key or os.getenv("OPENAI_API_KEY")
        if not api_token:
            raise LLMError("OPENAI_API_KEY is not configured")

        self._model_name = model_name
        self._client = OpenAI(
            api_key=api_token,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout_seconds,
        )

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> StructuredGenerationResponse:
        """Send one schema-constrained request to OpenAI and validate output."""

        try:
            response = self._client.responses.parse(
                model=request.model_name or self._model_name,
                input=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                text_format=request.response_model,
                temperature=request.temperature,
            )
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        parsed_content = getattr(response, "output_parsed", None)
        if parsed_content is None:
            refusal_message = _extract_refusal_message(response)
            if refusal_message:
                raise LLMError(f"OpenAI returned a refusal: {refusal_message}")
            raise LLMError("OpenAI response missing structured parsed content")

        parsed = _coerce_parsed_content(request.response_model, parsed_content)
        raw_text = _extract_output_text(response) or parsed.model_dump_json()
        return StructuredGenerationResponse(
            content=parsed,
            raw_text=raw_text,
            model_name=request.model_name or self._model_name,
        )
