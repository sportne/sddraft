"""Anthropic provider implementation (isolated in llm module)."""

from __future__ import annotations

import json
import os
from typing import Any

from engllm.domain.errors import LLMError
from engllm.llm.base import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    validate_payload,
)

_anthropic_sdk: Any = None
try:
    from anthropic import Anthropic as _AnthropicClient

    _anthropic_sdk = _AnthropicClient
except Exception:  # pragma: no cover - optional dependency
    pass

Anthropic: Any = _anthropic_sdk

_STRUCTURED_TOOL_NAME = "emit_structured_response"


def _extract_tool_payload(response: Any) -> dict[str, object]:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        raise LLMError("Anthropic response missing content blocks")

    refusal_texts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", None)
            if tool_name != _STRUCTURED_TOOL_NAME:
                continue
            if not isinstance(tool_input, dict):
                raise LLMError("Anthropic tool_use block missing structured input")
            return tool_input
        if block_type == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                refusal_texts.append(text.strip())

    if refusal_texts:
        raise LLMError(
            "Anthropic returned text instead of the required structured tool call: "
            + " ".join(refusal_texts)
        )
    raise LLMError("Anthropic response missing structured tool output")


class AnthropicLLMClient:
    """Structured Anthropic adapter behind the provider-neutral interface."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if Anthropic is None:
            raise LLMError(
                "Anthropic provider dependencies are unavailable. "
                "Install engllm[anthropic] to use this provider."
            )

        api_token = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_token:
            raise LLMError("ANTHROPIC_API_KEY is not configured")

        self._model_name = model_name
        self._client = Anthropic(
            api_key=api_token,
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"),
            timeout=timeout_seconds,
        )

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> StructuredGenerationResponse:
        """Send one schema-constrained request to Anthropic and validate output."""

        tool_schema = request.response_model.model_json_schema()
        try:
            response = self._client.messages.create(
                model=request.model_name or self._model_name,
                system=request.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": request.user_prompt,
                    }
                ],
                max_tokens=4096,
                temperature=request.temperature,
                tools=[
                    {
                        "name": _STRUCTURED_TOOL_NAME,
                        "description": (
                            "Return the final response as structured JSON that matches "
                            "the provided schema exactly."
                        ),
                        "input_schema": tool_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": _STRUCTURED_TOOL_NAME},
            )
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        payload = _extract_tool_payload(response)
        parsed = validate_payload(request.response_model, payload)
        return StructuredGenerationResponse(
            content=parsed,
            raw_text=json.dumps(payload, sort_keys=True),
            model_name=request.model_name or self._model_name,
        )
