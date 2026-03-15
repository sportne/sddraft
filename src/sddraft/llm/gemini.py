"""Gemini provider implementation (isolated in llm module)."""

from __future__ import annotations

import os

from sddraft.domain.errors import LLMError
from sddraft.llm.base import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    validate_json_text,
)

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
except Exception:  # pragma: no cover - optional dependency
    genai = None
    GenerateContentConfig = None


class GeminiLLMClient:
    """Structured Gemini adapter behind provider-neutral interface."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        if genai is None or GenerateContentConfig is None:
            raise LLMError(
                "Gemini provider dependencies are unavailable. Install google-genai to use this provider."
            )

        api_token = api_key or os.getenv("GEMINI_API_KEY")
        if not api_token:
            raise LLMError("GEMINI_API_KEY is not configured")

        self._model_name = model_name
        self._client = genai.Client(api_key=api_token)

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> StructuredGenerationResponse:
        prompt = f"{request.system_prompt}\n\n{request.user_prompt}"
        try:
            response = self._client.models.generate_content(
                model=request.model_name or self._model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    response_modalities=["TEXT"],
                    response_mime_type="application/json",
                    response_schema=request.response_model,
                    temperature=request.temperature,
                ),
            )
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise LLMError(f"Gemini request failed: {exc}") from exc

        candidates = getattr(response, "candidates", None)
        if not candidates:
            raise LLMError("Gemini returned no candidates")

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            raise LLMError("Gemini returned no content parts")

        raw_text = "".join(getattr(part, "text", "") for part in parts)
        parsed = validate_json_text(request.response_model, raw_text)
        return StructuredGenerationResponse(
            content=parsed,
            raw_text=raw_text,
            model_name=request.model_name or self._model_name,
        )
