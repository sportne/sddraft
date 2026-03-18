"""Local Ollama provider implementation (isolated in llm module)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request as request_lib

from sddraft.domain.errors import LLMError
from sddraft.llm.base import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    validate_json_text,
)

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _normalize_chat_url(base_url: str) -> str:
    raw = base_url.strip() or DEFAULT_OLLAMA_BASE_URL
    if "://" not in raw:
        raw = f"http://{raw}"

    parsed = urllib.parse.urlparse(raw)
    path = parsed.path.rstrip("/")

    if not path:
        normalized_path = "/api/chat"
    elif path == "/api":
        normalized_path = "/api/chat"
    elif path == "/api/chat":
        normalized_path = "/api/chat"
    else:
        normalized_path = f"{path}/api/chat"

    normalized = parsed._replace(path=normalized_path, params="", query="", fragment="")
    return urllib.parse.urlunparse(normalized)


class OllamaLLMClient:
    """Structured Ollama adapter behind the provider-neutral interface."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        env_base_url = os.getenv("OLLAMA_BASE_URL")
        resolved_base_url = base_url or env_base_url or DEFAULT_OLLAMA_BASE_URL
        self._model_name = model_name
        self._chat_url = _normalize_chat_url(resolved_base_url)
        self._timeout_seconds = timeout_seconds

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> StructuredGenerationResponse:
        """Send one schema-constrained request to Ollama and validate JSON output."""

        request_payload = {
            "model": request.model_name or self._model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": False,
            "format": request.response_model.model_json_schema(),
            "options": {"temperature": request.temperature},
        }

        body = json.dumps(request_payload).encode("utf-8")
        http_request = request_lib.Request(
            self._chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request_lib.urlopen(
                http_request, timeout=self._timeout_seconds
            ) as resp:
                status = int(getattr(resp, "status", 200))
                raw_response = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            detail = error_body or exc.reason or "Unknown error"
            raise LLMError(
                f"Ollama request failed with status {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(
                f"Cannot connect to Ollama at {self._chat_url}. "
                "Ensure Ollama is running and OLLAMA_BASE_URL is correct."
            ) from exc

        if status < 200 or status >= 300:
            raise LLMError(
                f"Ollama request failed with status {status}: "
                f"{raw_response or 'Empty response body'}"
            )

        try:
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise LLMError("Ollama returned malformed JSON response.") from exc

        if not isinstance(response_payload, dict):
            raise LLMError("Ollama returned malformed JSON response.")

        message = response_payload.get("message")
        if not isinstance(message, dict):
            raise LLMError("Ollama response missing 'message.content'")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama response missing 'message.content'")

        parsed = validate_json_text(request.response_model, content)
        return StructuredGenerationResponse(
            content=parsed,
            raw_text=content,
            model_name=request.model_name or self._model_name,
        )
