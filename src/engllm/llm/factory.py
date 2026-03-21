"""LLM provider factory."""

from __future__ import annotations

from engllm.domain.errors import LLMError
from engllm.domain.models import LLMConfig
from engllm.llm.base import LLMClient
from engllm.llm.gemini import GeminiLLMClient
from engllm.llm.mock import MockLLMClient
from engllm.llm.ollama import OllamaLLMClient


def create_llm_client(
    config: LLMConfig,
    provider: str | None = None,
    model_name: str | None = None,
) -> LLMClient:
    """Create provider client from config and optional CLI overrides."""

    resolved_provider = (provider or config.provider).lower()
    resolved_model = model_name or config.model_name

    if resolved_provider == "mock":
        return MockLLMClient(model_name=resolved_model)
    if resolved_provider == "gemini":
        return GeminiLLMClient(model_name=resolved_model)
    if resolved_provider == "ollama":
        return OllamaLLMClient(model_name=resolved_model)

    raise LLMError(f"Unsupported LLM provider '{resolved_provider}'")
