"""LLM provider factory."""

from __future__ import annotations

from sddraft.domain.errors import LLMError
from sddraft.domain.models import LLMConfig
from sddraft.llm.base import LLMClient
from sddraft.llm.gemini import GeminiLLMClient
from sddraft.llm.mock import MockLLMClient
from sddraft.llm.ollama import OllamaLLMClient


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
