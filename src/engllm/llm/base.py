"""Provider-neutral structured generation interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from engllm.domain.errors import LLMError, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredGenerationRequest:
    """Structured generation request."""

    system_prompt: str
    user_prompt: str
    response_model: type[BaseModel]
    model_name: str
    temperature: float = 0.2


@dataclass(frozen=True)
class StructuredGenerationResponse:
    """Structured generation response."""

    content: BaseModel
    raw_text: str
    model_name: str


class LLMClient(Protocol):
    """Provider-neutral client contract."""

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> StructuredGenerationResponse:
        """Generate and validate a structured response."""


def validate_payload(
    response_model: type[BaseModel],
    payload: dict[str, object],
) -> BaseModel:
    """Validate a provider payload against the requested schema."""

    try:
        return response_model.model_validate(payload)
    except PydanticValidationError as exc:
        raise ValidationError(f"Structured response validation failed: {exc}") from exc


def validate_json_text(response_model: type[BaseModel], json_text: str) -> BaseModel:
    """Validate JSON text against response schema."""

    try:
        return response_model.model_validate_json(json_text)
    except PydanticValidationError as exc:
        raise ValidationError(f"Structured JSON validation failed: {exc}") from exc
    except ValueError as exc:
        raise LLMError(
            f"Provider returned non-JSON structured response: {exc}"
        ) from exc
