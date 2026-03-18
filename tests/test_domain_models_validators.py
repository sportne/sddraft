"""Validator coverage for domain models."""

from __future__ import annotations

import pytest

from sddraft.domain.models import LLMConfig, QueryRequest


def test_llm_config_rejects_out_of_range_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be between 0.0 and 1.0"):
        LLMConfig(temperature=1.5)


def test_query_request_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        QueryRequest(question="   ")
