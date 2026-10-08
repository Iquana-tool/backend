"""Request bodies and vocabulary for personal and organisation API keys."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class CredentialKind(StrEnum):
    """What a stored key is for."""

    LLM = "llm"


def _strip_or_none(value):
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


class LlmCredentialSet(BaseModel):
    """Setting a personal or organisation LLM key.

    ``api_key`` can be left out to change only the model: the stored key is kept.
    Changing ``api_base`` needs the key again, though -- see
    `app.services.credentials.set_llm_credential`.
    """

    model: str = Field(min_length=3, max_length=200,
                       description='LiteLLM model id, "<provider>/<model>", e.g. "anthropic/claude-opus-4-8".')
    api_key: str | None = Field(None, max_length=512, description="Write-only; never returned.")
    api_base: str | None = Field(None, max_length=255,
                                 description="Only for self-hosted, Azure or Ollama endpoints.")

    _strip_optional = field_validator("api_key", "api_base", mode="before")(_strip_or_none)

    @field_validator("model", mode="before")
    @classmethod
    def _provider_prefixed(cls, value):
        value = value.strip() if isinstance(value, str) else value
        if isinstance(value, str) and "/" not in value:
            raise ValueError('Name the provider too, as "<provider>/<model>".')
        return value
