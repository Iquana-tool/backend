"""Schemas for the LLM-assisted label-space generation feature.

These models are intentionally kept separate from the database `Label` schema
(`iquana_toolbox.schemas.database.labels.Label`): a generated *draft* has no
database ids, dataset ids or label values. Those are only assigned when the user
approves the draft and it is persisted via the bulk-create endpoint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DraftLabel(BaseModel):
    """A single node in a proposed label hierarchy (no database identity yet)."""
    name: str = Field(..., description="Human-readable label name, unique within the whole dataset.")
    description: str | None = Field(
        None,
        description="Short rationale for this label, shown to the user in the review step.",
    )
    children: list["DraftLabel"] = Field(
        default_factory=list,
        description="Nested sub-labels forming the tree structure.",
    )


class LabelSpaceDraft(BaseModel):
    """A complete proposed label hierarchy returned by the LLM."""
    labels: list[DraftLabel] = Field(
        default_factory=list,
        description="Root-level labels. Each may contain nested children.",
    )


class GenerateLabelSpaceRequest(BaseModel):
    """Request to turn a plain-language description into a draft label space."""
    description: str = Field(..., min_length=1, description="Plain-language description of what to segment.")
    max_depth: int = Field(3, ge=1, le=5, description="Maximum nesting depth of the generated hierarchy.")
    max_labels: int = Field(50, ge=1, le=200, description="Maximum total number of labels to generate.")
    model: str | None = Field(
        None,
        description="Optional override of the configured LLM model (e.g. 'openai/gpt-4o'). "
                    "Server-side only; the API key is never accepted from the client.",
    )


class RefineLabelSpaceRequest(BaseModel):
    """Request to adjust an existing draft from a follow-up instruction."""
    current_draft: LabelSpaceDraft = Field(..., description="The draft to revise.")
    message: str = Field(..., min_length=1, description="Follow-up instruction, e.g. 'merge the two vehicle groups'.")
    description: str | None = Field(None, description="Original description, for additional context.")
    max_depth: int = Field(3, ge=1, le=5)
    max_labels: int = Field(50, ge=1, le=200)
    model: str | None = Field(None)


class GenerateLabelSpaceResponse(BaseModel):
    success: bool = True
    draft: LabelSpaceDraft


class LabelSpaceConfigResponse(BaseModel):
    """Tells the frontend whether generation is available and which model is used."""
    enabled: bool
    model: str | None = None
