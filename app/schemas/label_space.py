"""Schemas for the LLM-assisted label-space generation feature.

These models are intentionally kept separate from the database `Label` schema
(`iquana_toolbox.schemas.database.labels.Label`): a generated *draft* has no
database ids, dataset ids or label values. Those are only assigned when the user
approves the draft and it is persisted via the bulk-create endpoint.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DraftAttribute(BaseModel):
    """A proposed attribute of a label's objects: anything that does not change the outline."""
    name: str = Field(..., description="Short lower-case name, e.g. 'species' or 'bleached'.")
    value_type: Literal["categorical", "boolean", "number", "text"] = Field(
        "categorical", description="categorical for a fixed set of kinds, boolean for yes/no.")
    options: list[str] = Field(
        default_factory=list, description="The allowed values of a categorical attribute, e.g. the species.")
    unit: str | None = Field(None, description="Unit of a number attribute.")
    description: str | None = Field(None, description="What the attribute records.")


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
    attributes: list[DraftAttribute] = Field(
        default_factory=list,
        description="Attributes of this label's objects, e.g. species of a coral fragment.",
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
    dataset_id: int | None = Field(
        None, description="The dataset the labels are for, when it exists; its organisation's key applies.")


class RefineLabelSpaceRequest(BaseModel):
    """Request to adjust an existing draft from a follow-up instruction."""
    current_draft: LabelSpaceDraft = Field(..., description="The draft to revise.")
    message: str = Field(..., min_length=1, description="Follow-up instruction, e.g. 'merge the two vehicle groups'.")
    description: str | None = Field(None, description="Original description, for additional context.")
    max_depth: int = Field(3, ge=1, le=5)
    max_labels: int = Field(50, ge=1, le=200)
    model: str | None = Field(None)
    dataset_id: int | None = Field(None)


class GenerateLabelSpaceResponse(BaseModel):
    success: bool = True
    draft: LabelSpaceDraft


class LabelSpaceConfigResponse(BaseModel):
    """Tells the frontend whether generation is available and which model is used."""
    enabled: bool
    model: str | None = None
    source: str | None = Field(
        None, description="Whose key would be used: 'personal', 'organization' or 'instance'.")
