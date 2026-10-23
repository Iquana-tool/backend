"""Schemas for the annotation queue: the persisted order an annotator works in.

The queue lets an annotator pick how a dataset's images are sorted and filtered before the
editor opens, and the choice is saved per (dataset, user) so re-entering resumes it. Sorting
methods come from a registry (see ``app.services.queue_methods``) that describes itself here,
so a new method appears in the builder without touching this contract.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["ascending", "descending"]
#: What a method needs prepared before it can rank: vectors from an embedding model, or a
#: trained model's detections.
Need = Literal["embedding_model", "trained_model"]


class QueueSettings(BaseModel):
    """How to sort and filter a queue; sent to prepare and build, saved with the queue."""

    method: str = Field(..., description="Key of a sorting method from the summary.")
    direction: Direction | None = Field(
        None, description="Sort direction; None uses the method's default (its first)."
    )
    round_robin_with: str | None = Field(
        None, description="A second method to alternate with, one pick from each in turn."
    )
    embedding_model: str | None = Field(
        None, description="Embedding model registry key, for methods that need one."
    )
    trained_model: str | None = Field(
        None, description="Trained instance-segmentation model, for methods or filters that need one."
    )
    metadata_filters: dict[str, list[str] | dict] = Field(
        default_factory=dict, description="Image metadata filter, as in the gallery."
    )
    max_confidence: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Keep only images whose most confident detection is below this value. "
                    "Needs a trained model.",
    )
    show_builder: bool = Field(
        True, description="False skips the builder next time and opens this queue directly."
    )


class AnnotationQueueMethodOption(BaseModel):
    """One sorting method, as the builder shows it."""

    key: str
    label: str
    needs: list[Need] = Field(default_factory=list)
    #: Directions the method supports; the first is its default.
    directions: list[Direction]
    #: Direction -> what that order puts first.
    descriptions: dict[Direction, str]


class QueueModelOption(BaseModel):
    registry_key: str
    name: str


class QueueModelOptions(BaseModel):
    """Models the builder can offer for this dataset."""

    embedding: list[QueueModelOption] = Field(default_factory=list)
    trained: list[QueueModelOption] = Field(default_factory=list)


class AnnotationQueueRead(BaseModel):
    """A saved annotation queue: image ids in the order they will be worked."""

    settings: QueueSettings
    image_ids: list[int] = Field(default_factory=list)
    total: int
    #: {"<image id>": {"reason": str, "score": float | None}}; empty for plain orderings.
    details: dict[str, dict] = Field(default_factory=dict)
    updated_at: datetime | None = None


class AnnotationQueueSummary(BaseModel):
    """The dataset's annotation workload plus queue state, for the card and builder."""

    not_started: int = Field(..., description="Images whose mask is not started.")
    in_progress: int = Field(..., description="Images whose mask is in progress.")
    finished: int = Field(..., description="Images whose mask is finished.")
    total: int = Field(..., description="Total images in the dataset.")
    has_saved_queue: bool = Field(..., description="Whether the caller has a saved queue here.")
    saved_settings: QueueSettings | None = Field(
        None, description="Settings of the saved queue, if any."
    )
    methods: list[AnnotationQueueMethodOption] = Field(default_factory=list)


class QueuePrepJobRead(BaseModel):
    """Progress of a background job filling the data an ordering needs."""

    id: int
    status: str
    total: int
    done: int
    failed: int
