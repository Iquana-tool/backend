"""Schemas for the annotation queue: the persisted order an annotator works in.

The queue lets an annotator pick how a dataset's images are sorted and filtered before the
editor opens, and the choice is saved per (dataset, user) so re-entering resumes it. Sorting
methods come from a registry (see ``app.services.queue_methods``) that describes itself here,
so a new method appears in the builder without touching this contract.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.queue import Direction, Need, QueueMethodOption, RecommendedQueue, SortSettings

__all__ = ["Direction", "Need", "QueueSettings", "QueueMethodOption"]


class QueueSettings(SortSettings):
    """How to sort and filter the annotation queue; sent to prepare and build, saved with the
    queue. ``max_confidence`` here is an image's most confident detection and needs a trained
    model."""

    embedding_model: str | None = Field(
        None, description="Embedding model registry key, for methods that need one."
    )
    trained_model: str | None = Field(
        None, description="Trained instance-segmentation model, for methods or filters that need one."
    )
    show_builder: bool = Field(
        True, description="False skips the builder next time and opens this queue directly."
    )


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
    methods: list[QueueMethodOption] = Field(default_factory=list)
    authors: list[str] = Field(
        default_factory=list, description="Other people who drew here, for the 'By' filter."
    )
    recommended: RecommendedQueue | None = Field(
        None, description="The settings a curator recommends for this dataset, if any."
    )


class QueuePrepJobRead(BaseModel):
    """Progress of a background job filling the data an ordering needs."""

    id: int
    status: str
    total: int
    done: int
    failed: int
