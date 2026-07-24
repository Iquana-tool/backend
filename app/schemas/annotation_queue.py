"""Schemas for the annotation queue: the persisted order an annotator works in.

The queue lets an annotator pick how a dataset's images are ordered before the
editor opens — as uploaded, randomized, or a future active-learning ordering — and
the choice is saved per (dataset, user) so re-entering resumes it. Ordering is
delegated to a strategy registry (see ``app.services.annotation_queue``) so that
active-learning scorers can be added without touching this request/response
contract, exactly as the review queue does for its own orderings.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AnnotationQueueRequest(BaseModel):
    """Request body for building (and saving) an annotation queue."""

    strategy: str = Field(
        ...,
        description="Key of the ordering strategy to build the queue with. The "
                    "available keys are listed in the annotation-queue summary; "
                    "only strategies flagged `available` can be built.",
    )


class AnnotationQueueStrategyOption(BaseModel):
    """One selectable queue ordering, for the builder's option list."""

    key: str
    label: str
    description: str
    available: bool = Field(
        True,
        description="False for orderings shown as a placeholder (e.g. active "
                    "learning) that cannot be built yet.",
    )


class AnnotationQueueRead(BaseModel):
    """A saved annotation queue: image ids in the order they will be worked."""

    strategy: str
    image_ids: list[int] = Field(default_factory=list)
    total: int
    updated_at: datetime | None = None


class AnnotationQueueSummary(BaseModel):
    """The dataset's annotation workload plus queue state, for the card and builder."""

    not_started: int = Field(..., description="Images whose mask is not started.")
    in_progress: int = Field(..., description="Images whose mask is in progress.")
    finished: int = Field(..., description="Images whose mask is finished.")
    total: int = Field(..., description="Total images in the dataset.")
    has_saved_queue: bool = Field(..., description="Whether the caller has a saved queue here.")
    saved_strategy: str | None = Field(
        None, description="Strategy of the saved queue, if any."
    )
    strategies: list[AnnotationQueueStrategyOption] = Field(default_factory=list)
