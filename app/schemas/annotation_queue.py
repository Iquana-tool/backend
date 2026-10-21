"""Schemas for the annotation queue: the persisted order an annotator works in.

The queue lets an annotator pick how a dataset's images are ordered before the
editor opens — as uploaded, randomized, or an active-learning ordering — and
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
    model_key: str | None = Field(
        None,
        description="Registry key of the trained model that scores images. Required by "
                    "strategies flagged `needs_model`, ignored by the others.",
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
    needs_model: bool = Field(
        False, description="True when the ordering needs a trained model to score images."
    )


class AnnotationQueueRead(BaseModel):
    """A saved annotation queue: image ids in the order they will be worked."""

    strategy: str
    model_key: str | None = None
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
    saved_strategy: str | None = Field(
        None, description="Strategy of the saved queue, if any."
    )
    saved_model_key: str | None = Field(
        None, description="Model the saved queue was ranked with, if any."
    )
    strategies: list[AnnotationQueueStrategyOption] = Field(default_factory=list)


class QueuePrepJobRead(BaseModel):
    """Progress of a background job filling the data an ordering needs."""

    id: int
    status: str
    strategy: str
    model_key: str | None = None
    total: int
    done: int
    failed: int
