"""Schemas shared by the annotation and review queues: sorting methods and sort settings.

Both queues sort with methods from one registry (``app.services.queue_methods``). A method
sorts images (annotation queue) or objects (review queue), and describes itself here so a
new method appears in the builder without touching either queue's contract.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Direction = Literal["ascending", "descending"]
#: What a method can need prepared before it can rank: vectors from an embedding model, or
#: a trained model's detections.
Need = Literal["embedding_model", "trained_model"]
#: What a method sorts: whole images (a scan counts as one) or single objects.
Sorts = Literal["images", "objects"]
#: Whose work to keep: people relative to the person building the queue (so a recommended
#: queue means the same for everyone who uses it), or one person or model by name.
By = Literal["everyone", "mine", "others", "person", "model"]

#: The two queues, as keys of a dataset's recommended queues.
QueueKind = Literal["annotation", "review"]

#: Short forms the review queue used before both queues shared one vocabulary.
_DIRECTION_ALIASES = {"asc": "ascending", "desc": "descending"}


class QueueMethodOption(BaseModel):
    """One sorting method, as the builder shows it."""

    key: str
    label: str
    needs: list[Need] = Field(default_factory=list)
    #: Directions the method supports; the first is its default.
    directions: list[Direction]
    #: Direction -> what that order puts first.
    descriptions: dict[Direction, str]


class SortSettings(BaseModel):
    """How to sort and filter a queue: the part both queues share."""

    model_config = ConfigDict(populate_by_name=True)

    method: str = Field(..., description="Key of a sorting method from the summary.")
    direction: Direction | None = Field(
        None, description="Sort direction; None uses the method's default (its first)."
    )
    round_robin_with: str | None = Field(
        None, description="A second method to alternate with, one pick from each in turn."
    )
    metadata_filters: dict[str, list[str] | dict] = Field(
        default_factory=dict, description="Image metadata filter, as in the gallery."
    )
    balance_by: str | None = Field(
        None,
        description="Metadata key to take turns over, e.g. species: one item of each value in "
                    "turn, every value in the method's order, so no one works through one "
                    "species to the end before the next. Items without the key form their "
                    "own group.",
    )
    max_confidence: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Keep only items whose confidence is below this value.",
    )
    by: By = Field(
        "everyone",
        description="Whose work to keep: 'mine' (I drew on it), 'others' (I did not), "
                    "'person' (``by_person`` drew on it) or 'model' (``by_model`` made it). "
                    "An image counts as someone's when they made at least one object on it; "
                    "an object, when they made it. A model's object stays the model's after "
                    "a person edits or accepts it.",
    )
    by_person: str | None = Field(None, description="Username, for ``by='person'``.")
    by_model: str | None = Field(
        None, description="Model name as objects record it (``added_by``), for ``by='model'``."
    )

    @field_validator("direction", mode="before")
    @classmethod
    def _accept_short_directions(cls, value):
        # Older review clients send "asc" / "desc"; they keep working.
        return _DIRECTION_ALIASES.get(value, value)

    @model_validator(mode="after")
    def _person_needs_a_name(self):
        if self.by == "person" and not self.by_person:
            raise ValueError("Choose whose work to keep.")
        if self.by == "model" and not self.by_model:
            raise ValueError("Choose which model's work to keep.")
        return self


class RecommendedQueueWrite(BaseModel):
    """Settings a curator recommends for one queue of a dataset, and why."""

    settings: dict[str, Any] = Field(
        ..., description="The queue's builder settings: QueueSettings for the annotation queue, "
                         "ReviewRecommendedSettings for the review queue."
    )
    reason: str = Field(..., min_length=1, max_length=300,
                        description="One line the builder shows next to the recommendation.")


class RecommendedQueue(RecommendedQueueWrite):
    updated_by: str | None = None
    updated_at: datetime | None = None
