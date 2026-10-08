"""Every AI output that became an object, as it looked when the model produced it.

One row per suggested object: prompted segmentation, refinement, within-image
instance suggestion, instance segmentation, cross-image suggestion and batch
inference. The row keeps the geometry the model returned, so what the person later
did to it can be measured against the original: whether they kept it unchanged,
edited it, deleted it or approved it, and how far the final outline moved from the
suggestion.

Like ``user_events``, the table has no foreign keys. Study data must stay
analysable after the object, image or dataset it describes is deleted -- a deleted
suggestion is exactly the case that has to survive -- so ids are kept as plain
columns and resolved at analysis time.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Index, Integer, JSON, String

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SuggestionSource:
    """What produced a suggestion. Also the values of ``contours.origin`` for AI objects."""

    PROMPTED = "prompted"
    REFINE = "refine"
    INSTANCE_SUGGESTION = "instance_suggestion"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    CROSS_IMAGE = "cross_image"
    BATCH_INFERENCE = "batch_inference"


class AiSuggestions(database):
    __tablename__ = "ai_suggestions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    #: Rows from the same model call share a run id (a suggestion run that found
    #: thirty instances is one run).
    run_id = Column(String(64), nullable=False, index=True)
    source = Column(String(32), nullable=False)
    model_key = Column(String(255), nullable=True)
    #: Who asked for it (for batch runs: who started the run).
    username = Column(String, nullable=True, index=True)
    dataset_id = Column(Integer, nullable=True, index=True)
    image_id = Column(Integer, nullable=True, index=True)
    mask_id = Column(Integer, nullable=True, index=True)
    #: The object the suggestion became. Not a foreign key: it must outlive the object.
    contour_id = Column(Integer, nullable=True, index=True)
    label_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    #: The outline exactly as the model returned it (normalised coordinates).
    x = Column(JSON, nullable=False)
    y = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    #: First time a person changed the outline by hand (not by another AI refine).
    edited_at = Column(DateTime, nullable=True)
    #: When the object was deleted, i.e. the suggestion was rejected.
    deleted_at = Column(DateTime, nullable=True)
    #: First approval of the object.
    reviewed_at = Column(DateTime, nullable=True)
    #: A later AI refinement replaced this suggestion's outline.
    superseded_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_ai_suggestions_mask_created", "mask_id", "created_at"),
    )

    def as_dict(self) -> dict:
        return {
            "id": self.id, "run_id": self.run_id, "source": self.source,
            "model_key": self.model_key, "username": self.username,
            "dataset_id": self.dataset_id, "image_id": self.image_id, "mask_id": self.mask_id,
            "contour_id": self.contour_id, "label_id": self.label_id,
            "confidence": self.confidence, "x": self.x, "y": self.y,
            **{field: (getattr(self, field).isoformat() if getattr(self, field) else None)
               for field in ("created_at", "edited_at", "deleted_at", "reviewed_at",
                             "superseded_at")},
        }
