"""Whole-image vectors from any embedding model, for the annotation queue's vector orderings.

One row per (image, embedding model). Unlike ``embeddings`` (a fixed-width pgvector column
with a search index, used by cross-image retrieval), the vector here may have any length, so
any encoder can be used. The queue never searches this table: it loads one dataset's vectors
and ranks them in memory. Rows die with their image.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ImageVectors(database):
    __tablename__ = "image_vectors"
    __table_args__ = (
        UniqueConstraint("image_id", "model_key", name="uq_image_vectors_image_model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"),
                      nullable=False, index=True)
    #: Registry key of the embedding model in the ai-service, e.g. "dinov3" or "bioclip-2".
    model_key = Column(String(255), nullable=False)
    #: L2-normalized vector, as a JSON list of floats.
    vector = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
