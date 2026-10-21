"""How unsure a model is about each image, for the active-learning annotation queue.

One row per (image, model). A retrained model has a new registry key, so its scores are
new rows and never mix with the old model's. Rows die with their image.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ImageScores(database):
    __tablename__ = "image_scores"
    __table_args__ = (
        UniqueConstraint("image_id", "model_key", name="uq_image_scores_image_model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"),
                      nullable=False, index=True)
    model_key = Column(String(255), nullable=False)
    #: Higher means more unsure (see app.services.al_ranking.image_uncertainty).
    score = Column(Float, nullable=False)
    num_detections = Column(Integer, nullable=False, default=0)
    #: Every detection's confidence, so a score can be recomputed without the model.
    #: NULL on rows written before it was kept; those count as not scored.
    confidences = Column(JSON, nullable=True)
    scored_at = Column(DateTime, nullable=False, default=_utcnow)
