"""Measurements a dataset defines for itself, computed from the direct ones.

The metric registry holds the *direct* measurements: area, perimeter, colour, the
number of children -- things code computes from an outline and its image. A derived
measurement is a formula over those that only means something to one research group:
the choroidal vascularity index (vessel area over choroid area) on an OCT dataset,
polyp density (polyps per coral area) on a reef survey. Neither belongs in every
dataset's picker, so each dataset declares its own here.

Only the definition is stored. Values are computed when a quantification is read
(``app.services.derived_metrics``): every input is already a stored metric, so there is
nothing to keep fresh.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DerivedMetrics(database):
    """One derived measurement of one dataset.

    ``key`` names the measurement in exports and in other formulas, so it is fixed once
    created. ``label_ids`` are the labels it is reported on (``None`` for all).
    ``expression`` is the JSON formula tree validated by
    ``app.services.derived_metrics.Expression``; labels inside it are referenced by id, so
    renaming one does not break a formula.
    """

    __tablename__ = "derived_metrics"
    __table_args__ = (
        UniqueConstraint("dataset_id", "key", name="uq_derived_metrics_dataset_key"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    key = Column(String(48), nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(String(1024), nullable=False, default="")
    label_ids = Column(JSON, nullable=True)
    expression = Column(JSON, nullable=False)
    created_by = Column(
        String, ForeignKey("users.username", ondelete="SET NULL", onupdate="CASCADE"), nullable=True
    )
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)
