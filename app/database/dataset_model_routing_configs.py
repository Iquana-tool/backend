"""Persistent task-aware model routing configuration per dataset.

Stores one routing policy per dataset with canonical bindings for task defaults
and label-specific overrides. Used for interactive canvas selectors, per-label
suggestions, and batch preselection.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import relationship

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetModelRoutingConfigs(database):
    """A dataset-wide task-aware model routing policy."""

    __tablename__ = "dataset_model_routing_configs"

    # One policy row per dataset; dataset_id is the primary key.
    dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True
    )

    # Canonical list of ModelRoutingBinding JSON dictionaries.
    bindings = Column(JSON, nullable=False, default=list)

    # Nullable FK to users: if the user account is deleted, the dataset policy is preserved.
    updated_by = Column(
        String, ForeignKey("users.username", ondelete="SET NULL"), nullable=True
    )

    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    dataset = relationship("Datasets", foreign_keys=[dataset_id], passive_deletes=True)
    updater = relationship("Users", foreign_keys=[updated_by], passive_deletes=True)
