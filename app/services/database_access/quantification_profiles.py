"""Database-access layer for quantification profiles (Step 5).

CRUD over the ``quantification_profiles`` table plus the default-profile definition. A
profile is stored as a single row with its entry list in a JSON column; these helpers
translate between the ORM row and the toolbox ``QuantificationProfile`` schema and enforce
the invariants the routes rely on (at most one default per dataset; a dataset always has a
default once queried).

Default profile: the four GEOMETRY metrics (area / perimeter / circularity /
max_diameter) applied to ALL labels (``label_ids=None``). This reproduces exactly what the
quantification page showed before profiles existed, so auto-creating it is backward
compatible.
"""
from logging import getLogger

from iquana_toolbox.schemas.database.quantification_profile import (
    ProfileEntry,
    QuantificationProfile,
)
from sqlalchemy.orm import Session

from app.database.quantification_profiles import QuantificationProfiles
from app.services.quantification import GEOMETRY_METRIC_KEYS

logger = getLogger(__name__)

DEFAULT_PROFILE_NAME = "Default (geometry)"


def default_profile_entries() -> list[ProfileEntry]:
    """The default profile's entries: the geometry metrics on all labels."""
    return [ProfileEntry(metric_key=key, params={}, label_ids=None) for key in GEOMETRY_METRIC_KEYS]


def _ensure_single_default(db: Session, dataset_id: int, keep_id: int | None) -> None:
    """Unset ``is_default`` on every profile of the dataset except ``keep_id``."""
    others = (
        db.query(QuantificationProfiles)
        .filter(QuantificationProfiles.dataset_id == dataset_id, QuantificationProfiles.is_default.is_(True))
    )
    for profile in others.all():
        if profile.id != keep_id:
            profile.is_default = False


def get_or_create_default_profile(db: Session, dataset_id: int) -> QuantificationProfiles:
    """Return the dataset's default profile, creating it if the dataset has none.

    Creates the geometry-on-all-labels default (see :func:`default_profile_entries`) if no
    profile is marked default yet, so an existing dataset behaves exactly as before the
    first time the profiles endpoint is hit.
    """
    existing = (
        db.query(QuantificationProfiles)
        .filter(QuantificationProfiles.dataset_id == dataset_id, QuantificationProfiles.is_default.is_(True))
        .first()
    )
    if existing is not None:
        return existing

    schema = QuantificationProfile(
        dataset_id=dataset_id,
        name=DEFAULT_PROFILE_NAME,
        is_default=True,
        entries=default_profile_entries(),
    )
    row = QuantificationProfiles.from_schema(schema)
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("Auto-created default quantification profile %s for dataset %s.", row.id, dataset_id)
    return row


def list_profiles(db: Session, dataset_id: int) -> list[QuantificationProfile]:
    """List all profiles for a dataset, ensuring a default exists first."""
    get_or_create_default_profile(db, dataset_id)
    rows = (
        db.query(QuantificationProfiles)
        .filter(QuantificationProfiles.dataset_id == dataset_id)
        .order_by(QuantificationProfiles.id)
        .all()
    )
    return [row.to_schema() for row in rows]


def get_profile(db: Session, dataset_id: int, profile_id: int) -> QuantificationProfiles | None:
    """Fetch one profile row scoped to the dataset (None if not found)."""
    return (
        db.query(QuantificationProfiles)
        .filter(QuantificationProfiles.id == profile_id, QuantificationProfiles.dataset_id == dataset_id)
        .first()
    )


def create_profile(db: Session, schema: QuantificationProfile) -> QuantificationProfile:
    """Insert a new profile. If it is default, unsets the previous default."""
    row = QuantificationProfiles.from_schema(schema)
    db.add(row)
    db.flush()
    if row.is_default:
        _ensure_single_default(db, row.dataset_id, keep_id=row.id)
    db.commit()
    db.refresh(row)
    return row.to_schema()


def update_profile(
    db: Session,
    row: QuantificationProfiles,
    schema: QuantificationProfile,
) -> QuantificationProfile:
    """Update name / entries / is_default on an existing profile row.

    Setting ``is_default`` True unsets it on every other profile of the dataset.
    """
    row.name = schema.name
    row.entries = schema.entries_as_json()
    row.is_default = schema.is_default
    db.flush()
    if row.is_default:
        _ensure_single_default(db, row.dataset_id, keep_id=row.id)
    db.commit()
    db.refresh(row)
    return row.to_schema()


def delete_profile(db: Session, row: QuantificationProfiles) -> None:
    """Delete a profile. If it was the default, promote the oldest remaining one.

    A dataset should always keep a usable default, so deleting the current default
    re-marks the lowest-id remaining profile as default (if any remain). Deleting the last
    profile is allowed - the next ``list_profiles`` call re-creates the geometry default.
    """
    dataset_id = row.dataset_id
    was_default = bool(row.is_default)
    db.delete(row)
    db.flush()
    if was_default:
        remaining = (
            db.query(QuantificationProfiles)
            .filter(QuantificationProfiles.dataset_id == dataset_id)
            .order_by(QuantificationProfiles.id)
            .first()
        )
        if remaining is not None:
            remaining.is_default = True
    db.commit()
