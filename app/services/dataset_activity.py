"""Who did what on a dataset, per user, over a time window.

Read from the annotation data itself, not from the activity log: the summary has
to work on every deployment, including those where event capture is locked off.
The price is that it can only count what the database still holds -- an object
annotated and later deleted is gone -- and only what carries a timestamp. Reviews
and finished masks recorded before ``reviewed_at`` / ``fully_annotated_at``
existed have none, so they appear under "all time" and in no window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.database.contours import Contours, reviewer_contour_association
from app.database.image_calibrations import ImageCalibrations
from app.database.images import Images
from app.database.masks import Masks
from app.database.rejections import AnnotationRejections

#: `added_by` values that mean a person drew the geometry. Anything else names
#: the model that produced it.
MANUAL_SOURCES = ("User", "user", "manual", "")


@dataclass
class UserActivity:
    username: str
    annotated: int = 0
    annotated_ai: int = 0
    finished: int = 0
    reviewed: int = 0
    sent_back: int = 0
    resolved: int = 0
    calibrated: int = 0
    last_active: datetime | None = None

    def touch(self, when: datetime | None) -> None:
        if when is not None and (self.last_active is None or when > self.last_active):
            self.last_active = when


def get_dataset_activity(dataset_id: int, since: datetime | None, db: Session) -> list[dict]:
    """Per-user counts on `dataset_id` since `since` (None = all time).

    One grouped query per activity, each scoped to the dataset through its images.
    Users are ordered by their most recent action, then by name.
    """
    users: dict[str, UserActivity] = {}

    def row(username: str) -> UserActivity:
        return users.setdefault(username, UserActivity(username=username))

    def window(column):
        return [column >= since] if since is not None else []

    dataset_masks = (
        select(Masks.id).join(Images, Images.id == Masks.image_id)
        .where(Images.dataset_id == dataset_id)
    )

    # Annotated: surviving, non-temporary contours authored by the user. "With AI" comes
    # from the recorded origin; objects created before origins were recorded fall back
    # to `added_by`, which names the model for most (but not all) AI tools.
    ai_flag = case(
        (Contours.origin.in_(("manual", "import")), 0),
        (Contours.origin.is_not(None), 1),
        (func.coalesce(Contours.added_by, "").in_(MANUAL_SOURCES), 0),
        else_=1,
    )
    for username, total, ai, last in db.execute(
        select(Contours.author_username, func.count(), func.sum(ai_flag), func.max(Contours.created_at))
        .where(Contours.mask_id.in_(dataset_masks), Contours.temporary.is_(False),
               Contours.author_username.is_not(None), *window(Contours.created_at))
        .group_by(Contours.author_username)
    ):
        entry = row(username)
        entry.annotated, entry.annotated_ai = total, int(ai or 0)
        entry.touch(last)

    # Finished: masks currently marked finished, credited to whoever finished them.
    for username, total, last in db.execute(
        select(Masks.fully_annotated_by, func.count(), func.max(Masks.fully_annotated_at))
        .where(Masks.id.in_(dataset_masks), Masks.fully_annotated.is_(True),
               Masks.fully_annotated_by.is_not(None), *window(Masks.fully_annotated_at))
        .group_by(Masks.fully_annotated_by)
    ):
        entry = row(username)
        entry.finished = total
        entry.touch(last)

    # Reviewed: approvals given on the dataset's contours.
    assoc = reviewer_contour_association.c
    for username, total, last in db.execute(
        select(assoc.reviewer_id, func.count(), func.max(assoc.reviewed_at))
        .join(Contours, Contours.id == assoc.contour_id)
        .where(Contours.mask_id.in_(dataset_masks), *window(assoc.reviewed_at))
        .group_by(assoc.reviewer_id)
    ):
        entry = row(username)
        entry.reviewed = total
        entry.touch(last)

    # Sent back and resolved: both sides of a rejection.
    for column_by, column_at, field in (
        (AnnotationRejections.created_by, AnnotationRejections.created_at, "sent_back"),
        (AnnotationRejections.resolved_by, AnnotationRejections.resolved_at, "resolved"),
    ):
        for username, total, last in db.execute(
            select(column_by, func.count(), func.max(column_at))
            .where(AnnotationRejections.mask_id.in_(dataset_masks),
                   column_by.is_not(None), column_at.is_not(None), *window(column_at))
            .group_by(column_by)
        ):
            entry = row(username)
            setattr(entry, field, total)
            entry.touch(last)

    # Calibrated: distinct images the user set a calibration on.
    for username, total, last in db.execute(
        select(ImageCalibrations.created_by,
               func.count(func.distinct(ImageCalibrations.image_id)),
               func.max(ImageCalibrations.created_at))
        .join(Images, Images.id == ImageCalibrations.image_id)
        .where(Images.dataset_id == dataset_id, ImageCalibrations.created_by.is_not(None),
               *window(ImageCalibrations.created_at))
        .group_by(ImageCalibrations.created_by)
    ):
        entry = row(username)
        entry.calibrated = total
        entry.touch(last)

    ordered = sorted(
        users.values(),
        key=lambda u: (u.last_active is None, -(u.last_active.timestamp()) if u.last_active else 0,
                       u.username),
    )
    return [
        {**asdict(u), "last_active": u.last_active.isoformat() if u.last_active else None}
        for u in ordered
    ]
