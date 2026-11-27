"""Recording and reading completeness checks (see `app.database.completeness`)."""
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.completeness import CompletenessChecks
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.rejections import AnnotationRejections
from app.schemas.review import (
    CompletenessAnswer,
    CompletenessCheckRead,
    CompletenessCreate,
    CompletenessItem,
    CompletenessVerdict,
    RejectionCreate,
    RejectionReason,
)
from app.services.database_access import rejections as rejections_db

#: The send-back note is capped by the rejection column.
_NOTE_LIMIT = 1000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def to_read(check: CompletenessChecks) -> CompletenessCheckRead:
    return CompletenessCheckRead(
        id=check.id,
        mask_id=check.mask_id,
        item=CompletenessItem(check.item),
        label_id=check.label_id,
        answer=CompletenessAnswer(check.answer),
        note=check.note,
        created_by=check.created_by,
        created_at=_as_utc(check.created_at),
        rejection_id=check.rejection_id,
    )


def verdict_of(answers: list[CompletenessAnswer]) -> CompletenessVerdict:
    """One word for a set of answers: any "incomplete" wins, then any "unsure"."""
    if CompletenessAnswer.INCOMPLETE in answers:
        return CompletenessVerdict.INCOMPLETE
    if CompletenessAnswer.UNSURE in answers:
        return CompletenessVerdict.UNSURE
    return CompletenessVerdict.COMPLETE


def _send_back_note(missing: list[str], note: str | None) -> str:
    text = " ".join(part for part in (f"Missing: {', '.join(missing)}.", (note or "").strip()) if part)
    return text[:_NOTE_LIMIT]


async def record(mask_id: int, body: CompletenessCreate, username: str,
                 db: Session) -> tuple[list[CompletenessChecks], CompletenessVerdict, AnnotationRejections | None]:
    """Store one reviewer's answers for an image.

    An "incomplete" answer also sends the image back to its annotator, as a
    mask-level rejection with the reason "objects are missing" naming the labels.
    The answers and the send-back are written in one transaction; the answers then
    point at the rejection they caused.

    Returns the stored rows, the image-level verdict and the rejection (or None).
    """
    dataset_id = (
        db.query(Images.dataset_id)
        .join(Masks, Masks.image_id == Images.id)
        .filter(Masks.id == mask_id)
        .scalar()
    )
    if dataset_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask not found.")

    label_ids = {entry.label_id for entry in body.answers if entry.label_id is not None}
    names = dict(
        db.query(Labels.id, Labels.name)
        .filter(Labels.id.in_(label_ids), Labels.dataset_id == dataset_id)
        .all()
    ) if label_ids else {}
    unknown = label_ids - set(names)
    if unknown:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Labels {sorted(unknown)} do not belong to this dataset.")

    note = (body.note or "").strip() or None
    created_at = _utcnow()
    checks = [
        CompletenessChecks(
            mask_id=mask_id,
            item=entry.item.value,
            label_id=entry.label_id,
            answer=entry.answer.value,
            note=note if entry.answer is CompletenessAnswer.INCOMPLETE else None,
            created_by=username,
            # One timestamp for the whole submission, so it can be read back as one.
            created_at=created_at,
        )
        for entry in body.answers
    ]
    db.add_all(checks)

    verdict = verdict_of([entry.answer for entry in body.answers])
    rejection = None
    if verdict is CompletenessVerdict.INCOMPLETE:
        missing = [
            names[entry.label_id] if entry.item is CompletenessItem.LABEL else "relationships"
            for entry in body.answers if entry.answer is CompletenessAnswer.INCOMPLETE
        ]
        # Commits the pending answers together with the send-back.
        rejection = await rejections_db.reject(
            mask_id,
            RejectionCreate(reason=RejectionReason.MISSING_OBJECTS, note=_send_back_note(missing, note)),
            username=username,
            db=db,
        )
        for check in checks:
            if check.answer == CompletenessAnswer.INCOMPLETE.value:
                check.rejection_id = rejection.id
    db.commit()
    for check in checks:
        db.refresh(check)
    return checks, verdict, rejection


def current_for_mask(mask_id: int, db: Session) -> list[CompletenessChecks]:
    """Each reviewer's latest answer per label (and for the relationships) on an image."""
    rows = (
        db.query(CompletenessChecks)
        .filter(CompletenessChecks.mask_id == mask_id, CompletenessChecks.withdrawn_at.is_(None))
        .order_by(CompletenessChecks.created_at.desc(), CompletenessChecks.id.desc())
        .all()
    )
    latest: dict[tuple, CompletenessChecks] = {}
    for row in rows:
        latest.setdefault((row.created_by, row.item, row.label_id), row)
    return list(latest.values())


def unsure_masks(dataset_id: int, db: Session, exclude_username: str | None = None) -> set[int]:
    """Masks whose most recent completeness answer was "unsure", awaiting a second look.

    The most recent submission decides: once someone else answers the image without
    being unsure, it is settled. ``exclude_username`` drops images the caller was
    unsure about themselves — the second look has to come from someone else.
    """
    rows = (
        db.query(CompletenessChecks.mask_id, CompletenessChecks.created_by,
                 CompletenessChecks.created_at, CompletenessChecks.answer)
        .join(Masks, Masks.id == CompletenessChecks.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id, CompletenessChecks.withdrawn_at.is_(None))
        .all()
    )
    latest: dict[int, tuple] = {}
    answers: dict[tuple, list[CompletenessAnswer]] = {}
    for mask, author, created_at, answer in rows:
        submission = (created_at, author or "")
        answers.setdefault((mask, submission), []).append(CompletenessAnswer(answer))
        if mask not in latest or submission > latest[mask]:
            latest[mask] = submission
    return {
        mask for mask, submission in latest.items()
        if verdict_of(answers[(mask, submission)]) is CompletenessVerdict.UNSURE
        and submission[1] != exclude_username
    }
