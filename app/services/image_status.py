"""Where an image stands in the Calibrate -> Annotate -> Review workflow.

An image goes through three phases, and each one is tracked separately with the
same three states: ``not_started``, ``in_progress``, ``finished``. The overall
status of an image is a *function* of the three, not a fourth thing that can drift
away from them: finished when all three are, not started when none of them are, in
progress otherwise.

This replaces a single five-value lifecycle (``not_started`` / ``in_progress`` /
``rejected`` / ``reviewable`` / ``finished``) that conflated the phases. It could
only describe one dimension of progress at a time, so it had to rank states that
are not actually ordered -- a fully reviewed image that nobody ever calibrated
read as ``finished``, and an image calibrated but not yet annotated read as
``not_started``. Three independent axes say both.

Where each phase's answer comes from:

``calibrate``
    How many of the *registered* calibration kinds the image has set, via
    :func:`app.services.calibration.calibrated_counts`. All of them is finished;
    none is not started.
``annotate`` / ``review``
    ``Masks.annotate_status`` / ``Masks.review_status`` -- hybrid properties, so
    the same definition serves both Python objects and SQL filters. An image with
    no mask row at all has neither phase started.

Sending work back (an open rejection) is what resets the two mask phases; the
image's calibration is untouched by it, and instance-level state (unverified /
verified / sent back) is a separate axis that this module does not model.
"""
from logging import getLogger
from typing import Iterable, Sequence

from sqlalchemy.orm import Session

from app.database.images import Images
from app.database.masks import Masks
from app.services.calibration import calibrated_counts

logger = getLogger(__name__)

NOT_STARTED = "not_started"
IN_PROGRESS = "in_progress"
FINISHED = "finished"

#: The three states every phase (and the overall status) can be in, in order.
PHASE_STATES: tuple[str, ...] = (NOT_STARTED, IN_PROGRESS, FINISHED)

#: The three phases, in workflow order. Also the display order of the progress bars.
PHASES: tuple[str, ...] = ("calibrate", "annotate", "review")

PHASE_LABELS: dict[str, str] = {
    "calibrate": "Calibrated",
    "annotate": "Annotated",
    "review": "Reviewed",
}


def combine(calibrate: str, annotate: str, review: str) -> str:
    """The overall status implied by the three phase statuses.

    Deliberately strict at both ends: an image counts as finished only once every
    phase is, and as not started only while no phase has been touched. Everything
    between is in progress, which is the honest answer -- there is no useful way to
    rank "calibrated but unannotated" against "annotated but uncalibrated".
    """
    phases = (calibrate, annotate, review)
    if all(phase == FINISHED for phase in phases):
        return FINISHED
    if all(phase == NOT_STARTED for phase in phases):
        return NOT_STARTED
    return IN_PROGRESS


def calibrate_status_from_counts(calibrated: int, total: int) -> str:
    """Map a (kinds set, kinds registered) pair onto a phase state.

    ``total == 0`` -- no calibration kinds registered at all -- reads as finished
    rather than not started: there is nothing left to do, so the phase must not
    hold the image back from ever being finished.
    """
    if total == 0 or calibrated >= total:
        return FINISHED
    if calibrated == 0:
        return NOT_STARTED
    return IN_PROGRESS


def _phase_dict(calibrate: str, annotate: str, review: str) -> dict[str, str]:
    return {"calibrate": calibrate, "annotate": annotate, "review": review}


def status_for_images(
        db: Session,
        images: Sequence[Images],
        masks_by_image: dict[int, Masks] | None = None,
) -> dict[int, dict]:
    """Phase statuses for a batch of images, keyed by image id.

    Args:
        db: SQLAlchemy session.
        images: The images to report on.
        masks_by_image: Pre-fetched mask per image id, when the caller already has
            them (the gallery listing joins masks anyway). Looked up here when omitted.

    Returns:
        ``{image_id: {"status": overall, "phases": {...}, "mask_id": int | None}}``.
    """
    images = list(images)
    if not images:
        return {}

    if masks_by_image is None:
        masks_by_image = {}
        for mask in (
                db.query(Masks)
                .filter(Masks.image_id.in_([image.id for image in images]))
                .all()
        ):
            # One mask per image in practice; if a second ever appears, the first
            # one wins so the reported status is at least stable.
            masks_by_image.setdefault(mask.image_id, mask)

    counts = calibrated_counts(db, images)

    result: dict[int, dict] = {}
    for image in images:
        calibrated, total = counts.get(image.id, (0, 0))
        calibrate = calibrate_status_from_counts(calibrated, total)

        mask = masks_by_image.get(image.id)
        annotate = mask.annotate_status if mask is not None else NOT_STARTED
        review = mask.review_status if mask is not None else NOT_STARTED

        result[image.id] = {
            "mask_id": mask.id if mask is not None else None,
            "status": combine(calibrate, annotate, review),
            "phases": _phase_dict(calibrate, annotate, review),
            "calibrated_kinds": calibrated,
            "total_kinds": total,
        }
    return result


def status_for_image(db: Session, image: Images) -> dict:
    """Phase statuses for a single image. Thin wrapper over :func:`status_for_images`."""
    return status_for_images(db, [image])[image.id]


def status_for_mask(db: Session, mask: Masks) -> dict:
    """Phase statuses of the image a mask belongs to.

    The annotation workspace holds a mask id, not an image id, but the status it
    shows is the image's -- calibration is part of it.
    """
    return status_for_image(db, mask.image)


def empty_phase_counts() -> dict[str, dict[str, int]]:
    """A zeroed ``{phase: {state: 0}}`` table, plus the ``overall`` row."""
    return {
        phase: {state: 0 for state in PHASE_STATES}
        for phase in (*PHASES, "overall")
    }


def count_phases(statuses: Iterable[dict]) -> dict[str, dict[str, int]]:
    """Tally :func:`status_for_images` entries into ``{phase: {state: count}}``.

    Includes an ``overall`` row alongside the three phases so a caller can render
    the bars and the headline number from one payload.
    """
    counts = empty_phase_counts()
    for entry in statuses:
        for phase in PHASES:
            state = entry["phases"][phase]
            counts[phase][state] = counts[phase].get(state, 0) + 1
        counts["overall"][entry["status"]] = counts["overall"].get(entry["status"], 0) + 1
    return counts
