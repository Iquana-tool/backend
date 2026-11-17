"""Building review queues: which annotations need eyes on them, in what order.

The queue has two halves:

* **Candidate collection** — every non-temporary contour of the dataset whose mask
  is review-ready (submitted and without open rejections, unless the caller asks
  for in-progress work too). By default only contours nobody has approved yet
  qualify; with `include_reviewed` every contour qualifies, the caller's own past
  approvals included — a solo reviewer must be able to re-sweep their own work.
  Re-accepting is a no-op (approvals are a set); rejecting a previously approved
  contour withdraws the rejecting reviewer's own approval (see
  `database_access.rejections.reject`), so a changed mind overwrites the old
  verdict instead of coexisting with it.
* **Ordering** — an object method from ``app.services.queue_methods`` scores each
  candidate, and ``app.services.queue_core`` sorts, alternates two methods (round
  robin), filters and takes turns by a metadata key, exactly as for the annotation
  queue. "Hierarchy" (score = nesting depth, so root instances come first) is the
  default. Registering a new object method is all it takes for it to show up in the
  setup page, the API contract does not change.

Queues are snapshots, not reservations: nothing is locked server-side, and an item
that someone else approves mid-session simply becomes a no-op when acted on.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.rejections import AnnotationRejections
from app.schemas.review import (
    ReviewGranularity,
    ReviewQueueImageItem,
    ReviewQueueInstanceItem,
    ReviewQueueRead,
    ReviewQueueRequest,
    ReviewSummary,
)
from app.services import queue_core
from app.services.queue_methods import OBJECTS, MethodContext


@dataclass
class QueueCandidate:
    """One contour under consideration, with everything a scorer may need."""

    contour_id: int
    mask_id: int
    image_id: int
    label_id: int | None
    parent_id: int | None
    depth: int
    confidence: float
    reviewed: bool  # True once anyone has approved it.
    author: str | None = None
    model: str | None = None  # The model that made it; None when a person drew it.

    @property
    def key(self) -> int:
        """The unit key the shared sorting code ranks by."""
        return self.contour_id

    @property
    def pending(self) -> bool:
        """Untouched by any reviewer — the default queue's eligibility."""
        return not self.reviewed

    def eligible(self, include_reviewed: bool) -> bool:
        """Whether this contour belongs in the queue.

        `include_reviewed` re-opens everything, own approvals included, so a solo
        reviewer can re-sweep work they already signed off on.
        """
        return include_reviewed or not self.reviewed


# -- Candidate collection ----------------------------------------------------

def _open_rejection_mask_ids(dataset_id: int, db: Session) -> set[int]:
    rows = (
        db.query(AnnotationRejections.mask_id)
        .join(Masks, Masks.id == AnnotationRejections.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id,
                AnnotationRejections.resolved_at.is_(None))
        .distinct()
    )
    return {mask_id for (mask_id,) in rows}


def _collect_candidates(dataset_id: int, db: Session,
                        only_submitted: bool = True) -> list[QueueCandidate]:
    """Every contour of the dataset's review-ready masks, with hierarchy depth.

    Reviewed contours are always included (queues filter on eligibility later)
    because depth can only be computed against the full tree — a pending child
    may hang under an already approved parent.
    """
    query = (
        db.query(
            Contours.id,
            Contours.mask_id,
            Contours.parent_id,
            Contours.label_id,
            Contours.confidence_score,
            Contours.author_username,
            Contours.added_by,
            Masks.image_id,
            Contours.reviewed_by.any().label("reviewed"),
        )
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id,
                Contours.temporary.is_(False))
    )
    if only_submitted:
        query = query.filter(Masks.fully_annotated.is_(True))
    rows = query.all()

    # A rejected mask is the annotator's again — keep it out of the queue.
    rejected_masks = _open_rejection_mask_ids(dataset_id, db)
    rows = [row for row in rows if row.mask_id not in rejected_masks]

    # Depth by walking parent links in memory; one pass per level.
    parent_of = {row.id: row.parent_id for row in rows}
    depth_of: dict[int, int] = {}

    def depth(contour_id: int) -> int:
        if contour_id in depth_of:
            return depth_of[contour_id]
        seen = []
        current = contour_id
        # Walk up until a known depth or a root; `parent_of.get` treats a parent
        # outside the candidate set (e.g. filtered mask) as a root.
        while current is not None and current not in depth_of:
            seen.append(current)
            current = parent_of.get(current)
        base = depth_of[current] if current is not None else -1
        for offset, node in enumerate(reversed(seen), start=1):
            depth_of[node] = base + offset
        return depth_of[contour_id]

    candidates = [
        QueueCandidate(
            contour_id=row.id,
            mask_id=row.mask_id,
            image_id=row.image_id,
            label_id=row.label_id,
            parent_id=row.parent_id,
            depth=depth(row.id),
            confidence=row.confidence_score,
            reviewed=bool(row.reviewed),
            author=row.author_username,
            model=queue_core.model_of(row.added_by),
        )
        for row in rows
    ]
    # Methods sort stably, so this is the order among equal scores: same image together,
    # parents before their children.
    candidates.sort(key=lambda c: (c.image_id, c.depth, c.contour_id))
    return candidates


# -- Queue building ----------------------------------------------------------

def _validate_labels(dataset_id: int, label_ids: list[int], db: Session) -> None:
    known = {
        label_id for (label_id,) in
        db.query(Labels.id).filter(Labels.dataset_id == dataset_id,
                                   Labels.id.in_(label_ids))
    }
    unknown = set(label_ids) - known
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Labels {sorted(unknown)} do not belong to dataset {dataset_id}.",
        )


def build_queue(dataset_id: int, request: ReviewQueueRequest,
                db: Session, username: str | None = None) -> ReviewQueueRead:
    """Build the ordered work list for one review session; ``username`` is the reviewer, for
    the "By" filter's "mine" and "others"."""
    methods = queue_core.chosen_methods(OBJECTS, request)
    direction = request.direction or methods[0].directions[0]

    candidates = _collect_candidates(dataset_id, db,
                                     only_submitted=request.only_submitted)
    candidates = queue_core.filter_by_metadata(db, dataset_id, candidates,
                                               request.metadata_filters,
                                               lambda c: (c.image_id,))
    candidates = queue_core.filter_by_author(candidates, request, username,
                                             lambda c: {c.author} if c.author else set(),
                                             lambda c: {c.model} if c.model else set())
    echo = dict(granularity=request.granularity, method=methods[0].key,
                direction=direction, include_reviewed=request.include_reviewed)

    if request.granularity is ReviewGranularity.IMAGES:
        by_image: dict[int, list[QueueCandidate]] = {}
        for candidate in candidates:
            by_image.setdefault(candidate.image_id, []).append(candidate)
        items = [
            ReviewQueueImageItem(
                image_id=image_id,
                mask_id=group[0].mask_id,
                pending_instances=sum(
                    1 for c in group if c.eligible(request.include_reviewed)),
                total_instances=len(group),
            )
            for image_id, group in by_image.items()
        ]
        # Only images with something left to look at, most pending work first
        # (ascending flips that for reviewers who want the quick wins).
        items = [item for item in items if item.pending_instances > 0]
        items.sort(key=lambda item: (item.pending_instances, item.image_id),
                   reverse=direction == "ascending")
        return ReviewQueueRead(**echo, total=len(items), images=items)

    todo = [candidate for candidate in candidates
            if candidate.eligible(request.include_reviewed)]
    if request.granularity is ReviewGranularity.CUSTOM:
        _validate_labels(dataset_id, request.label_ids, db)
        wanted = set(request.label_ids)
        todo = [candidate for candidate in todo if candidate.label_id in wanted]
    todo = queue_core.filter_by_confidence(todo, request.max_confidence,
                                           lambda c: c.confidence)

    ranked = queue_core.sort_units(methods, request, MethodContext(db, todo))
    if request.balance_by:
        ranked = queue_core.balance(db, dataset_id, ranked, request.balance_by,
                                    {c.key: c.image_id for c in todo})

    by_key = {candidate.key: candidate for candidate in todo}
    items = [
        ReviewQueueInstanceItem(
            contour_id=by_key[key].contour_id,
            mask_id=by_key[key].mask_id,
            image_id=by_key[key].image_id,
            label_id=by_key[key].label_id,
            parent_id=by_key[key].parent_id,
            depth=by_key[key].depth,
            score=score,
            reason=reason,
        )
        for key, reason, score in ranked
    ]
    return ReviewQueueRead(**echo, total=len(items), instances=items)


def summarize(dataset_id: int, db: Session, username: str | None = None) -> ReviewSummary:
    """The numbers behind "There are x instances to review"."""
    candidates = _collect_candidates(dataset_id, db, only_submitted=True)
    pending = [candidate for candidate in candidates if candidate.pending]
    reviewed = [candidate for candidate in candidates if candidate.reviewed]
    open_rejections = (
        db.query(AnnotationRejections)
        .join(Masks, Masks.id == AnnotationRejections.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id,
                AnnotationRejections.resolved_at.is_(None))
        .count()
    )
    return ReviewSummary(
        pending_instances=len(pending),
        pending_images=len({candidate.image_id for candidate in pending}),
        reviewed_instances=len(reviewed),
        open_rejections=open_rejections,
        methods=queue_core.method_options(OBJECTS),
        authors=queue_core.dataset_authors(db, dataset_id, exclude=username),
        models=queue_core.dataset_models(db, dataset_id),
        recommended=queue_core.recommended_queue(db, dataset_id, "review"),
        require_independent_review=bool(
            db.query(Datasets.require_independent_review).filter_by(id=dataset_id).scalar()),
    )
