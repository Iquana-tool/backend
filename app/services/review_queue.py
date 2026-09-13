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
* **Ordering** — a scoring strategy from `SORT_STRATEGIES` maps each candidate to a
  float; the queue is the candidates sorted by that score. "Hierarchy" (score =
  nesting depth, so root instances come first) is the default. Active-learning
  orderings plug in here: registering a new strategy is all it takes for it to
  show up in the setup page's dropdown, the API contract does not change.

Queues are snapshots, not reservations: nothing is locked server-side, and an item
that someone else approves mid-session simply becomes a no-op when acted on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
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
    ReviewSortDirection,
    ReviewSortStrategyOption,
    ReviewSummary,
)


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


#: A scoring strategy: candidate -> sort key. Lower scores are served first when
#: the direction is ascending. Strategies must be pure and cheap — they run once
#: per candidate at queue-build time.
ScoreFn = Callable[[QueueCandidate], float]


@dataclass(frozen=True)
class SortStrategy:
    key: str
    label: str
    description: str
    score: ScoreFn


SORT_STRATEGIES: dict[str, SortStrategy] = {}


def register_strategy(key: str, label: str, description: str):
    """Register a queue ordering. Registered keys appear in the setup UI."""

    def decorator(fn: ScoreFn) -> ScoreFn:
        SORT_STRATEGIES[key] = SortStrategy(key=key, label=label,
                                            description=description, score=fn)
        return fn

    return decorator


@register_strategy(
    "hierarchy",
    "By hierarchy",
    "Root instances first, then their children — verify containers before contents.",
)
def _score_by_depth(candidate: QueueCandidate) -> float:
    return float(candidate.depth)


@register_strategy(
    "uncertainty",
    "By model confidence",
    "Least confident instances first. Manual annotations (confidence 1.0) come last.",
)
def _score_by_confidence(candidate: QueueCandidate) -> float:
    return float(candidate.confidence)


def strategy_options() -> list[ReviewSortStrategyOption]:
    return [
        ReviewSortStrategyOption(key=s.key, label=s.label, description=s.description)
        for s in SORT_STRATEGIES.values()
    ]


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

    return [
        QueueCandidate(
            contour_id=row.id,
            mask_id=row.mask_id,
            image_id=row.image_id,
            label_id=row.label_id,
            parent_id=row.parent_id,
            depth=depth(row.id),
            confidence=row.confidence_score,
            reviewed=bool(row.reviewed),
        )
        for row in rows
    ]


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
                db: Session) -> ReviewQueueRead:
    """Build the ordered work list for one review session."""
    strategy = SORT_STRATEGIES.get(request.sort_strategy)
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown sort strategy '{request.sort_strategy}'. "
                   f"Available: {sorted(SORT_STRATEGIES)}.",
        )

    candidates = _collect_candidates(dataset_id, db,
                                     only_submitted=request.only_submitted)
    todo = [candidate for candidate in candidates
            if candidate.eligible(request.include_reviewed)]
    descending = request.direction is ReviewSortDirection.DESCENDING

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
                   reverse=not descending)
        return ReviewQueueRead(granularity=request.granularity,
                               sort_strategy=request.sort_strategy,
                               direction=request.direction,
                               include_reviewed=request.include_reviewed,
                               total=len(items), images=items)

    if request.granularity is ReviewGranularity.CUSTOM:
        _validate_labels(dataset_id, request.label_ids, db)
        wanted = set(request.label_ids)
        todo = [candidate for candidate in todo if candidate.label_id in wanted]

    scored = [(strategy.score(candidate), candidate) for candidate in todo]
    # Tiebreakers keep the order stable and group work naturally: same score ->
    # same image together, then parents before their children.
    scored.sort(key=lambda pair: (pair[0], pair[1].image_id,
                                  pair[1].depth, pair[1].contour_id),
                reverse=descending)

    items = [
        ReviewQueueInstanceItem(
            contour_id=candidate.contour_id,
            mask_id=candidate.mask_id,
            image_id=candidate.image_id,
            label_id=candidate.label_id,
            parent_id=candidate.parent_id,
            depth=candidate.depth,
            score=score,
        )
        for score, candidate in scored
    ]
    return ReviewQueueRead(granularity=request.granularity,
                           sort_strategy=request.sort_strategy,
                           direction=request.direction,
                           include_reviewed=request.include_reviewed,
                           total=len(items), instances=items)


def summarize(dataset_id: int, db: Session) -> ReviewSummary:
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
        strategies=strategy_options(),
    )
