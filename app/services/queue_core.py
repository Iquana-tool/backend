"""What the annotation and review queues share: picking methods, sorting, filters, turns.

Each queue collects its own units (image units, or objects awaiting review) and decides what
goes last (finished work, work someone else started). Everything between is here, written
once:

* :func:`get_method` / :func:`method_options` -- the registry of
  ``app.services.queue_methods``, for one kind of unit.
* :func:`filter_by_metadata`, :func:`filter_by_confidence` and :func:`filter_by_author` --
  the builder's filters.
* :func:`sort_units` -- score with the chosen method in the chosen direction, or alternate
  two methods (round robin). Units a method cannot score follow its scored ones.
* :func:`balance` -- take turns between the values of a metadata key.

A ranked unit is ``(key, reason, score)``: the reason is the method that placed it (or why it
sits where it does) and the score is what that method gave it, for display.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.schemas.queue import QueueKind, QueueMethodOption, RecommendedQueue, SortSettings
from app.services import al_ranking
from app.services.database_access.image_metadata import (
    filter_image_ids,
    get_metadata_for_dataset,
    normalize_key,
)
from app.services.queue_methods import METHODS, MethodContext, QueueMethod

Ranked = tuple[int, str, float | None]
T = TypeVar("T")


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def get_method(sorts: str, key: str) -> QueueMethod:
    method = METHODS[sorts].get(key)
    if method is None:
        raise bad_request(f"Unknown sorting method '{key}'. Available: {sorted(METHODS[sorts])}.")
    return method


def chosen_methods(sorts: str, settings: SortSettings) -> list[QueueMethod]:
    """The settings' method and round-robin partner. Raises 400 on an unknown method or a
    direction the method does not support."""
    methods = [get_method(sorts, settings.method)]
    if settings.round_robin_with:
        methods.append(get_method(sorts, settings.round_robin_with))
    if settings.direction and settings.direction not in methods[0].directions:
        raise bad_request(f"'{methods[0].label}' cannot be sorted {settings.direction}.")
    return methods


def method_options(sorts: str) -> list[QueueMethodOption]:
    return [
        QueueMethodOption(key=m.key, label=m.label, needs=list(m.needs),
                          directions=list(m.directions), descriptions=m.descriptions)
        for m in METHODS[sorts].values()
    ]


# -- Filters ------------------------------------------------------------------------

def filter_by_metadata(db: Session, dataset_id: int, units: list[T],
                       filters: dict, image_ids_of: Callable[[T], Iterable[int]]) -> list[T]:
    """Keep units with at least one image matching the gallery-style metadata filter."""
    if not filters:
        return units
    allowed = set(filter_image_ids(db, dataset_id, filters))
    return [u for u in units if allowed.intersection(image_ids_of(u))]


def filter_by_confidence(units: list[T], max_confidence: float | None,
                         confidence_of: Callable[[T], float]) -> list[T]:
    """Keep units whose confidence is below the ceiling. What a unit's confidence is depends on
    the queue: an image's most confident detection, or an object's own score."""
    if max_confidence is None:
        return units
    return [u for u in units if confidence_of(u) < max_confidence]


def filter_by_author(units: list[T], settings: SortSettings, username: str | None,
                     authors_of: Callable[[T], set[str]]) -> list[T]:
    """Keep units by the people the settings' ``by`` picks; ``authors_of`` is who drew the
    unit (everyone who drew on an image, or an object's author)."""
    if settings.by == "everyone":
        return units
    if settings.by == "person":
        return [u for u in units if settings.by_person in authors_of(u)]
    mine = settings.by == "mine"
    return [u for u in units if (username in authors_of(u)) == mine]


def dataset_authors(db: Session, dataset_id: int, exclude: str | None = None) -> list[str]:
    """Everyone who has drawn an object in the dataset, for the builder's "By" list. Built
    from the work itself, so it needs no permission to list members and names only people
    the filter can match."""
    rows = (
        db.query(Contours.author_username)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id, Contours.temporary.is_(False),
                Contours.author_username.isnot(None))
        .distinct()
    )
    return sorted(author for (author,) in rows if author != exclude)


def recommended_queue(db: Session, dataset_id: int, kind: QueueKind) -> RecommendedQueue | None:
    """The dataset's recommended settings for one queue, for its summary (written by
    ``app.services.queue_recommendation``)."""
    stored = db.query(Datasets.recommended_queues).filter_by(id=dataset_id).scalar() or {}
    return RecommendedQueue.model_validate(stored[kind]) if kind in stored else None


# -- Sorting ------------------------------------------------------------------------

def _shown(score: Any) -> float:
    # Tuple scores carry tie-breakers; the first item is the one worth showing.
    return round(float(score[0] if isinstance(score, tuple) else score), 4)


def rank(method: QueueMethod, direction: str, ctx: MethodContext) -> list[Ranked]:
    """Units in ``method``'s order: scored units, then unscored ones in their given order.

    The sort is stable, so units with equal scores keep the order the queue passed them in.
    """
    scores = method.score(ctx)
    scored = sorted(scores, key=scores.get, reverse=direction == "descending")
    unscored = [u.key for u in ctx.units if u.key not in scores]
    return ([(key, method.key, _shown(scores[key])) for key in scored]
            + [(key, "no_data", None) for key in unscored])


def sort_units(methods: Sequence[QueueMethod], settings: SortSettings,
               ctx: MethodContext) -> list[Ranked]:
    """Sort with the chosen method, or alternate it with its round-robin partner. The partner
    always runs in its default direction: the one that serves its purpose."""
    primary, *partner = methods
    orders = [rank(primary, settings.direction or primary.directions[0], ctx)]
    orders += [rank(m, m.directions[0], ctx) for m in partner]
    return al_ranking.round_robin(*orders)


def balance(db: Session, dataset_id: int, ranked: list[Ranked], key: str,
            image_of: Mapping[int, int]) -> list[Ranked]:
    """Take turns between the values of a metadata key, keeping each value's own order.

    ``image_of`` maps a unit key to the image whose metadata counts for it (a scan's frames
    share the scan's metadata, so its first frame). Values take turns in the order they first
    appear in ``ranked``, so the best-ranked unit still comes first; units without the key are
    one more group.
    """
    wanted = normalize_key(key)
    metadata = get_metadata_for_dataset(db, dataset_id)
    groups: dict[str | None, list[Ranked]] = {}
    for entry in ranked:
        groups.setdefault(metadata.get(image_of[entry[0]], {}).get(wanted), []).append(entry)
    return al_ranking.round_robin(*groups.values())
