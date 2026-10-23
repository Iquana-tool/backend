"""Building and persisting annotation queues: the order an annotator works images.

Mirrors ``app.services.review_queue`` but with two deliberate differences:

* **Whole-image ordering, not instance scoring.** The queue orders the dataset's images; a
  scan stack counts as one unit whose frames stay together.
* **Persisted, not a snapshot.** The built order is saved per (dataset, user) in
  ``annotation_queues`` so re-entering resumes it; rebuilding overwrites the row.

How a queue is built (:func:`build_order`):

1. Filter units by image metadata and, with a trained model, by its most confident detection.
2. Set finished units, and units someone else has started, aside; they go last.
3. Score the rest with the chosen method (``app.services.queue_methods``) and sort them in
   the chosen direction -- or alternate two methods (round robin). Units a method cannot
   score (missing data) follow its scored ones.

The data methods need is filled beforehand by ``app.services.queue_prep``; building never
waits for it.
"""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.annotation_queues import AnnotationQueues
from app.database.contours import Contours
from app.database.images import Images
from app.database.masks import Masks
from app.schemas.annotation_queue import (
    AnnotationQueueMethodOption,
    AnnotationQueueRead,
    AnnotationQueueSummary,
    QueueModelOption,
    QueueSettings,
)
from app.services import al_ranking
from app.services.database_access.image_metadata import filter_image_ids
from app.services.queue_methods import METHODS, TRAINED_MODEL, MethodContext, QueueMethod, load_confidences


_NEED_NAMES = {"embedding_model": "an embedding model", "trained_model": "a trained model"}


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def get_method(key: str) -> QueueMethod:
    method = METHODS.get(key)
    if method is None:
        raise _bad_request(f"Unknown sorting method '{key}'. Available: {sorted(METHODS)}.")
    return method


def required_models(settings: QueueSettings) -> dict[str, str]:
    """{need: model key} for everything the settings' methods and filters need.

    Raises 400 when a needed model was not chosen or a direction is not supported.
    """
    methods = [get_method(settings.method)]
    if settings.round_robin_with:
        methods.append(get_method(settings.round_robin_with))
    if settings.direction and settings.direction not in methods[0].directions:
        raise _bad_request(f"'{methods[0].label}' cannot be sorted {settings.direction}.")
    needs = {need for method in methods for need in method.needs}
    if settings.max_confidence is not None:
        needs.add(TRAINED_MODEL)
    chosen = {need: getattr(settings, need) for need in sorted(needs)}
    missing = [_NEED_NAMES[need] for need, model in chosen.items() if not model]
    if missing:
        raise _bad_request(f"Choose {' and '.join(missing)} first.")
    return chosen


def method_options() -> list[AnnotationQueueMethodOption]:
    return [
        AnnotationQueueMethodOption(key=m.key, label=m.label, needs=list(m.needs),
                                    directions=list(m.directions), descriptions=m.descriptions)
        for m in METHODS.values()
    ]


def trained_models_for_dataset(models: list[dict], dataset_id: int) -> list[QueueModelOption]:
    """Instance-segmentation models trained on this dataset (``list_available_models`` rows).

    Training stamps ``trained_on_dataset_id`` (and the older ``dataset_id``) on a model's
    tags; a base model carries neither and cannot judge a dataset it has never seen.
    """
    wanted = str(dataset_id)
    return [
        QueueModelOption(registry_key=m["registry_key"], name=m.get("name") or m["registry_key"])
        for m in models
        if str((m.get("tags") or {}).get("trained_on_dataset_id")
               or (m.get("tags") or {}).get("dataset_id") or "") == wanted
    ]


# -- Building ---------------------------------------------------------------------

def image_states(db: Session, dataset_id: int) -> dict[int, str]:
    """``annotate_status`` per image that has a mask; images without one are not started.

    Computed in a subquery for the reason given in ``_status_counts``.
    """
    status_sq = (
        db.query(Masks.image_id.label("image_id"), Masks.annotate_status.label("status"))
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .subquery()
    )
    return dict(db.query(status_sq.c.image_id, status_sq.c.status).all())


def _image_authors(db: Session, dataset_id: int) -> dict[int, set[str]]:
    """Who has drawn on each image (non-temporary objects with a known author)."""
    rows = (
        db.query(Masks.image_id, Contours.author_username)
        .join(Contours, Contours.mask_id == Masks.id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id, Contours.temporary.is_(False),
                Contours.author_username.isnot(None))
        .distinct()
    )
    authors: dict[int, set[str]] = {}
    for image_id, author in rows:
        authors.setdefault(image_id, set()).add(author)
    return authors


def _unit_state(unit: al_ranking.Unit, states: dict[int, str],
                authors: dict[int, set[str]], username: str) -> str:
    """"finished", "others" (someone else has started it) or "open"."""
    frame_states = [states.get(image_id, "not_started") for image_id in unit.image_ids]
    if all(state == "finished" for state in frame_states):
        return "finished"
    started = [i for i, s in zip(unit.image_ids, frame_states) if s == "in_progress"]
    unit_authors = set().union(*(authors.get(i, set()) for i in started))
    return "others" if unit_authors and username not in unit_authors else "open"


def _uploaded_units(db: Session, dataset_id: int) -> list[al_ranking.Unit]:
    rows = (db.query(Images.id, Images.stack_id, Images.frame_index)
            .filter(Images.dataset_id == dataset_id).order_by(Images.id.asc()))
    return al_ranking.build_units([tuple(row) for row in rows])


def _filter_units(db: Session, dataset_id: int, units: list[al_ranking.Unit],
                  settings: QueueSettings) -> list[al_ranking.Unit]:
    """Keep units matching the metadata filter and the confidence ceiling."""
    if settings.metadata_filters:
        allowed = set(filter_image_ids(db, dataset_id, settings.metadata_filters))
        units = [u for u in units if allowed.intersection(u.image_ids)]
    if settings.max_confidence is not None:
        confidences = load_confidences(db, [i for u in units for i in u.image_ids],
                                       settings.trained_model)
        units = [u for u in units
                 if max((c for i in u.image_ids for c in confidences.get(i, [])), default=0.0)
                 < settings.max_confidence]
    return units


def _rank(method: QueueMethod, direction: str,
          ctx: MethodContext) -> list[tuple[int, str, float | None]]:
    """(unit key, reason, score) in queue order: scored units, then unscored ones."""
    scores = method.score(ctx)
    scored = sorted(scores, key=scores.get, reverse=direction == "descending")
    unscored = [u.key for u in ctx.units if u.key not in scores]

    def shown(score) -> float:
        return round(float(score[0] if isinstance(score, tuple) else score), 4)

    return ([(key, method.key, shown(scores[key])) for key in scored]
            + [(key, "no_data", None) for key in unscored])


def build_order(db: Session, dataset_id: int, username: str,
                settings: QueueSettings) -> tuple[list[int], dict[str, dict]]:
    """The queue's image ids and why each sits where it does."""
    models = required_models(settings)
    units = _filter_units(db, dataset_id, _uploaded_units(db, dataset_id), settings)
    states = image_states(db, dataset_id)
    authors = _image_authors(db, dataset_id)
    by_state: dict[str, list[al_ranking.Unit]] = {"open": [], "others": [], "finished": []}
    for unit in units:
        by_state[_unit_state(unit, states, authors, username)].append(unit)

    ctx = MethodContext(db, by_state["open"], by_state["finished"],
                        embedding_model=models.get("embedding_model"),
                        trained_model=models.get("trained_model"))
    primary = get_method(settings.method)
    orders = [_rank(primary, settings.direction or primary.directions[0], ctx)]
    if settings.round_robin_with:
        secondary = get_method(settings.round_robin_with)
        orders.append(_rank(secondary, secondary.directions[0], ctx))
    ranked = al_ranking.round_robin(*orders)
    ranked += [(u.key, "started_by_other", None) for u in by_state["others"]]
    ranked += [(u.key, "finished", None) for u in by_state["finished"]]

    by_key = {unit.key: unit for unit in units}
    image_ids: list[int] = []
    details: dict[str, dict] = {}
    for key, reason, score in ranked:
        for image_id in by_key[key].image_ids:
            image_ids.append(image_id)
            details[str(image_id)] = {"reason": reason, "score": score}
    return image_ids, details


# -- Persistence --------------------------------------------------------------------

def _status_counts(dataset_id: int, db: Session) -> dict[str, int]:
    """Per-state image counts for the dataset's Annotate phase.

    This card is about annotating, so it asks ``Masks.annotate_status`` rather than
    the image's combined status: an image whose annotation is done but which was
    never calibrated should not show up here as unfinished work for the annotator.

    ``annotate_status`` is a CASE expression built from correlated subqueries over
    ``masks.id``. Grouping by it directly makes SQLAlchemy render that CASE twice --
    once in SELECT, once in GROUP BY -- with *different* bound parameters each time, so
    PostgreSQL no longer recognises the two as the same expression and rejects
    ``masks.id`` with "subquery uses ungrouped column" (SQLite tolerated the mismatch).
    Computing the status once in a subquery and grouping by the resulting plain column
    avoids the double render.
    """
    status_sq = (
        db.query(Masks.annotate_status.label("status"))
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .subquery()
    )
    rows = (
        db.query(status_sq.c.status, func.count())
        .group_by(status_sq.c.status)
        .all()
    )
    return {mask_status: count for mask_status, count in rows}


def _saved_row(dataset_id: int, username: str, db: Session) -> AnnotationQueues | None:
    return (db.query(AnnotationQueues)
            .filter(AnnotationQueues.dataset_id == dataset_id,
                    AnnotationQueues.username == username)
            .one_or_none())


def _settings_of(row: AnnotationQueues) -> QueueSettings:
    # Rows from before settings were stored only know their method.
    return QueueSettings.model_validate(
        {**(row.options or {"method": row.strategy}), "show_builder": row.show_builder})


def get_saved_queue(dataset_id: int, username: str, db: Session) -> AnnotationQueueRead | None:
    """The caller's saved queue for the dataset, or None if they have not built one."""
    row = _saved_row(dataset_id, username, db)
    if row is None:
        return None
    image_ids = list(row.image_order or [])
    return AnnotationQueueRead(settings=_settings_of(row), image_ids=image_ids,
                               total=len(image_ids), details=dict(row.details or {}),
                               updated_at=row.updated_at)


def build_and_save_queue(dataset_id: int, username: str, settings: QueueSettings,
                         db: Session) -> AnnotationQueueRead:
    """Build the queue for the settings and persist it (upsert)."""
    image_ids, details = build_order(db, dataset_id, username, settings)
    row = _saved_row(dataset_id, username, db)
    if row is None:
        row = AnnotationQueues(dataset_id=dataset_id, username=username)
        db.add(row)
    row.strategy = settings.method
    row.options = settings.model_dump(exclude={"show_builder"})
    row.show_builder = settings.show_builder
    row.image_order = image_ids
    row.details = details or None
    db.commit()
    return get_saved_queue(dataset_id, username, db)


def summarize(dataset_id: int, username: str, db: Session) -> AnnotationQueueSummary:
    """Counts behind the Annotation card's subcaption plus the saved-queue state."""
    counts = _status_counts(dataset_id, db)
    total = db.query(func.count(Images.id)).filter(Images.dataset_id == dataset_id).scalar() or 0
    row = _saved_row(dataset_id, username, db)
    in_progress = counts.get("in_progress", 0)
    finished = counts.get("finished", 0)
    return AnnotationQueueSummary(
        # Derived rather than read off the tally: an image that has never been
        # opened has no mask row, so it contributes to no bucket at all. It is
        # not started, and the three counts have to add up to the total.
        not_started=max(total - in_progress - finished, 0),
        in_progress=in_progress,
        finished=finished,
        total=total,
        has_saved_queue=row is not None,
        saved_settings=_settings_of(row) if row else None,
        methods=method_options(),
    )
