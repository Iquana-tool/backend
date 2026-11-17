"""Sorting methods for the annotation and review queues, as plugins.

A method is one function that scores the units it is given; a higher score comes first when
sorting descending. Each method sorts one kind of unit (``sorts``):

* ``"images"`` -- the annotation queue's units: plain images, or whole scans whose frames
  stay together (:class:`app.services.al_ranking.Unit`).
* ``"objects"`` -- the review queue's units: single contours
  (:class:`app.services.review_queue.QueueCandidate`).

Register a method with :func:`register_method` and it appears in that queue's builder, with
the pickers its ``needs`` call for. Everything around the score is shared and lives in
``app.services.queue_core``: sort direction, round robin between two methods, filters and
taking turns by a metadata key. What only one queue does stays in that queue's module
(``app.services.annotation_queue``, ``app.services.review_queue``), and preparing the data
image methods need lives in ``app.services.queue_prep``::

    @register_method("margin", "Margin", {
        "descending": "Images with the closest top-two scores first.",
        "ascending": "Images with the clearest top score first.",
    }, needs=("trained_model",))
    def _margin(ctx: MethodContext) -> dict[int, float]:
        return {unit.key: ... for unit in ctx.units}

Each supported direction comes with the sentence the builder shows for it; the first is the
default. A unit a method gives no score (missing data) is placed after the scored ones. Scores may be
tuples to break ties. A method that builds its order step by step returns its pick order as
the score (see Diversity).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Sequence

import numpy as np
from sqlalchemy.orm import Session

from app.database.image_scores import ImageScores
from app.database.image_vectors import ImageVectors
from app.services import al_ranking
from app.services.al_ranking import Unit

#: What a method can need prepared; mirrors app.schemas.annotation_queue.Need.
EMBEDDING_MODEL = "embedding_model"
TRAINED_MODEL = "trained_model"


def load_vectors(db: Session, image_ids: Sequence[int], model_key: str) -> dict[int, np.ndarray]:
    """Each image's vector from one embedding model."""
    rows = (db.query(ImageVectors.image_id, ImageVectors.vector)
            .filter(ImageVectors.model_key == model_key, ImageVectors.image_id.in_(image_ids)))
    return {image_id: np.asarray(vector, dtype=float) for image_id, vector in rows}


def load_confidences(db: Session, image_ids: Sequence[int], model_key: str) -> dict[int, list[float]]:
    """Each scored image's detection confidences from one trained model."""
    rows = (db.query(ImageScores.image_id, ImageScores.confidences)
            .filter(ImageScores.model_key == model_key, ImageScores.confidences.isnot(None),
                    ImageScores.image_id.in_(image_ids)))
    return {image_id: list(confidences) for image_id, confidences in rows}


@dataclass
class MethodContext:
    """What a method may read. Data is loaded on first access, so methods pay only for what
    they use."""

    db: Session
    #: The units to rank, matching the queue's filters: unfinished image units for the
    #: annotation queue, objects awaiting review for the review queue. Each has a ``key``.
    units: list[Any]
    #: Finished image units: what is already covered (Diversity measures distance from them).
    seed_units: list[Unit] = field(default_factory=list)
    embedding_model: str | None = None
    trained_model: str | None = None

    def _image_ids(self) -> list[int]:
        return [i for unit in (*self.units, *self.seed_units) for i in unit.image_ids]

    @cached_property
    def vectors(self) -> dict[int, np.ndarray]:
        """Unit key -> unit vector (mean of its frames), for units and seeds that have one."""
        per_image = load_vectors(self.db, self._image_ids(), self.embedding_model)
        vectors = {unit.key: al_ranking.unit_vector(unit, per_image)
                   for unit in (*self.units, *self.seed_units)}
        return {key: vector for key, vector in vectors.items() if vector is not None}

    @cached_property
    def confidences(self) -> dict[int, list[float]]:
        """Image id -> the trained model's detection confidences."""
        return load_confidences(self.db, self._image_ids(), self.trained_model)


ScoreFn = Callable[[MethodContext], dict[int, Any]]


IMAGES = "images"
OBJECTS = "objects"


@dataclass(frozen=True)
class QueueMethod:
    key: str
    label: str
    #: Supported direction -> what that order puts first. The first direction is the default.
    descriptions: dict[str, str]
    score: ScoreFn
    needs: tuple[str, ...] = ()
    sorts: str = IMAGES

    @property
    def directions(self) -> tuple[str, ...]:
        return tuple(self.descriptions)


#: What a method sorts -> key -> method. Keys are unique per kind only, so both queues can
#: offer an "Uncertainty" that means the same idea for their own units.
METHODS: dict[str, dict[str, QueueMethod]] = {IMAGES: {}, OBJECTS: {}}


def register_method(key: str, label: str, descriptions: dict[str, str], *,
                    sorts: str = IMAGES, needs: tuple[str, ...] = ()):
    """Register a sorting method; registered methods appear in that queue's builder."""

    def decorator(fn: ScoreFn) -> ScoreFn:
        METHODS[sorts][key] = QueueMethod(key, label, dict(descriptions), fn, tuple(needs), sorts)
        return fn

    return decorator


# -- Images (annotation queue) ------------------------------------------------------


@register_method("as_uploaded", "As uploaded", {
    "ascending": "The first images added to the dataset come first.",
    "descending": "The most recently added images come first.",
})
def _as_uploaded(ctx: MethodContext) -> dict[int, int]:
    return {unit.key: position for position, unit in enumerate(ctx.units)}


@register_method("random", "Random", {
    "ascending": "Shuffled once. The order stays fixed until you rebuild it.",
})
def _random(ctx: MethodContext) -> dict[int, float]:
    return {unit.key: random.random() for unit in ctx.units}


@register_method("diversity", "Diversity", {
    "descending": "Images least like anything already annotated come first.",
    "ascending": "Images most like what is already annotated come first.",
}, needs=(EMBEDDING_MODEL,))
def _diversity(ctx: MethodContext) -> dict[int, int]:
    candidates = {u.key: ctx.vectors[u.key] for u in ctx.units if u.key in ctx.vectors}
    seeds = [ctx.vectors[u.key] for u in ctx.seed_units if u.key in ctx.vectors]
    picks = al_ranking.k_center_order(candidates, seeds)
    return {key: len(picks) - position for position, key in enumerate(picks)}


@register_method("density", "Density", {
    "descending": "Typical images, ones that look like many others, come first.",
    "ascending": "Unusual images, ones that look like few others, come first.",
}, needs=(EMBEDDING_MODEL,))
def _density(ctx: MethodContext) -> dict[int, float]:
    return al_ranking.density_scores(
        {u.key: ctx.vectors[u.key] for u in ctx.units if u.key in ctx.vectors})


@register_method("uncertainty", "Uncertainty", {
    "descending": "Images where the chosen model is most torn about an object come first. "
                  "These teach it the most.",
    "ascending": "Images where the chosen model is most sure come first. "
                 "Useful to check what it already gets right.",
}, needs=(TRAINED_MODEL,))
def _uncertainty(ctx: MethodContext) -> dict[int, tuple[float, int]]:
    scores = {u.key: al_ranking.unit_uncertainty(u, ctx.confidences) for u in ctx.units}
    return {key: score for key, score in scores.items() if score is not None}


# -- Objects (review queue) ---------------------------------------------------------
#
# Objects arrive sorted by image, then nesting depth, then id, and sorting is stable, so
# objects with the same score stay grouped by image with parents before their children.

@register_method("hierarchy", "Hierarchy", {
    "ascending": "Whole objects come first, then the parts inside them. "
                 "Check containers before their contents.",
    "descending": "The innermost parts come first, the objects that contain them last.",
}, sorts=OBJECTS)
def _hierarchy(ctx: MethodContext) -> dict[int, int]:
    return {obj.key: obj.depth for obj in ctx.units}


@register_method("confidence", "Confidence", {
    "ascending": "The least confident objects come first, to fix likely mistakes. "
                 "Hand-drawn objects count as fully confident.",
    "descending": "The most confident objects come first, to accept good predictions quickly.",
}, sorts=OBJECTS)
def _object_confidence(ctx: MethodContext) -> dict[int, float]:
    return {obj.key: float(obj.confidence) for obj in ctx.units}


@register_method("uncertainty", "Uncertainty", {
    "descending": "Objects the model was most torn about (closest to 50/50) come first, "
                  "where mistakes are most likely. Hand-drawn objects come last.",
    "ascending": "Objects the model was most sure about come first.",
}, sorts=OBJECTS)
def _object_uncertainty(ctx: MethodContext) -> dict[int, float]:
    return {obj.key: al_ranking.binary_entropy(obj.confidence) for obj in ctx.units}
