"""Ranking maths for the active-learning annotation queue.

Pure functions over ids, scores and vectors, so they can be tested without a database.
A *unit* is what the queue ranks: a plain image, or a whole scan stack whose frames stay
together. Vectors are DINOv3 ``image_cls`` embeddings; distances are cosine distances.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: (image id, stack id or None, frame index or None), in upload order.
ImageRef = tuple[int, int | None, int | None]

#: Density: how many most-similar neighbours define how "typical" a unit is.
DENSITY_NEIGHBOURS = 30
#: A detection counts as borderline when its confidence lies in this range.
BORDERLINE_RANGE = (0.3, 0.7)


@dataclass(frozen=True)
class Unit:
    #: The unit's first image id (first frame, for a stack); used as the unit's id.
    key: int
    #: Every image in the unit, in the order they appear in the queue.
    image_ids: tuple[int, ...]


def build_units(images: Sequence[ImageRef]) -> list[Unit]:
    """Group images into units, keeping upload order; a stack sits where its first image did."""
    frames_by_stack: dict[int, list[tuple[int, int]]] = defaultdict(list)
    first_seen: list[tuple[str, int]] = []
    for image_id, stack_id, frame_index in images:
        if stack_id is None:
            first_seen.append(("image", image_id))
            continue
        if stack_id not in frames_by_stack:
            first_seen.append(("stack", stack_id))
        frames_by_stack[stack_id].append((frame_index if frame_index is not None else 0, image_id))

    units: list[Unit] = []
    for kind, ident in first_seen:
        if kind == "image":
            units.append(Unit(key=ident, image_ids=(ident,)))
        else:
            ordered = tuple(image_id for _, image_id in sorted(frames_by_stack[ident]))
            units.append(Unit(key=ordered[0], image_ids=ordered))
    return units


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def image_uncertainty(confidences: Sequence[float]) -> float:
    """An image is as uncertain as its most borderline detection; no detections -> 0."""
    return max((binary_entropy(c) for c in confidences), default=0.0)


def unit_uncertainty(unit: Unit,
                     confidences: Mapping[int, Sequence[float]]) -> tuple[float, int] | None:
    """(most borderline detection over all frames, borderline detections in total).

    The second value breaks ties: of two equally torn images, the one with more
    borderline objects comes first. None when no frame has been scored.
    """
    scored = [confidences[i] for i in unit.image_ids if i in confidences]
    if not scored:
        return None
    return (max(image_uncertainty(c) for c in scored),
            sum(borderline_count(c) for c in scored))


def unit_vector(unit: Unit, vectors: Mapping[int, np.ndarray]) -> np.ndarray | None:
    """The normalised mean of the unit's frame vectors, or None when it has none."""
    present = [np.asarray(vectors[i], dtype=float) for i in unit.image_ids if i in vectors]
    if not present:
        return None
    return _normalise(np.mean(present, axis=0))


def k_center_order(candidates: Mapping[int, np.ndarray], seeds: Sequence[np.ndarray]) -> list[int]:
    """Order candidates so each next one is as far as possible from everything chosen so far.

    "Chosen so far" starts with the seeds (already annotated units). Without seeds the first
    pick is the candidate farthest from the candidates' mean, i.e. the most unusual one.
    """
    keys = list(candidates)
    if not keys:
        return []
    matrix = np.stack([_normalise(np.asarray(candidates[k], dtype=float)) for k in keys])
    remaining = np.ones(len(keys), dtype=bool)
    order: list[int] = []

    if seeds:
        seed_matrix = np.stack([_normalise(np.asarray(s, dtype=float)) for s in seeds])
        min_dist = (1.0 - matrix @ seed_matrix.T).min(axis=1)
    else:
        centre = _normalise(matrix.mean(axis=0))
        first = int(np.argmax(1.0 - matrix @ centre))
        order.append(keys[first])
        remaining[first] = False
        min_dist = 1.0 - matrix @ matrix[first]

    while remaining.any():
        index = int(np.argmax(np.where(remaining, min_dist, -np.inf)))
        order.append(keys[index])
        remaining[index] = False
        min_dist = np.minimum(min_dist, 1.0 - matrix @ matrix[index])
    return order


def binary_entropy(confidence: float) -> float:
    """How torn the model is about one prediction, from 0 (sure either way) to 1 (50/50).

    Binary entropy of the confidence, divided by its maximum ln 2 so the value reads as
    a share. Used for both images (annotation queue) and single objects (review queue).
    """
    c = min(max(float(confidence), 1e-12), 1.0 - 1e-12)
    return -(c * math.log(c) + (1.0 - c) * math.log(1.0 - c)) / math.log(2.0)


def borderline_count(confidences: Sequence[float]) -> int:
    low, high = BORDERLINE_RANGE
    return sum(1 for c in confidences if low <= c <= high)


def density_scores(vectors: Mapping[int, np.ndarray],
                   neighbours: int = DENSITY_NEIGHBOURS) -> dict[int, float]:
    """How typical each unit is: mean cosine similarity to its most similar other units.

    Similarity rather than YAPAT's 1 / distance: it stays between -1 and 1, so a few
    near-duplicate photos cannot produce extreme values that swamp everyone else.
    With fewer units than ``neighbours + 1``, every other unit is a neighbour.
    """
    keys = list(vectors)
    if len(keys) < 2:
        return {key: 0.0 for key in keys}
    matrix = np.stack([_normalise(np.asarray(vectors[k], dtype=float)) for k in keys])
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)  # a unit is not its own neighbour
    k = min(neighbours, len(keys) - 1)
    top = -np.sort(-similarity, axis=1)[:, :k]
    return {key: float(value) for key, value in zip(keys, top.mean(axis=1))}


def round_robin(*orders: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Interleave orders, one entry from each in turn, skipping units already taken.

    Entries are tuples whose first item is the unit key (the rest is carried along).
    """
    taken: set[int] = set()
    merged: list[tuple[int, ...]] = []
    iterators = [iter(order) for order in orders]
    while iterators:
        for iterator in list(iterators):
            for entry in iterator:
                if entry[0] not in taken:
                    taken.add(entry[0])
                    merged.append(entry)
                    break
            else:
                iterators.remove(iterator)
    return merged
