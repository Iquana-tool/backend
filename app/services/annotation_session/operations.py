"""Transport-agnostic AI segmentation operations.

Each ``run_*`` function takes plain primitives plus an AI service instance, performs the
inference, parses the result into domain objects, and returns it alongside the service's
``success`` / ``message``. They never touch a WebSocket, a ``ClientMessage`` or the
database, so they are reusable from HTTP routes, Celery tasks or tests.

Alongside the ``run_*`` functions are pure post-processing helpers (candidate selection,
exemplar-overlap filtering, hierarchy placement). These also avoid the database -- the
caller passes in already-fetched contours / hierarchies -- so they stay reusable.

Persistence (adding/replacing contours on a mask) and session bookkeeping are the
caller's responsibility -- see ``app.routes.websockets.annotation_handlers`` for the
WebSocket adapter.
"""

from dataclasses import dataclass
from logging import getLogger

import numpy as np
from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.labels import LabelHierarchy
from iquana_toolbox.schemas.prompts import Prompts
from iquana_toolbox.schemas.networking.http.services import (
    InstanceSuggestionRequest,
    InstanceSegmentationRequest,
    PromptedSegmentationRequest,
    SemanticSegmentationRequest,
)

from app.services.ai_services.base_service import BaseService

logger = getLogger(__name__)

# IoU at/above which two contours are considered the "same" object. Used both to discard
# a prompted candidate that merely re-segments the focussed parent and to drop discovered
# instances that just reproduce one of their own exemplars.
DUPLICATE_IOU_THRESHOLD = 0.9

# Fraction of a discovered contour's area that must lie inside a candidate parent contour
# for it to be nested under that parent.
CONTAINMENT_THRESHOLD = 0.5

# Resolution at which contours are rasterised for IoU / containment comparisons. Contour
# coordinates are normalised to [0, 1], so a fixed square grid is consistent across images.
_COMPARE_RESOLUTION = (1000, 1000)


def _iou(a: Contour, b: Contour, resolution: tuple[int, int] = _COMPARE_RESOLUTION) -> float:
    """Intersection-over-union of two contours, rasterised at a fixed resolution."""
    mask_a = a.to_binary_mask(*resolution)
    mask_b = b.to_binary_mask(*resolution)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)


def _containment(inner: Contour, outer: Contour,
                 resolution: tuple[int, int] = _COMPARE_RESOLUTION) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer`` (1.0 == fully contained)."""
    inner_mask = inner.to_binary_mask(*resolution)
    inner_area = inner_mask.sum()
    if inner_area == 0:
        return 0.0
    outer_mask = outer.to_binary_mask(*resolution)
    return float(np.logical_and(inner_mask, outer_mask).sum() / inner_area)


@dataclass
class PromptedSegmentationResult:
    # ``contour`` is None when every candidate was discarded as a duplicate of the
    # focussed parent -- the caller should then add nothing.
    contour: Contour | None
    candidates: list[Contour]
    success: bool
    message: str


@dataclass
class SemanticSegmentationResult:
    hierarchy: ContourHierarchy
    success: bool
    message: str


@dataclass
class InstanceSegmentationResult:
    contours: list[Contour]
    success: bool
    message: str


@dataclass
class SuggestionResult:
    contours: list[Contour]
    success: bool
    message: str


def select_best_prompted_contour(
        candidates: list[Contour],
        focus_contour: Contour | None = None,
) -> Contour | None:
    """Pick the best prompted-segmentation candidate.

    Candidates that overlap the focussed contour by more than ``DUPLICATE_IOU_THRESHOLD``
    IoU are discarded (they just re-segment the parent we are annotating inside), then the
    highest-confidence survivor is returned. Returns None if nothing survives.
    """
    if focus_contour is not None:
        candidates = [c for c in candidates if _iou(c, focus_contour) <= DUPLICATE_IOU_THRESHOLD]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.confidence)


def filter_exemplar_overlaps(
        found: list[Contour],
        exemplars: list[Contour],
        threshold: float = DUPLICATE_IOU_THRESHOLD,
) -> list[Contour]:
    """Drop discovered contours that merely reproduce one of the seed exemplars.

    A discovered contour is removed when its IoU with any exemplar is >= ``threshold``.
    """
    kept = []
    for contour in found:
        if any(_iou(contour, exemplar) >= threshold for exemplar in exemplars):
            continue
        kept.append(contour)
    return kept


def assign_hierarchy_parents(
        found: list[Contour],
        hierarchy: ContourHierarchy,
        label_hierarchy: LabelHierarchy,
        concept_label_id: int | None,
        threshold: float = CONTAINMENT_THRESHOLD,
) -> list[Contour]:
    """Label discovered contours with the concept and nest them under the right parent.

    Each found contour is tagged with ``concept_label_id``. If that label has a parent
    label in ``label_hierarchy``, the contour is nested under the existing contour of that
    parent label which geometrically contains it (highest containment above ``threshold``).
    Contours with no valid containing parent stay at root level. Mutates and returns
    ``found``.
    """
    parent_label = None
    if concept_label_id is not None and concept_label_id in label_hierarchy.id_to_label_object:
        parent_label = label_hierarchy.get_parent_by_id_of_child(concept_label_id)

    for contour in found:
        contour.label_id = concept_label_id
        contour.parent_id = None

        if parent_label is None:
            # Concept is a root label (or unknown) -> nothing to nest under.
            continue

        # Only existing contours carrying the correct parent label are eligible parents.
        candidate_parents = hierarchy.label_id_to_contours.get(parent_label.id, [])
        best_parent = None
        best_ratio = threshold
        for candidate in candidate_parents:
            ratio = _containment(contour, candidate)
            if ratio > best_ratio:
                best_parent = candidate
                best_ratio = ratio
        if best_parent is not None:
            contour.parent_id = best_parent.id
        else:
            logger.debug("Discovered contour has no correct-label parent containing it; "
                         "adding at root level.")
    return found


async def run_prompted_segmentation(
        *,
        service: BaseService,
        image_url: str,
        image_width: int,
        image_height: int,
        model_key: str,
        prompts: Prompts,
        user_id: str,
        previous_mask=None,
        parent_id: int | None = None,
        focus_contour: Contour | None = None,
) -> PromptedSegmentationResult:
    """Run prompted segmentation and select the best resulting contour.

    The service returns a list of candidate contours; the best one (after discarding
    duplicates of ``focus_contour``) gets its ``parent_id`` set and its SVG path computed.
    The add-vs-replace / refinement decision and any persistence are left to the caller.
    """
    request = PromptedSegmentationRequest(
        user_id=str(user_id),
        image_url=image_url,
        model_registry_key=model_key,
        previous_mask=previous_mask,
        prompts=prompts,
    )
    response = await service.inference(request)

    # The service may return a list of candidates or (for backwards compatibility) a single
    # contour / None.
    result_data = response["result"]
    if result_data is None:
        items = []
    elif isinstance(result_data, list):
        items = result_data
    else:
        items = [result_data]
    candidates = [Contour.model_validate(item) for item in items]

    best = select_best_prompted_contour(candidates, focus_contour)
    if best is not None:
        best.parent_id = parent_id
        best.compute_path(image_width=image_width, image_height=image_height)

    return PromptedSegmentationResult(
        contour=best,
        candidates=candidates,
        success=response["success"],
        message=response["message"],
    )


async def run_semantic_segmentation(
        *,
        service: BaseService,
        image_url: str,
        model_registry_key: str,
        user_id: str,
) -> SemanticSegmentationResult:
    """Run semantic segmentation and parse the resulting contour hierarchy."""
    request = SemanticSegmentationRequest(
        model_registry_key=model_registry_key,
        image_url=image_url,
        user_id=user_id,
    )
    response = await service.inference(request)
    hierarchy = ContourHierarchy.model_validate(response["result"])
    return SemanticSegmentationResult(
        hierarchy=hierarchy,
        success=response["success"],
        message=response["message"],
    )


async def run_instance_segmentation(
        *,
        service: BaseService,
        image_url: str,
        image_width: int,
        image_height: int,
        model_registry_key: str,
        user_id: str,
) -> InstanceSegmentationResult:
    """Run instance segmentation and parse the detected instance contours.

    The (multiclass) model returns every detected instance as a flat list of contours.
    Each contour's SVG path is computed for the frontend. Persistence -- including
    whether to replace the existing contours -- is the caller's responsibility.
    """
    request = InstanceSegmentationRequest(
        model_registry_key=model_registry_key,
        image_url=image_url,
        user_id=user_id,
    )
    response = await service.inference(request)
    contours = [Contour.model_validate(item) for item in (response["result"] or [])]
    for contour in contours:
        contour.compute_path(image_width=image_width, image_height=image_height)
    return InstanceSegmentationResult(
        contours=contours,
        success=response["success"],
        message=response["message"],
    )


async def run_suggestion_segmentation(
        *,
        service: BaseService,
        image_url: str,
        model_key: str,
        user_id: str,
        positive_exemplars: list,
        concept=None,
        negative_exemplars: list | None = None,
) -> SuggestionResult:
    """Run annotation suggestion and parse the discovered contours.

    Returns the raw discovered contours. Exemplar-overlap filtering
    (``filter_exemplar_overlaps``) and hierarchy placement (``assign_hierarchy_parents``)
    are applied by the caller, which has access to the seed contours and the mask /
    label hierarchies.
    """
    request = InstanceSuggestionRequest(
        image_url=image_url,
        model_registry_key=model_key,
        user_id=user_id,
        positive_exemplars=positive_exemplars,
        negative_exemplars=negative_exemplars,
        concept=concept,
    )
    response = await service.inference(request)
    contours = [Contour.model_validate(contour_json) for contour_json in (response["result"] or [])]
    return SuggestionResult(
        contours=contours,
        success=response["success"],
        message=response["message"],
    )
