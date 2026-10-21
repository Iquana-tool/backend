"""Running one unit of a plan: one model, one image, written into the dataset.

The pipeline for a unit, in order:

1. **Predict.** One HTTP call to the AI service for the step's task. The service is the only
   thing that touches a GPU; this worker is an HTTP client that writes rows.
2. **Filter to the step's label** (:func:`filter_for_step`). A multiclass model returns every
   class it knows; a step is responsible for exactly one label, so everything else is
   dropped. A class-agnostic model returns unlabelled objects, which are stamped with the
   step's label instead. This is what lets one plan mix "one specialist per label" with "one
   multiclass model bound to three labels" -- both look the same by the time they are merged.
3. **Nest under a parent** (:func:`attach_parents`), for steps below the root level. The
   parent is the already-written instance of the parent label that contains the prediction
   best; predictions that land inside nothing follow the run's `unparented` policy.
4. **Drop duplicates** (NMS). Candidates compete with each other and -- in patch mode -- with
   the contours that are already on the image. Existing contours always win: a patching run
   adds, it never removes.
5. **Write.** Ordinary contours, `added_by` naming the model, unreviewed, so they land in the
   review queue like any other annotation.

Everything here is synchronous: it runs inside a Celery worker, not inside the event loop.
The AI-service clients are async (they are shared with the interactive routes), so the two
calls that need them are bridged with `asyncio.run` -- once per unit, which is noise next to
a forward pass.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Optional

from iquana_toolbox.inference import best_parent, nms
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.labels import Label
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.contours import Contours, save_contour_tree
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.schemas.inference import InferenceOptions, ResolvedStep, UnparentedPolicy
from app.database.ai_suggestions import SuggestionSource
from app.services import model_label_mapping, provenance
from app.services.database_access.contours import invalidate_metrics_for_new_contours

logger = getLogger(__name__)


@dataclass
class UnitResult:
    """What one (step, image) unit did."""

    created: int = 0
    suppressed: int = 0
    unparented: int = 0
    contour_ids: list[int] = field(default_factory=list)


class InferenceUnitError(RuntimeError):
    """A unit failed. Carries a message short enough to render in the failed-items list."""


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict(db: Session, step: ResolvedStep, image: Images, username: str) -> list[Contour]:
    """Ask the AI service for this step's instances on one image.

    Both supported tasks return a list of contours in normalized coordinates. The difference
    is only in how the request is built: an instance-segmentation model runs on the image
    alone, while a cross-image model additionally needs exemplars retrieved from the
    embedding store (which the gateway's cross-image orchestration assembles).
    """
    if step.task == "cross-image-suggestion":
        return _predict_cross_image(db, step, image, username)
    return _predict_instance_segmentation(db, step, image, username)


def _predict_instance_segmentation(
    db: Session, step: ResolvedStep, image: Images, username: str
) -> list[Contour]:
    response = _instance_segmentation_response(db, step, image, username)
    contours, _, _ = _mapped_instance_response(db, step, image, response)
    return contours


def _instance_segmentation_response(
    db: Session, step: ResolvedStep, image: Images, username: str
) -> dict[str, Any]:
    from app.services.ai_services.instance_segmentation import InstanceSegmentationService
    from app.services.inference.conditioning_dispatcher import dispatch_conditioning

    cond_result = dispatch_conditioning(db, step, image, username)
    label_row = db.get(Labels, step.label_id)
    parameters = step.inputs.get("parameters", {})

    request = InstanceSegmentationRequest(
        image_url=str(image.file_path),
        user_id=username,
        model_registry_key=step.model_registry_key,
        # Multiclass models honour this and return only the asked-for class; models that
        # ignore it are filtered gateway-side anyway (see filter_for_step).
        label=Label.from_db(label_row) if label_row is not None else None,
        parameters=parameters,
        contour_ids=cond_result.get("contour_ids", []),
        positive_exemplars=cond_result.get("positive_exemplars", []),
        exemplars=cond_result.get("exemplars", []),
        embeddings=cond_result.get("vectors", {}),
    )
    return asyncio.run(InstanceSegmentationService().inference(request))


def _mapped_instance_response(
    db: Session, step: ResolvedStep, image: Images, response: dict[str, Any]
) -> tuple[list[Contour], dict[str, Any], dict[int, str]]:
    """Map provider output while retaining per-contour source instance identities."""
    raw_items = response.get("result", []) if isinstance(response, dict) else response
    contours = _as_contours(response)
    source_ids: dict[int, str] = {}
    if isinstance(raw_items, list):
        for raw, contour in zip(raw_items, contours):
            if isinstance(raw, dict) and raw.get("source_instance_id") is not None:
                source_ids[id(contour)] = str(raw["source_instance_id"])

    # A model with a fixed label space names its classes; read them as this dataset's
    # labels. Its objects may arrive nested (vessels in the choroid) -- flatten them, since
    # each level's step nests its own objects under the level above (see attach_parents).
    contours = model_label_mapping.apply_mapping(
        db, image.dataset_id, step.model_registry_key, contours
    )
    return (
        model_label_mapping.flatten(contours),
        response.get("prediction_metadata", {}) if isinstance(response, dict) else {},
        source_ids,
    )


def _preview_roi_regions(db: Session, step: ResolvedStep, image: Images) -> list[dict]:
    """Rasterize explicitly selected saved contours as native-size COCO RLEs."""
    if not step.roi_contour_ids:
        return []

    rows = (
        db.query(Contours)
        .join(Masks, Masks.id == Contours.mask_id)
        .filter(
            Contours.id.in_(step.roi_contour_ids),
            Contours.temporary.is_(False),
            Masks.image_id == image.id,
        )
        .all()
    )
    by_id = {row.id: row for row in rows}
    if len(by_id) != len(step.roi_contour_ids):
        raise InferenceUnitError("A selected ROI contour no longer belongs to the preview image.")

    regions: list[dict] = []
    for contour_id in step.roi_contour_ids:
        row = by_id[contour_id]
        regions.append(
            Contour(x=row.x or [], y=row.y or []).to_rle_encoding(image.height, image.width)
        )
    return regions


_CELLPOSE_POLYP_MODEL_KEY = "cpsam_polyp_v3_b_seed17"
_POLYP_ROI_MODEL_KEYS = {_CELLPOSE_POLYP_MODEL_KEY, "yolo_polyp_v3_c_seed1"}


def run_preview_unit(
    db: Session,
    step: ResolvedStep,
    image: Images,
    username: str,
    options: Optional[InferenceOptions] = None,
) -> dict[str, Any]:
    """Predict one image and return bounded-ready geometry without writing annotation rows."""
    parameters = dict((step.inputs or {}).get("parameters") or {})
    if step.roi_contour_ids:
        if step.model_registry_key in _POLYP_ROI_MODEL_KEYS:
            parameters["parent_regions"] = _preview_roi_regions(db, step, image)
        else:
            raise InferenceUnitError(
                f"Model '{step.model_registry_key}' does not support explicit ROI selections."
            )
    else:
        parameters.pop("parent_regions", None)

    preview_step = step.model_copy(update={
        "inputs": {**(step.inputs or {}), "parameters": parameters},
    })
    try:
        response = _instance_segmentation_response(db, preview_step, image, username)
        contours, metadata, source_ids = _mapped_instance_response(
            db, preview_step, image, response
        )
    except Exception as exc:
        raise InferenceUnitError(_prediction_error(exc, step)) from exc

    candidates = filter_for_step(
        contours, step, min_confidence=step.min_confidence or 0.0
    )

    parent_rows: list[Contours] = []
    if step.roi_contour_ids:
        parent_rows = (
            db.query(Contours)
            .join(Masks, Masks.id == Contours.mask_id)
            .filter(
                Contours.id.in_(step.roi_contour_ids),
                Contours.temporary.is_(False),
                Masks.image_id == image.id,
            )
            .all()
        )
    elif step.level > 0:
        mask = db.query(Masks).filter(Masks.image_id == image.id).first()
        if mask is not None:
            if step.parent_label_id is not None:
                parent_rows = _existing_contours(db, mask.id, label_id=step.parent_label_id)
            if not parent_rows:
                from app.services.inference.planning import _preview_roi_label_ids
                allowed_ancestors = _preview_roi_label_ids(db, image.dataset_id, step)
                if allowed_ancestors:
                    parent_rows = (
                        db.query(Contours)
                        .join(Masks, Masks.id == Contours.mask_id)
                        .filter(
                            Contours.label_id.in_(allowed_ancestors),
                            Contours.temporary.is_(False),
                            Masks.image_id == image.id,
                        )
                        .all()
                    )

    if parent_rows and candidates:
        if len(parent_rows) == 1 and step.roi_contour_ids:
            for candidate in candidates:
                candidate.parent_id = parent_rows[0].id
        else:
            containment = 0.05
            if options is not None and "min_parent_containment" in options.model_fields_set:
                containment = max(0.05, min(1.0, float(options.min_parent_containment)))
            candidates, _ = attach_parents(
                candidates,
                parent_rows,
                options=InferenceOptions(
                    min_parent_containment=containment,
                    unparented=UnparentedPolicy.KEEP_AT_ROOT,
                ),
            )

    payload = []
    for contour in candidates:
        item = contour.model_dump(mode="json")
        source_instance_id = source_ids.get(id(contour))
        if source_instance_id is not None:
            item["source_instance_id"] = source_instance_id
        payload.append(item)

    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    if isinstance(metadata.get("instances"), list):
        kept_ids = {source_ids[id(contour)] for contour in candidates if id(contour) in source_ids}
        metadata["instances"] = [
            instance for instance in metadata["instances"]
            if isinstance(instance, dict)
            and str(instance.get("source_instance_id")) in kept_ids
        ]
    return {"contours": payload, "prediction_metadata": metadata}


def _predict_cross_image(
    db: Session, step: ResolvedStep, image: Images, username: str
) -> list[Contour]:
    from app.services.ai_services.cross_image import CrossImageService
    from app.services.inference.conditioning_dispatcher import dispatch_conditioning
    from iquana_toolbox.schemas.networking.http.services import CrossImageSuggestionRequest

    cond_result = dispatch_conditioning(db, step, image, username)
    cond_kind = step.input_contract.conditioning.kind

    if cond_kind in ("reference_images", "instances"):
        exemplars = cond_result.get("exemplars", [])
        if not exemplars and step.input_contract.conditioning.min_units > 0:
            logger.info(
                "No exemplars for label %s on image %s; nothing to predict.",
                step.label_id,
                image.id,
            )
            return []
    else:
        exemplars = cond_result.get("exemplars", [])

    parameters = step.inputs.get("parameters", {})

    vectors_by_kind = cond_result.get("vectors_by_kind", {})
    embeddings = cond_result.get("vectors", {}) if not vectors_by_kind else {}

    request = CrossImageSuggestionRequest(
        image_url=str(image.file_path),
        user_id=username,
        model_registry_key=step.model_registry_key,
        exemplars=exemplars,
        concept=cond_result.get("concept"),
        parameters=parameters,
        contour_ids=cond_result.get("contour_ids", []),
        positive_exemplars=cond_result.get("positive_exemplars", []),
        embeddings=embeddings,
        exemplar_embeddings=vectors_by_kind,
    )
    logger.debug(
        "Image %s: %d exemplars retrieved for label %s (conditioning kind: %s).",
        image.id,
        len(exemplars),
        step.label_id,
        cond_kind,
    )
    return _as_contours(asyncio.run(CrossImageService().inference(request)))


def _as_contours(response) -> list[Contour]:
    """Coerce an AI-service response into contours.

    The services are inconsistent by design: the plain `/inference` route returns a bare
    list, the annotation-session routes wrap it in the `{success, message, result}` envelope
    the WebSocket gateway expects. Accept both rather than making the batch path depend on
    which one a given task happens to use.
    """
    payload = response.get("result", []) if isinstance(response, dict) else response
    if payload is None:
        return []
    return [item if isinstance(item, Contour) else Contour.model_validate(item) for item in payload]


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def filter_for_step(
    contours: list[Contour], step: ResolvedStep, *, min_confidence: float = 0.0
) -> list[Contour]:
    """Reduce a model's raw output to the objects this step is responsible for.

    Two regimes, decided by whether the model declares the classes it predicts:

    * **Multiclass** (``model_label_ids`` non-empty). The model returns everything it knows;
      keep only the step's label. This is what makes "point three labels at one multiclass
      model" behave identically to "point them at three specialists" -- each step sees only
      its own class, and the steps merge independently.
    * **Class-agnostic** (no declared classes -- a base model, or a concept segmenter driven
      by exemplars). There is nothing to filter by, so every returned object *is* an instance
      of the label the step was bound to, and gets stamped with it.
    """
    kept: list[Contour] = []
    for contour in contours:
        if contour.confidence < min_confidence:
            continue
        # A prediction naming its model class was labelled through the dataset's mapping,
        # so it is multiclass output even if this step's snapshot predates the mapping.
        if step.model_label_ids or contour.model_class is not None:
            if contour.label_id != step.label_id:
                continue
        else:
            contour.label_id = step.label_id
        contour.added_by = step.model_registry_key
        # Predictions arrive unreviewed and childless; a nested run fills children in later,
        # level by level, by writing them as their own contours. The identity fields are
        # cleared rather than trusted: a model has no idea what this dataset's contour ids
        # are, and a stale parent_id echoed back would nest the object under an unrelated
        # instance (or a row belonging to a different image entirely).
        contour.reviewed_by = []
        contour.children = []
        contour.id = None
        contour.parent_id = None
        kept.append(contour)
    return kept


# --------------------------------------------------------------------------- #
# Hierarchy
# --------------------------------------------------------------------------- #
def attach_parents(
    candidates: list[Contour],
    parents: list[Contours],
    options: InferenceOptions,
) -> tuple[list[Contour], int]:
    """Nest each candidate under the parent instance that contains it.

    ``parents`` are the contours of the step's *parent label* that are already on the image --
    which they are, because the work list runs level by level and this step's level did not
    start until the level above it finished the whole dataset.

    Returns the candidates that found a home (with ``parent_id`` set) plus the number that
    did not and were dropped. With ``unparented=keep_at_root`` nothing is dropped; the
    homeless predictions are written at root level for a human to sort out.
    """
    if not candidates:
        return [], 0
    if not parents:
        if options.unparented == UnparentedPolicy.KEEP_AT_ROOT:
            return candidates, 0
        return [], len(candidates)

    parent_shapes = [shape_of(parent) for parent in parents]
    attached: list[Contour] = []
    dropped = 0
    for candidate in candidates:
        index = best_parent(
            candidate, parent_shapes, min_containment=options.min_parent_containment
        )
        if index is None:
            if options.unparented == UnparentedPolicy.KEEP_AT_ROOT:
                candidate.parent_id = None
                attached.append(candidate)
            else:
                dropped += 1
            continue
        candidate.parent_id = parents[index].id
        attached.append(candidate)
    return attached, dropped


# --------------------------------------------------------------------------- #
# The unit
# --------------------------------------------------------------------------- #
def run_unit(
    db: Session,
    step: ResolvedStep,
    image: Images,
    options: InferenceOptions,
    username: str,
    mask: Optional[Masks] = None,
) -> UnitResult:
    """Predict, merge and write one step's output for one image. Does not commit."""
    if mask is None:
        mask = db.query(Masks).filter(Masks.image_id == image.id).first()
    if mask is None:
        raise InferenceUnitError(f"Image {image.id} has no mask to write to.")

    try:
        raw = predict(db, step, image, username)
    except Exception as exc:  # network, model load, bad response -- all report the same way
        raise InferenceUnitError(_prediction_error(exc, step)) from exc

    candidates = filter_for_step(raw, step, min_confidence=step.min_confidence or 0.0)
    if not candidates:
        return UnitResult()

    unparented = 0
    if step.level > 0:
        parents = _existing_contours(db, mask.id, label_id=step.parent_label_id)
        candidates, unparented = attach_parents(candidates, parents, options)
        if not candidates:
            return UnitResult(unparented=unparented)

    # Duplicate detection is per sibling group: two objects only compete when they could be
    # the same object, and objects under different parents cannot be. Within a group the
    # comparison deliberately ignores labels -- the tool's hierarchy already forbids two
    # same-level contours from overlapping (see ContourHierarchy.add_contour), so a predicted
    # cell landing on top of an existing piece of debris is a conflict, not two objects.
    result = UnitResult(unparented=unparented)
    saved_rows: list[Contours] = []
    for parent_id, group in _group_by_parent(candidates).items():
        existing = _existing_contours(db, mask.id, parent_id=parent_id, same_level=True)
        # In replace mode the image was already wiped, so `existing` is only what earlier
        # levels of this same run wrote -- which is exactly what should still suppress.
        decision = nms(
            group,
            iou_threshold=options.nms_iou,
            existing=[shape_of(contour) for contour in existing],
        )
        result.suppressed += len(decision.suppressed)
        for index in decision.kept:
            # Metric invalidation is deferred to one batched call below: it fans out over a
            # whole sibling group, and doing it per contour would rescan the group once per
            # instance -- quadratic on an image with hundreds of them.
            saved = save_contour_tree(
                db, group[index], mask.id, parent_id=parent_id,
                author_username=username, invalidate_metrics=False,
            )
            saved_rows.append(saved)
            result.contour_ids.append(saved.id)
            result.created += 1

    invalidate_metrics_for_new_contours(db, saved_rows)
    # Batch output lands unreviewed, so every object is a suggestion someone will accept,
    # fix or delete. The caller owns the transaction, hence commit=False.
    provenance.record_suggestions(
        db, [row.id for row in saved_rows],
        source=(SuggestionSource.CROSS_IMAGE if step.task == "cross-image-suggestion"
                else SuggestionSource.BATCH_INFERENCE),
        model_key=step.model_registry_key, username=username, commit=False,
    )
    # Classes mapped to a label *and* an attribute value (species) set that value too.
    model_label_mapping.apply_class_attributes(
        db, image.dataset_id, step.model_registry_key, [row.id for row in saved_rows])
    return result


def shape_of(contour: Contours) -> Contour:
    """A geometry-only :class:`Contour` view of a stored row.

    Deliberately narrower than ``Contour.from_db``: that also reads ``reviewed_by``, which
    lazy-loads the reviewer association one query per contour. Overlap tests need nothing but
    the outline, and a unit compares against every instance already on the image.
    """
    return Contour(
        id=contour.id,
        label_id=contour.label_id,
        parent_id=contour.parent_id,
        x=list(contour.x or []),
        y=list(contour.y or []),
        added_by=contour.added_by,
        confidence=contour.confidence_score,
    )


def _prediction_error(exc: Exception, step: ResolvedStep) -> str:
    """A failure message that says which model failed, and what the service replied.

    The bare exception is close to useless in the failed-items list: an httpx error reads
    "Server error '500 Internal Server Error' for url .../annotation_session/run", which names
    neither the model nor the reason. Naming the step's model matters most when a plan mixes
    several -- one broken model should not read like "batch inference is broken". The response
    body is appended when the service sent one (4xx replies carry a `detail`; an unhandled
    500 does not, and its traceback is only in the AI service's own log).
    """
    detail = f"{type(exc).__name__}: {exc}"
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
            body = payload.get("detail", payload) if isinstance(payload, dict) else payload
        except Exception:
            body = (response.text or "").strip()
        if body and str(body).lower() != "internal server error":
            detail = f"{detail} -- {body}"
    return f"Model {step.model_registry_key!r} ({step.task}) failed: {detail}"


def _group_by_parent(candidates: list[Contour]) -> dict[int | None, list[Contour]]:
    grouped: dict[int | None, list[Contour]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.parent_id, []).append(candidate)
    return grouped


def _existing_contours(
    db: Session,
    mask_id: int,
    *,
    label_id: int | None = None,
    parent_id: int | None = None,
    same_level: bool = False,
) -> list[Contours]:
    """Contours already on a mask, optionally restricted to one label or one sibling group.

    ``same_level`` selects the group a candidate would join: root contours when ``parent_id``
    is None, that parent's children otherwise. Without it, ``parent_id=None`` would be read
    as "no filter" and root candidates would be compared against every contour on the image.
    """
    query = db.query(Contours).filter(Contours.mask_id == mask_id, Contours.temporary.is_(False))
    if label_id is not None:
        query = query.filter(Contours.label_id == label_id)
    if same_level:
        query = query.filter(
            Contours.parent_id.is_(None) if parent_id is None else Contours.parent_id == parent_id
        )
    elif parent_id is not None:
        query = query.filter(Contours.parent_id == parent_id)
    return query.all()


# --------------------------------------------------------------------------- #
# Replace
# --------------------------------------------------------------------------- #
def wipe_images(
    db: Session, image_ids: list[int], *, preserve_reviewed: bool = True
) -> int:
    """Delete the contours in scope ahead of a replace run. Commits in batches.

    Deleting a contour takes its child objects with it (``ON DELETE CASCADE`` on
    ``contours.parent_id``), which is the whole point of the warning the UI shows: removing a
    cell removes the nuclei annotated inside it, however carefully those were drawn.

    With ``preserve_reviewed`` the approved objects -- and their descendants, so an approval
    never ends up describing a half-deleted subtree -- are kept. Masks are marked not fully
    annotated either way: whatever the run produces has not been through review.

    Returns the number of contours deleted.
    """
    if not image_ids:
        return 0
    from app.services.inference.planning import protected_contour_ids

    protected = protected_contour_ids(db, image_ids) if preserve_reviewed else set()

    mask_ids = [
        row[0] for row in db.query(Masks.id).filter(Masks.image_id.in_(image_ids)).all()
    ]
    deleted = 0
    for chunk_start in range(0, len(mask_ids), 200):
        chunk = mask_ids[chunk_start:chunk_start + 200]
        query = db.query(Contours).filter(Contours.mask_id.in_(chunk))
        if protected:
            query = query.filter(Contours.id.notin_(protected))
        # Counted before the DELETE rather than taken from its rowcount: SQLite does not
        # report rows removed by a foreign-key cascade, so a parent taking its children with
        # it would be counted as one deletion instead of the whole subtree.
        deleted += query.with_entities(func.count(Contours.id)).scalar() or 0
        query.delete(synchronize_session=False)
        db.query(Masks).filter(Masks.id.in_(chunk)).update(
            {Masks.fully_annotated: False}, synchronize_session=False
        )
        db.commit()
    logger.info("Replace run deleted %d contours across %d images.", deleted, len(image_ids))
    return deleted
