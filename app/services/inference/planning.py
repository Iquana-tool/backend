"""Turning a requested orchestration into a frozen, hierarchy-ordered work list.

Everything here is read-mostly and synchronous, so the whole planning step is unit-testable
without a broker or an AI service. It answers four questions:

1. **Which models may a label be pointed at?** (:func:`model_catalog`) -- every ready model
   advertising an instance-producing task, annotated with whether it was trained on *this*
   dataset and which labels it natively predicts.
2. **Which images are in scope?** (:func:`resolve_scope`) -- resolved once, at submit time,
   and frozen on the job.
3. **In what order do the steps run?** (:func:`resolve_steps`) -- by label depth. This is the
   load-bearing decision of the whole feature: a child-level prediction is nested under a
   parent instance, so every parent must already exist in the database before the child model
   is allowed to run. Sorting the steps is not enough on its own -- the *work list* is
   ordered by level too, so level 1 does not start on image 1 until level 0 has finished
   image N.
4. **What would a replace run destroy?** (:func:`replace_preview`) -- the numbers the
   confirmation dialog puts in front of the user before they agree to it.
"""
from __future__ import annotations

from collections import deque
from logging import getLogger

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.images import Images
from app.database.inference_jobs import InferenceJobItems, InferenceJobs, TERMINAL_JOB_STATUSES
from app.database.labels import Labels
from app.database.masks import Masks
from app.schemas.inference import (
    ImageSelection,
    InferenceJobCreate,
    InferenceStepRequest,
    ModelCatalog,
    ModelOption,
    ReplacePreview,
    ResolvedStep,
    ScopeCounts,
    WriteMode,
)
from app.services.exemplar_retrieval import strategy_options
from app.services.inference.contract_resolver import resolve_input_contract
from app.services.inference.input_validator import validate_and_normalize_inputs
from app.services.model_registry import list_available_models

logger = getLogger(__name__)

#: AI-service task surfaces whose output is a set of instances on one image -- the only kinds
#: of model a plan step can be bound to. Prompted segmentation is excluded on purpose: it
#: needs a click or a box per object, which nobody is there to give in a batch run.
BATCH_TASKS: tuple[str, ...] = ("instance-segmentation", "cross-image-suggestion")


# --------------------------------------------------------------------------- #
# The label hierarchy
# --------------------------------------------------------------------------- #
def label_levels(db: Session, dataset_id: int) -> dict[int, int]:
    """Map every label id in a dataset to its depth (0 for root labels).

    Walked breadth-first from the roots rather than by following ``parent_id`` upwards per
    label, so a hierarchy of any depth costs one query and one pass. A label whose parent is
    missing (a broken row) is unreachable from the roots and is simply absent from the
    result, which surfaces as a 404 when a step targets it.
    """
    rows = db.query(Labels.id, Labels.parent_id).filter(Labels.dataset_id == dataset_id).all()
    children: dict[int | None, list[int]] = {}
    for label_id, parent_id in rows:
        children.setdefault(parent_id, []).append(label_id)

    levels: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque((label_id, 0) for label_id in children.get(None, []))
    while queue:
        label_id, depth = queue.popleft()
        levels[label_id] = depth
        queue.extend((child_id, depth + 1) for child_id in children.get(label_id, []))
    return levels


# --------------------------------------------------------------------------- #
# The model catalog
# --------------------------------------------------------------------------- #
def _dataset_label_ids(db: Session, dataset_id: int) -> set[int]:
    return {row[0] for row in db.query(Labels.id).filter(Labels.dataset_id == dataset_id).all()}


def model_catalog(db: Session, dataset_id: int) -> ModelCatalog:
    """Every model a step in this dataset may be bound to, plus the retrieval strategies.

    A model's ``label_ids`` is what makes per-label orchestration work. A model fine-tuned on
    this dataset carries the labels it was trained on, so the picker can offer it under
    exactly those labels and the executor can filter its output. A base model carries none,
    which means class-agnostic: it may be bound to any label, and whatever it returns is
    labelled with that step's label.

    Cross-image models requiring exemplar retrieval are only offered when at least one retrieval
    strategy is actually available. Models with self-contained or concept-text conditioning remain
    available even without embeddings.
    """
    dataset_labels = _dataset_label_ids(db, dataset_id)
    # Scoped to this dataset: a strategy that ranks by visual similarity is only listed
    # where the embeddings it needs actually exist, so the picker never offers a step
    # that would fail on every image.
    strategies = [option.model_dump() for option in strategy_options(db, dataset_id)]
    cross_image_usable = any(option.get("available") for option in strategies)

    options: list[ModelOption] = []
    for task in BATCH_TASKS:
        try:
            listing = list_available_models(task)
        except Exception:
            logger.exception("Could not list %s models; the picker will omit them.", task)
            continue
        for info in listing.get("result", []):
            try:
                contract, provenance = resolve_input_contract(info, task)
            except Exception:
                logger.exception(
                    "Failed to resolve input contract for model '%s', task '%s'; omitting model.",
                    info.get("name") or info.get("registry_key"),
                    task,
                )
                continue

            # If conditioning requires external exemplar retrieval (reference_images/embeddings/instances)
            # and no retrieval strategies are usable, skip offering this option.
            # Models with 'none' or 'concept_text' conditioning are always included regardless of embeddings.
            if (
                task == "cross-image-suggestion"
                and not cross_image_usable
                and contract.conditioning.kind in {"reference_images", "embeddings", "instances"}
            ):
                continue

            label_ids = [int(lid) for lid in info.get("label_ids") or []]
            options.append(ModelOption(
                registry_key=info.get("registry_key") or info.get("name"),
                name=info.get("name") or info.get("registry_key"),
                task=task,
                description=info.get("description"),
                usage_tip=info.get("usage_tip"),
                badges=list(info.get("badges") or []),
                architecture=info.get("architecture"),
                label_ids=label_ids,
                # A model whose classes are labels of this dataset was fine-tuned on it.
                trained_on_dataset=bool(label_ids) and bool(set(label_ids) & dataset_labels),
                input_contract=contract,
                provenance=provenance,
            ))

    options.sort(key=lambda option: (not option.trained_on_dataset, option.name.lower()))
    return ModelCatalog(models=options, retrieval_strategies=strategies)


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
def _not_started_image_ids(db: Session, dataset_id: int) -> list[int]:
    """Images whose masks hold no contours at all."""
    annotated = (
        db.query(Masks.image_id)
        .join(Contours, Contours.mask_id == Masks.id)
        .distinct()
        .subquery()
    )
    rows = (
        db.query(Images.id)
        .filter(Images.dataset_id == dataset_id, Images.id.notin_(annotated))
        .order_by(Images.id)
        .all()
    )
    return [row[0] for row in rows]


def _unreviewed_image_ids(db: Session, dataset_id: int) -> list[int]:
    """Images that are not done: no contours yet, or at least one nobody has approved."""
    has_unreviewed = (
        db.query(Masks.image_id)
        .join(Contours, Contours.mask_id == Masks.id)
        .filter(~Contours.reviewed_by.any())
        .distinct()
        .subquery()
    )
    not_started = set(_not_started_image_ids(db, dataset_id))
    rows = (
        db.query(Images.id)
        .filter(Images.dataset_id == dataset_id, Images.id.in_(has_unreviewed))
        .all()
    )
    return sorted(not_started | {row[0] for row in rows})


def scope_counts(db: Session, dataset_id: int) -> ScopeCounts:
    """Image counts per selection, so the scope picker can label its options."""
    total = db.query(func.count(Images.id)).filter(Images.dataset_id == dataset_id).scalar() or 0
    return ScopeCounts(
        total=int(total),
        not_started=len(_not_started_image_ids(db, dataset_id)),
        unreviewed=len(_unreviewed_image_ids(db, dataset_id)),
    )


def resolve_scope(db: Session, dataset_id: int, body: InferenceJobCreate) -> list[int]:
    """The image ids a run covers, frozen at submit time.

    Freezing matters: a run over a few thousand images takes hours, and an upload that lands
    halfway through must not silently join it -- the progress bar would move backwards and
    nobody could say afterwards what the run actually touched.
    """
    if body.image_selection == ImageSelection.CUSTOM:
        rows = (
            db.query(Images.id)
            .filter(Images.dataset_id == dataset_id, Images.id.in_(body.image_ids))
            .order_by(Images.id)
            .all()
        )
        found = [row[0] for row in rows]
        missing = sorted(set(body.image_ids) - set(found))
        if missing:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Images not found in dataset {dataset_id}: {missing[:10]}"
                + (" ..." if len(missing) > 10 else ""),
            )
        return found
    if body.image_selection == ImageSelection.NOT_STARTED:
        return _not_started_image_ids(db, dataset_id)
    if body.image_selection == ImageSelection.UNREVIEWED:
        return _unreviewed_image_ids(db, dataset_id)
    rows = db.query(Images.id).filter(Images.dataset_id == dataset_id).order_by(Images.id).all()
    return [row[0] for row in rows]


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def resolve_steps(
    db: Session, dataset_id: int, steps: list[InferenceStepRequest]
) -> list[ResolvedStep]:
    """Validate each step against the dataset and the registry, then order by hierarchy.

    The returned order *is* the execution order. Steps are sorted by label depth first (root
    labels before their children) and by label name within a level, which only affects the
    order two independent root labels are annotated in.

    Raises 404 for a label that is not in the dataset or a model that is not registered for
    the step's task, and 400 for a model that cannot produce the step's label.
    """
    levels = label_levels(db, dataset_id)
    labels = {
        label.id: label
        for label in db.query(Labels).filter(Labels.dataset_id == dataset_id).all()
    }
    catalog = {(option.registry_key, option.task): option for option in model_catalog(db, dataset_id).models}

    resolved: list[ResolvedStep] = []
    for step in steps:
        label = labels.get(step.label_id)
        if label is None or step.label_id not in levels:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Label {step.label_id} is not part of dataset {dataset_id}'s hierarchy.",
            )
        option = catalog.get((step.model_registry_key, step.task))
        if option is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"No ready '{step.task}' model named {step.model_registry_key!r}.",
            )
        # A model that declares its classes can only be asked for one of them. A model that
        # declares none is class-agnostic and may be bound to any label.
        if option.label_ids and step.label_id not in option.label_ids:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Model {option.name!r} does not predict label {label.name!r}; it predicts "
                f"label ids {sorted(option.label_ids)}.",
            )
        # Extract inputs or synthesize from legacy request fields
        raw_inputs = step.inputs
        if raw_inputs is None:
            raw_cond: dict[str, Any] = {}
            if step.task == "cross-image-suggestion":
                if step.retrieval_strategy is not None:
                    raw_cond["strategy"] = step.retrieval_strategy
                cond = option.input_contract.conditioning
                top_k = step.top_k if step.top_k is not None else 5
                if cond.user_selectable_count:
                    count = top_k
                    if cond.max_units is not None:
                        count = min(count, cond.max_units)
                    if cond.min_units is not None:
                        count = max(count, cond.min_units)
                    raw_cond["count"] = count
                elif cond.kind in ("reference_images", "instances", "embeddings"):
                    raw_cond["count"] = cond.max_units or cond.min_units or 1

            raw_params: dict[str, Any] = {}
            declared_param_keys = {p.key for p in option.input_contract.parameters}
            if "threshold" in declared_param_keys and step.min_confidence is not None and step.min_confidence > 0.0:
                raw_params["threshold"] = step.min_confidence
            raw_inputs = {"conditioning": raw_cond, "parameters": raw_params}

        try:
            normalized_inputs = validate_and_normalize_inputs(option.input_contract, raw_inputs)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Invalid inference inputs for model {option.name!r} (task '{step.task}'): {exc}",
            ) from exc

        resolved.append(ResolvedStep(
            label_id=step.label_id,
            model_registry_key=step.model_registry_key,
            task=step.task,
            level=levels[step.label_id],
            parent_label_id=label.parent_id,
            label_name=label.name,
            model_name=option.name,
            model_label_ids=list(option.label_ids),
            inputs=normalized_inputs,
            input_contract=option.input_contract,
            provenance=option.provenance,
            min_confidence=step.min_confidence,
            retrieval_strategy=step.retrieval_strategy,
            top_k=step.top_k,
        ))

    resolved.sort(key=lambda step: (step.level, step.label_name.lower()))
    return resolved


# --------------------------------------------------------------------------- #
# Replace preview
# --------------------------------------------------------------------------- #
def protected_contour_ids(db: Session, image_ids: list[int]) -> set[int]:
    """Approved contours in scope, plus their descendants.

    A reviewed cell is kept together with the nuclei inside it: deleting the children of an
    object somebody signed off on would leave that approval describing something that no
    longer exists.
    """
    if not image_ids:
        return set()
    approved = {
        row[0] for row in
        db.query(Contours.id)
        .join(Masks, Masks.id == Contours.mask_id)
        .filter(Masks.image_id.in_(image_ids), Contours.reviewed_by.any())
        .all()
    }
    if not approved:
        return set()

    parents = {
        contour_id: parent_id for contour_id, parent_id in
        db.query(Contours.id, Contours.parent_id)
        .join(Masks, Masks.id == Contours.mask_id)
        .filter(Masks.image_id.in_(image_ids))
        .all()
    }
    children: dict[int, list[int]] = {}
    for contour_id, parent_id in parents.items():
        if parent_id is not None:
            children.setdefault(parent_id, []).append(contour_id)

    protected = set(approved)
    queue = deque(approved)
    while queue:
        for child_id in children.get(queue.popleft(), []):
            if child_id not in protected:
                protected.add(child_id)
                queue.append(child_id)
    return protected


def replace_preview(
    db: Session, image_ids: list[int], *, preserve_reviewed: bool = True
) -> ReplacePreview:
    """Count what a replace run would delete, for the confirmation dialog.

    The dialog names the reviewed count separately because that is the number people care
    about: unreviewed model output is cheap to regenerate, an approved annotation is not.
    """
    if not image_ids:
        return ReplacePreview(images=0, contours=0, reviewed_contours=0, root_contours=0)

    base = db.query(Contours).join(Masks, Masks.id == Contours.mask_id).filter(
        Masks.image_id.in_(image_ids)
    )
    total = base.with_entities(func.count(Contours.id)).scalar() or 0
    roots = base.filter(Contours.parent_id.is_(None)).with_entities(
        func.count(Contours.id)
    ).scalar() or 0
    reviewed = base.filter(Contours.reviewed_by.any()).with_entities(
        func.count(Contours.id)
    ).scalar() or 0

    protected = len(protected_contour_ids(db, image_ids)) if preserve_reviewed else 0
    return ReplacePreview(
        images=len(image_ids),
        contours=int(total) - protected,
        reviewed_contours=int(reviewed),
        root_contours=int(roots),
        protected_contours=protected,
    )


# --------------------------------------------------------------------------- #
# Job creation
# --------------------------------------------------------------------------- #
def active_job(db: Session, dataset_id: int) -> InferenceJobs | None:
    """The dataset's running (or queued) job, if any."""
    return (
        db.query(InferenceJobs)
        .filter(
            InferenceJobs.dataset_id == dataset_id,
            InferenceJobs.status.notin_(tuple(TERMINAL_JOB_STATUSES)),
        )
        .order_by(InferenceJobs.id.desc())
        .first()
    )


def create_job(
    db: Session, dataset_id: int, username: str, body: InferenceJobCreate
) -> InferenceJobs:
    """Validate, resolve and persist a run. Does not enqueue it -- the route does that.

    The whole work list is written here, one row per (step, image), each stamped with its
    step's hierarchy level. From this point on the worker never has to reason about ordering:
    "the next unit" is the lowest-level pending row.
    """
    if body.options.write_mode == WriteMode.REPLACE and not body.confirm_replace:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "A replace run deletes existing annotations and their child objects. "
            "Set confirm_replace=true to acknowledge this.",
        )
    running = active_job(db, dataset_id)
    if running is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Inference job {running.id} is still {running.status} on this dataset. "
            "Wait for it to finish or cancel it first.",
        )

    steps = resolve_steps(db, dataset_id, body.steps)
    image_ids = resolve_scope(db, dataset_id, body)
    if not image_ids:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "No images match the selected scope."
        )

    job = InferenceJobs(
        dataset_id=dataset_id,
        created_by=username,
        name=body.name,
        status="pending",
        write_mode=body.options.write_mode.value,
        plan_steps=[step.model_dump(mode="json") for step in steps],
        options=body.options.model_dump(mode="json"),
        image_ids=image_ids,
        total_units=len(steps) * len(image_ids),
    )
    db.add(job)
    db.flush()

    db.bulk_insert_mappings(InferenceJobItems, [
        {
            "job_id": job.id,
            "level": step.level,
            "step_index": step_index,
            "image_id": image_id,
            "status": "pending",
        }
        for step_index, step in enumerate(steps)
        for image_id in image_ids
    ])
    db.commit()
    db.refresh(job)
    return job
