"""Conditioning dispatcher and resolvers for inference execution.

Resolves model-required conditioning inputs (reference images, instances, embeddings,
concept text, or none) based on the model's effective InputContract before worker execution.
"""
from __future__ import annotations

from collections import OrderedDict
from logging import getLogger
from typing import TYPE_CHECKING, Any, Sequence

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.embeddings import get_embedding_vector
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.schemas.inference import ResolvedStep
from app.services.exemplar_retrieval import (
    IMAGE_CLS,
    REGION_MEAN,
    ExemplarMatch,
    RetrievalQuery,
    is_region_based_strategy,
    retrieve_exemplars,
)
from config import EMBEDDING_MODEL_ID

if TYPE_CHECKING:
    from iquana_toolbox.schemas.networking.http.services import CrossImageExemplar

logger = getLogger(__name__)


def is_image_fully_annotated(session: Session, image_id: int) -> bool:
    """Check if an image has a mask marked as fully_annotated with no open rejections."""
    masks = session.query(Masks).filter(Masks.image_id == image_id).all()
    if not masks:
        return False
    for m in masks:
        if m.fully_annotated:
            has_open_rejection = any(r.is_open for r in m.rejections) if m.rejections else False
            if not has_open_rejection:
                return True
    return False


def resolve_reference_images(
    session: Session,
    *,
    target_image_id: int,
    dataset_id: int,
    strategy: str,
    concept_label_id: int | None = None,
    query_contour_id: int | None = None,
    max_images: int = 1,
    requires_complete_annotation: bool = True,
    model_id: str = EMBEDDING_MODEL_ID,
) -> tuple[list["CrossImageExemplar"], list[ExemplarMatch]]:
    """Resolve reference image exemplars for cross-image conditioning.

    Steps:
    1. Retrieve ranked exemplar matches using the specified strategy.
    2. Exclude the target image (an image cannot be its own exemplar).
    3. Group matches by source image_id preserving the best rank per image.
    4. If requires_complete_annotation is True, filter candidate images to only those fully annotated.
    5. Select up to max_images unique reference images.
    6. Resolve pixel image paths and binary masks for the selected exemplar contours.

    Returns:
        A tuple (exemplars, selected_matches).
    """
    from iquana_toolbox.schemas.database.contours import Contour
    from iquana_toolbox.schemas.networking.http.services import CrossImageExemplar

    target = session.get(Images, target_image_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Target image {target_image_id} not found.")

    query_vector = None
    if query_contour_id is not None and is_region_based_strategy(strategy):
        contour_row = (
            session.query(Contours.id)
            .join(Masks, Masks.id == Contours.mask_id)
            .join(Images, Images.id == Masks.image_id)
            .filter(Contours.id == query_contour_id, Images.dataset_id == dataset_id)
            .first()
        )
        if contour_row is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Query contour {query_contour_id} not found in dataset {dataset_id}.",
            )
        query_vector = get_embedding_vector(
            session, contour_id=query_contour_id, kind=REGION_MEAN, model_id=model_id
        )
        if query_vector is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Query contour {query_contour_id} has no '{REGION_MEAN}' embedding for model "
                f"{model_id!r}; embed it first.",
            )

    # Retrieve candidate matches with progressive search to avoid small hard caps
    # when filtering candidates by complete annotation. Search until candidate source is exhausted.
    eligible_image_ids: list[int] = []
    matches_by_image: dict[int, list[ExemplarMatch]] = OrderedDict()
    batch_top_k = 64

    while len(eligible_image_ids) < max_images:
        raw_matches = retrieve_exemplars(
            session,
            strategy,
            RetrievalQuery(
                dataset_id=dataset_id,
                target_image_id=target_image_id,
                concept_label_id=concept_label_id,
                query_vector=query_vector,
                top_k=batch_top_k,
            ),
            model_id=model_id,
        )
        if not raw_matches:
            break

        matches_by_image.clear()
        for match in raw_matches:
            if match.image_id == target_image_id:
                continue
            if match.image_id not in matches_by_image:
                matches_by_image[match.image_id] = []
            matches_by_image[match.image_id].append(match)

        eligible_image_ids = []
        for image_id in matches_by_image.keys():
            if requires_complete_annotation:
                if not is_image_fully_annotated(session, image_id):
                    continue
            eligible_image_ids.append(image_id)
            if len(eligible_image_ids) >= max_images:
                break

        if len(eligible_image_ids) >= max_images or len(raw_matches) < batch_top_k:
            break
        batch_top_k *= 2

    if not eligible_image_ids:
        return [], []

    # Collect selected matches from chosen reference images.
    # Take the highest-ranked exemplar from each chosen image.
    selected_matches: list[ExemplarMatch] = []
    for img_id in eligible_image_ids:
        img_matches = matches_by_image[img_id]
        selected_matches.extend(img_matches[:1])

    contour_ids = [m.contour_id for m in selected_matches]
    rows = (
        session.query(Contours, Images)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Contours.id.in_(contour_ids))
        .all()
    )
    by_id = {contour.id: (contour, image) for contour, image in rows}

    exemplars: list[CrossImageExemplar] = []
    for match in selected_matches:
        pair = by_id.get(match.contour_id)
        if pair is None:
            logger.warning("Exemplar contour %s vanished before resolution; skipping.", match.contour_id)
            continue
        contour, image = pair
        mask = Contour(
            x=contour.x,
            y=contour.y,
            confidence=contour.confidence_score,
        ).to_binary_mask_model(image.height, image.width)
        exemplars.append(CrossImageExemplar(image_url=str(image.file_path), mask=mask))

    return exemplars, selected_matches


def resolve_instance_conditioning(
    session: Session,
    *,
    dataset_id: int,
    concept_label_id: int | None,
    max_instances: int = 5,
    target_image_id: int | None = None,
) -> tuple[list[int], list[Any], list["CrossImageExemplar"]]:
    """Resolve ranked exemplar contour IDs, materialized BinaryMasks, and source-aware CrossImageExemplars."""
    from iquana_toolbox.schemas.database.contours import Contour
    from iquana_toolbox.schemas.networking.http.services import CrossImageExemplar

    q = (
        session.query(Contours, Images)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(
            Images.dataset_id == dataset_id,
            Contours.temporary.is_(False),
        )
    )
    if concept_label_id is not None:
        q = q.filter(Contours.label_id == concept_label_id)
    if target_image_id is not None:
        q = q.filter(Images.id != target_image_id)

    q = q.order_by(Contours.reviewed_by.any().desc(), Contours.created_at.desc())
    if max_instances:
        q = q.limit(max_instances)

    rows = q.all()
    contour_ids: list[int] = []
    positive_exemplars: list[Any] = []
    cross_image_exemplars: list[CrossImageExemplar] = []
    for contour, image in rows:
        contour_ids.append(contour.id)
        bm = Contour(
            x=contour.x,
            y=contour.y,
            confidence=contour.confidence_score,
        ).to_binary_mask_model(image.height, image.width)
        positive_exemplars.append(bm)
        cross_image_exemplars.append(CrossImageExemplar(image_url=str(image.file_path), mask=bm))

    return contour_ids, positive_exemplars, cross_image_exemplars


CONTOUR_SCOPED_EMBEDDING_KINDS = {REGION_MEAN}


def resolve_embedding_conditioning(
    session: Session,
    *,
    target_image_id: int | None = None,
    dataset_id: int | None = None,
    contour_id: int | None = None,
    kind: str | None = None,
    kinds: Sequence[str] | None = None,
    model_id: str = EMBEDDING_MODEL_ID,
) -> dict[str, list[float]] | list[float] | None:
    """Resolve precomputed embedding vectors for requested kind(s) against appropriate subject.

    - Whole-image embedding kinds (e.g. `image_cls`) query the store using `image_id`.
    - Object/contour-level embedding kinds (e.g. `region_mean`) query the store using `contour_id`,
      scoped to the authorized dataset.
    """
    def _find_target_contour_id(img_id: int) -> int | None:
        c = (
            session.query(Contours.id)
            .join(Masks, Masks.id == Contours.mask_id)
            .filter(Masks.image_id == img_id)
            .order_by(
                Contours.reviewed_by.any().desc(),
                Contours.created_at.desc(),
                Contours.id.desc(),
            )
            .first()
        )
        return c[0] if c else None

    def _resolve_contour_id(cid: int | None, img_id: int | None) -> int | None:
        if cid is not None:
            if dataset_id is not None:
                # Verify that the explicit contour_id belongs to dataset_id
                in_ds = (
                    session.query(Contours.id)
                    .join(Masks, Masks.id == Contours.mask_id)
                    .join(Images, Images.id == Masks.image_id)
                    .filter(Contours.id == cid, Images.dataset_id == dataset_id)
                    .first()
                )
                if in_ds is None:
                    return None
            return cid
        if img_id is not None:
            return _find_target_contour_id(img_id)
        return None

    if kind is not None:
        if kind in CONTOUR_SCOPED_EMBEDDING_KINDS or (contour_id is not None and target_image_id is None):
            cid = _resolve_contour_id(contour_id, target_image_id)
            if cid is None:
                return None
            return get_embedding_vector(
                session, image_id=None, contour_id=cid, kind=kind, model_id=model_id
            )
        else:
            if target_image_id is None:
                return None
            return get_embedding_vector(
                session, image_id=target_image_id, contour_id=None, kind=kind, model_id=model_id
            )

    requested_kinds = kinds if kinds is not None else (IMAGE_CLS,)
    vectors: dict[str, list[float]] = {}
    for k in requested_kinds:
        if k in CONTOUR_SCOPED_EMBEDDING_KINDS:
            cid = _resolve_contour_id(contour_id, target_image_id)
            if cid is not None:
                vec = get_embedding_vector(
                    session, image_id=None, contour_id=cid, kind=k, model_id=model_id
                )
                if vec is not None:
                    vectors[k] = vec
        else:
            if target_image_id is not None:
                vec = get_embedding_vector(
                    session, image_id=target_image_id, contour_id=None, kind=k, model_id=model_id
                )
                if vec is not None:
                    vectors[k] = vec
    return vectors


def resolve_concept_text(
    *,
    step_inputs: dict[str, Any],
    label: Labels | None = None,
) -> str | None:
    """Extract concept text from normalized step inputs or target label name."""
    cond_inputs = step_inputs.get("conditioning", {})
    if cond_inputs.get("concept_text"):
        return str(cond_inputs["concept_text"])
    if label is not None:
        return label.name
    return None


def dispatch_conditioning(
    session: Session,
    step: ResolvedStep,
    target_image: Images,
    username: str = "system",
) -> dict[str, Any]:
    """Unified closed-enum conditioning dispatcher for worker execution.

    Inspects step.input_contract.conditioning.kind and resolves the required payload.

    Returns a dict with:
        "kind": conditioning kind string ("none", "concept_text", "reference_images", "instances", "embeddings")
        and associated resolved payloads.

    Raises:
        ValueError: If required conditioning cannot satisfy the model contract's min_units,
            if required embeddings are missing, or if conditioning kind is unsupported.
    """
    kind = step.input_contract.conditioning.kind
    cond_spec = step.input_contract.conditioning
    cond_inputs = step.inputs.get("conditioning", {})

    if kind == "none":
        return {"kind": "none"}

    if kind == "concept_text":
        label_row = session.get(Labels, step.label_id)
        text = resolve_concept_text(step_inputs=step.inputs, label=label_row)
        from iquana_toolbox.schemas.database.labels import Label
        if label_row is not None:
            concept = Label(
                id=label_row.id,
                dataset_id=label_row.dataset_id,
                name=text or label_row.name,
                value=label_row.value or 0,
                parent_id=label_row.parent_id,
                children=[],
            )
        elif text:
            concept = Label(
                id=0,
                dataset_id=0,
                name=text,
                value=0,
                parent_id=None,
                children=[],
            )
        else:
            concept = None
        return {
            "kind": "concept_text",
            "concept_text": text,
            "concept": concept,
        }

    if kind == "reference_images":
        strategy = cond_inputs.get("strategy") or "global_scene"
        query_contour_id = cond_inputs.get("query_contour_id")
        count = cond_inputs.get("count")
        if count is None or count < 1:
            count = cond_spec.max_units or 1
        if cond_spec.max_units is not None:
            count = min(count, cond_spec.max_units)
        if cond_spec.min_units is not None:
            count = max(count, cond_spec.min_units)

        exemplars, matches = resolve_reference_images(
            session,
            target_image_id=target_image.id,
            dataset_id=target_image.dataset_id,
            strategy=strategy,
            concept_label_id=step.label_id,
            query_contour_id=query_contour_id,
            max_images=count,
            requires_complete_annotation=cond_spec.requires_complete_annotation,
        )

        unique_images = {ex.image_url for ex in exemplars}
        if cond_spec.min_units > 0 and len(unique_images) < cond_spec.min_units:
            raise ValueError(
                f"Resolved {len(unique_images)} unique reference image(s) for label {step.label_id} "
                f"on image {target_image.id}, but model contract for '{step.task}' requires at least "
                f"{cond_spec.min_units} min_units."
            )

        from iquana_toolbox.schemas.database.labels import Label
        label_row = session.get(Labels, step.label_id)
        concept = Label.from_db(label_row) if label_row is not None else None

        return {
            "kind": "reference_images",
            "exemplars": exemplars,
            "matches": matches,
            "contour_ids": [m.contour_id for m in matches],
            "positive_exemplars": [ex.mask.model_dump(mode="json") if hasattr(ex.mask, "model_dump") else ex.mask for ex in exemplars],
            "concept": concept,
        }

    if kind == "instances":
        strategy = cond_inputs.get("strategy")
        query_contour_id = cond_inputs.get("query_contour_id")
        count = cond_inputs.get("count") or 5
        if cond_spec.max_units is not None:
            count = min(count, cond_spec.max_units)
        if cond_spec.min_units is not None:
            count = max(count, cond_spec.min_units)

        from iquana_toolbox.schemas.database.labels import Label
        label_row = session.get(Labels, step.label_id)
        concept = Label.from_db(label_row) if label_row is not None else None

        if strategy:
            query_vector = None
            if query_contour_id is not None and is_region_based_strategy(strategy):
                contour_row = (
                    session.query(Contours.id)
                    .join(Masks, Masks.id == Contours.mask_id)
                    .join(Images, Images.id == Masks.image_id)
                    .filter(Contours.id == query_contour_id, Images.dataset_id == target_image.dataset_id)
                    .first()
                )
                if contour_row is None:
                    raise HTTPException(
                        status.HTTP_404_NOT_FOUND,
                        f"Query contour {query_contour_id} not found in dataset {target_image.dataset_id}.",
                    )
                query_vector = get_embedding_vector(
                    session, contour_id=query_contour_id, kind=REGION_MEAN, model_id=EMBEDDING_MODEL_ID
                )
                if query_vector is None:
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        f"Query contour {query_contour_id} has no '{REGION_MEAN}' embedding for model "
                        f"{EMBEDDING_MODEL_ID!r}; embed it first.",
                    )

            raw_matches = retrieve_exemplars(
                session,
                strategy,
                RetrievalQuery(
                    dataset_id=target_image.dataset_id,
                    target_image_id=target_image.id,
                    concept_label_id=step.label_id,
                    query_vector=query_vector,
                    top_k=count,
                ),
                model_id=EMBEDDING_MODEL_ID,
            )
            matches = [m for m in raw_matches if m.image_id != target_image.id][:count]
            contour_ids = [m.contour_id for m in matches]
            rows = (
                session.query(Contours, Images)
                .join(Masks, Masks.id == Contours.mask_id)
                .join(Images, Images.id == Masks.image_id)
                .filter(Contours.id.in_(contour_ids))
                .all()
            )
            by_id = {contour.id: (contour, image) for contour, image in rows}
            exemplars: list[CrossImageExemplar] = []
            positive_exemplars: list[Any] = []
            for match in matches:
                pair = by_id.get(match.contour_id)
                if pair is None:
                    continue
                contour, image = pair
                from iquana_toolbox.schemas.database.contours import Contour
                from iquana_toolbox.schemas.networking.http.services import CrossImageExemplar
                bm = Contour(
                    x=contour.x,
                    y=contour.y,
                    confidence=contour.confidence_score,
                ).to_binary_mask_model(image.height, image.width)
                positive_exemplars.append(bm)
                exemplars.append(CrossImageExemplar(image_url=str(image.file_path), mask=bm))
        else:
            matches = []
            contour_ids, positive_exemplars, exemplars = resolve_instance_conditioning(
                session,
                dataset_id=target_image.dataset_id,
                concept_label_id=step.label_id,
                max_instances=count,
                target_image_id=target_image.id,
            )

        if cond_spec.min_units > 0 and len(contour_ids) < cond_spec.min_units:
            raise ValueError(
                f"Resolved {len(contour_ids)} instance exemplar(s) for label {step.label_id} "
                f"on image {target_image.id}, but model contract for '{step.task}' requires at least "
                f"{cond_spec.min_units} min_units."
            )
        return {
            "kind": "instances",
            "contour_ids": contour_ids,
            "positive_exemplars": [bm.model_dump(mode="json") if hasattr(bm, "model_dump") else bm for bm in positive_exemplars],
            "exemplars": exemplars,
            "matches": matches,
            "concept": concept,
        }

    if kind == "embeddings":
        kinds = cond_spec.embedding_kinds or [IMAGE_CLS]
        query_contour_id = cond_inputs.get("query_contour_id")
        vectors = resolve_embedding_conditioning(
            session,
            target_image_id=target_image.id,
            dataset_id=target_image.dataset_id,
            contour_id=query_contour_id,
            kinds=kinds,
            model_id=EMBEDDING_MODEL_ID,
        )
        missing = [k for k in kinds if k not in vectors]
        if missing:
            raise ValueError(
                f"Image {target_image.id} is missing required embedding(s) {missing} "
                f"for model {EMBEDDING_MODEL_ID!r}; run the embedding backfill first."
            )
        if cond_spec.min_units > 0 and len(vectors) < cond_spec.min_units:
            raise ValueError(
                f"Resolved {len(vectors)} embedding vector(s) for image {target_image.id}, "
                f"but model contract requires at least {cond_spec.min_units} min_units."
            )
        return {
            "kind": "embeddings",
            "vectors": vectors,
            "vector": vectors.get(kinds[0]) if len(kinds) == 1 else None,
        }

    raise ValueError(f"Unsupported conditioning kind '{kind}' for task '{step.task}'")
