"""Glue the retrieval store to the cross-image concept segmenter.

The one non-trivial step between "user wants to annotate concept X on image Y" and a set of
suggested contours: pick the best exemplars from the store (retrieval strategy), turn those
exemplar *contour ids* back into image URLs + masks the ai-service can consume, and assemble
the ai-service request. The HTTP call itself is trivial and lives in the route.

:func:`build_cross_image_request` is deliberately synchronous and side-effect-free (a read over
the DB), so it is unit-testable without the ai-service. The new toolbox request/exemplar schemas
are imported lazily inside it -- this module is reachable from the app import path, and the
schemas only exist once the toolbox pin is bumped (the feature can't run before then anyway).
"""
from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.embeddings import get_embedding_vector
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.services.exemplar_retrieval import (
    REGION_MEAN,
    ExemplarMatch,
    RetrievalQuery,
    retrieve_exemplars,
)
from config import CROSS_IMAGE_MODEL_KEY, EMBEDDING_MODEL_ID

if TYPE_CHECKING:
    from iquana_toolbox.schemas.networking.http.services import CrossImageSuggestionRequest

logger = getLogger(__name__)

_SYSTEM_USER = "system"


def build_cross_image_request(
    session: Session,
    *,
    target_image_id: int,
    strategy: str,
    concept_label_id: int | None = None,
    query_contour_id: int | None = None,
    top_k: int = 5,
    model_id: str = EMBEDDING_MODEL_ID,
    cross_image_model_key: str = CROSS_IMAGE_MODEL_KEY,
    user_id: str = _SYSTEM_USER,
) -> tuple["CrossImageSuggestionRequest | None", list[ExemplarMatch]]:
    """Retrieve exemplars for a target image and assemble the ai-service request.

    Returns ``(request, matches)``. ``request`` is ``None`` when retrieval finds no exemplars
    (the caller then returns an empty suggestion rather than calling the model). Raises 404 for a
    missing target image and 400 when a region strategy's query contour has no embedding.
    """
    from iquana_toolbox.schemas.networking.http.services import CrossImageSuggestionRequest

    target = session.get(Images, target_image_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Target image {target_image_id} not found.")

    # A region-based strategy needs an explicit query vector; source it from an existing
    # contour's precomputed region embedding.
    query_vector = None
    if query_contour_id is not None:
        query_vector = get_embedding_vector(
            session, contour_id=query_contour_id, kind=REGION_MEAN, model_id=model_id
        )
        if query_vector is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Query contour {query_contour_id} has no '{REGION_MEAN}' embedding for model "
                f"{model_id!r}; embed it first.",
            )

    matches = retrieve_exemplars(
        session, strategy,
        RetrievalQuery(dataset_id=target.dataset_id, target_image_id=target_image_id,
                       concept_label_id=concept_label_id, query_vector=query_vector, top_k=top_k),
        model_id=model_id,
    )
    if not matches:
        return None, []

    exemplars = _resolve_exemplars(session, matches)
    if not exemplars:
        return None, []

    request = CrossImageSuggestionRequest(
        image_url=target.file_path,
        user_id=user_id,
        model_registry_key=cross_image_model_key,
        exemplars=exemplars,
        concept=_concept_label(session, concept_label_id),
    )
    return request, matches


def _resolve_exemplars(session: Session, matches: list[ExemplarMatch]) -> list:
    """Turn ranked exemplar contours into ai-service exemplars (image URL + rasterized mask)."""
    from iquana_toolbox.schemas.database.contours import Contour
    from iquana_toolbox.schemas.networking.http.services import CrossImageExemplar

    contour_ids = [m.contour_id for m in matches]
    rows = (
        session.query(Contours, Images)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Contours.id.in_(contour_ids))
        .all()
    )
    by_id = {contour.id: (contour, image) for contour, image in rows}

    exemplars = []
    for match in matches:  # preserve retrieval order (best first)
        pair = by_id.get(match.contour_id)
        if pair is None:
            logger.warning("Exemplar contour %s vanished before resolution; skipping.", match.contour_id)
            continue
        contour, image = pair
        mask = Contour(x=contour.x, y=contour.y, confidence=contour.confidence_score) \
            .to_binary_mask_model(image.height, image.width)
        exemplars.append(CrossImageExemplar(image_url=image.file_path, mask=mask))
    return exemplars


def _concept_label(session: Session, concept_label_id: int | None):
    """The toolbox ``Label`` for the concept (a text prompt for the segmenter), or ``None``."""
    if concept_label_id is None:
        return None
    from iquana_toolbox.schemas.database.labels import Label

    row = session.get(Labels, concept_label_id)
    return Label.from_db(row) if row is not None else None
