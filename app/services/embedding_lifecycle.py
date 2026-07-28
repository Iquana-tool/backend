"""Keeping the embedding store in step with the data it describes.

Two ways a subject gets embedded:

* **On write (opt-in).** When an image is uploaded or contours are saved, a Celery task
  embeds them in the background. Gated by ``EMBEDDING_LIFECYCLE_ENABLED`` (default off): until
  a validated ai-service embed surface + worker exist, writes enqueue nothing.
* **Backfill.** :func:`backfill_embeddings` (run via ``scripts/backfill_embeddings.py``) embeds
  existing images/contours that have no embedding yet -- the way to populate the store the
  first time, or after enabling a new strategy.

Both paths funnel through the same :func:`embed_image` / :func:`embed_contours`, which call the
ai-service ``embed`` surface and upsert the results. What actually gets computed is driven by
the enabled retrieval strategies' ``required_kinds`` (see :mod:`app.services.exemplar_retrieval`):
no enabled strategy needs a kind ⇒ it is never computed.

The concrete ``model_id`` used to version each vector comes back *in the embed response* (the
backbone the embedder ran), not from config -- config only names which registry model to call.
"""
from __future__ import annotations

from itertools import islice
from logging import getLogger
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence

from sqlalchemy.orm import Session

# The embed request/response schemas live in the toolbox and only exist once its pin is bumped.
# This module is reachable from the app import path (via the image-upload hook), so importing
# them at module level would crash startup before the bump. They are imported lazily inside the
# functions that build them -- which only run when the embedding feature is actually used.
if TYPE_CHECKING:
    from iquana_toolbox.schemas.networking.http.services import EmbedRequest, EmbeddingVector

from app.database.contours import Contours
from app.database.embeddings import Embeddings, upsert_embedding
from app.database.images import Images
from app.database.masks import Masks
from app.services.ai_services.embedding import EmbeddingService
from app.services.celery_app import celery_app
from app.services.exemplar_retrieval import RETRIEVAL_STRATEGIES
from config import EMBEDDING_LIFECYCLE_ENABLED, EMBEDDING_MODEL_KEY

logger = getLogger(__name__)

# Which kinds are whole-image vs per-region. Used both to split a request and to filter a
# response (a model must never write a region kind onto an image subject, or vice versa).
IMAGE_KINDS = frozenset({"image_cls"})
REGION_KINDS = frozenset({"region_mean"})

#: user_id stamped on background embed requests (no human initiates them).
_SYSTEM_USER = "system"


class EmbedClient(Protocol):
    """The slice of :class:`EmbeddingService` the lifecycle uses (so tests can fake it)."""

    def request_embeddings(self, request: EmbedRequest) -> list[EmbeddingVector]: ...


# --------------------------------------------------------------------------- #
# Which kinds to compute
# --------------------------------------------------------------------------- #
def required_kinds(available_only: bool = True) -> set[str]:
    """Union of ``required_kinds`` over the registered retrieval strategies.

    With ``available_only`` (the default) placeholder strategies are ignored, so enabling a
    strategy is what turns its kinds on. This is the single source of truth for what the
    lifecycle and backfill compute.
    """
    kinds: set[str] = set()
    for strategy in RETRIEVAL_STRATEGIES.values():
        if available_only and not strategy.available:
            continue
        kinds.update(strategy.required_kinds)
    return kinds


def _required_image_kinds() -> list[str]:
    return sorted(required_kinds() & IMAGE_KINDS)


def _region_kinds_needed() -> bool:
    return bool(required_kinds() & REGION_KINDS)


# --------------------------------------------------------------------------- #
# Embedding + storing
# --------------------------------------------------------------------------- #
def embed_image(
    session: Session,
    image: Images,
    *,
    model_registry_key: str = EMBEDDING_MODEL_KEY,
    client: EmbedClient | None = None,
) -> list[Embeddings]:
    """Compute + upsert the required whole-image embeddings for one image.

    No-op (returns ``[]``) when no enabled strategy needs a whole-image kind. Does not commit.
    """
    image_kinds = _required_image_kinds()
    if not image_kinds:
        return []
    from iquana_toolbox.schemas.networking.http.services import EmbedRequest

    client = client or EmbeddingService()

    request = EmbedRequest(
        image_url=image.file_path, user_id=_SYSTEM_USER,
        model_registry_key=model_registry_key, image_kinds=image_kinds, regions=[],
    )
    stored: list[Embeddings] = []
    for vector in client.request_embeddings(request):
        if vector.kind not in IMAGE_KINDS:  # ignore anything not an image kind
            continue
        stored.append(upsert_embedding(
            session, image_id=image.id, kind=vector.kind,
            model_id=vector.model_id, vector=vector.vector,
        ))
    return stored


def embed_contours(
    session: Session,
    contours: Sequence[Contours],
    *,
    model_registry_key: str = EMBEDDING_MODEL_KEY,
    client: EmbedClient | None = None,
) -> list[Embeddings]:
    """Compute + upsert the required region embeddings for a batch of contours.

    Contours are grouped by their source image so each image is embedded once (one request,
    many regions). Each contour's polygon is rasterized to a mask at the image's resolution --
    that mask selects the foreground patches the region descriptor is pooled over. No-op when
    no enabled strategy needs a region kind. Does not commit.
    """
    if not contours or not _region_kinds_needed():
        return []
    from iquana_toolbox.schemas.database.contours import Contour
    from iquana_toolbox.schemas.networking.http.services import EmbedRegion, EmbedRequest

    client = client or EmbeddingService()

    images = _contour_images(session, contours)
    by_image: dict[int, list[Contours]] = {}
    for contour in contours:
        image = images.get(contour.id)
        if image is None:
            logger.warning("No image for contour %s; skipping its region embedding.", contour.id)
            continue
        by_image.setdefault(image.id, []).append(contour)

    stored: list[Embeddings] = []
    for image_id, group in by_image.items():
        image = images[group[0].id]
        regions = [
            EmbedRegion(
                region_id=contour.id,
                mask=Contour(x=contour.x, y=contour.y, confidence=contour.confidence_score)
                .to_binary_mask_model(image.height, image.width),
            )
            for contour in group
        ]
        request = EmbedRequest(
            image_url=image.file_path, user_id=_SYSTEM_USER,
            model_registry_key=model_registry_key, image_kinds=[], regions=regions,
        )
        for vector in client.request_embeddings(request):
            if vector.kind not in REGION_KINDS or vector.region_id is None:
                continue
            stored.append(upsert_embedding(
                session, contour_id=vector.region_id, kind=vector.kind,
                model_id=vector.model_id, vector=vector.vector,
            ))
    return stored


def _contour_images(session: Session, contours: Sequence[Contours]) -> dict[int, Images]:
    """Map each contour id to its source image (via its mask), in one query."""
    mask_ids = {c.mask_id for c in contours}
    if not mask_ids:
        return {}
    rows = (
        session.query(Masks.id, Images)
        .join(Images, Images.id == Masks.image_id)
        .filter(Masks.id.in_(mask_ids))
        .all()
    )
    mask_to_image = {mask_id: image for mask_id, image in rows}
    return {c.id: mask_to_image[c.mask_id] for c in contours if c.mask_id in mask_to_image}


# --------------------------------------------------------------------------- #
# On-write hooks (opt-in) -- enqueue background work, never block the write
# --------------------------------------------------------------------------- #
@celery_app.task(name="embeddings.embed_image")
def embed_image_task(image_id: int) -> None:
    from app.database import get_context_session

    with get_context_session() as session:
        image = session.get(Images, image_id)
        if image is None:
            return
        embed_image(session, image)
        session.commit()


@celery_app.task(name="embeddings.embed_contours")
def embed_contours_task(contour_ids: list[int]) -> None:
    from app.database import get_context_session

    with get_context_session() as session:
        contours = session.query(Contours).filter(Contours.id.in_(contour_ids)).all()
        embed_contours(session, contours)
        session.commit()


def enqueue_embed_image(image_id: int) -> None:
    """Best-effort enqueue of image embedding. No-op unless ``EMBEDDING_LIFECYCLE_ENABLED``.

    Wrapped so a broker hiccup can never break the image write that triggered it.
    """
    if not EMBEDDING_LIFECYCLE_ENABLED:
        return
    try:
        embed_image_task.delay(image_id)
    except Exception:
        logger.exception("Failed to enqueue image embedding for image %s", image_id)


def enqueue_embed_contours(contour_ids: Iterable[int]) -> None:
    """Best-effort enqueue of contour embedding. No-op unless ``EMBEDDING_LIFECYCLE_ENABLED``."""
    if not EMBEDDING_LIFECYCLE_ENABLED:
        return
    ids = [int(cid) for cid in contour_ids]
    if not ids:
        return
    try:
        embed_contours_task.delay(ids)
    except Exception:
        logger.exception("Failed to enqueue contour embedding for %s", ids)


# --------------------------------------------------------------------------- #
# Backfill
# --------------------------------------------------------------------------- #
def _chunks(items: list, size: int):
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


def _images_missing_kind(session: Session, dataset_id: int | None) -> list[Images]:
    q = session.query(Images)
    if dataset_id is not None:
        q = q.filter(Images.dataset_id == dataset_id)
    have = {
        row[0] for row in
        session.query(Embeddings.image_id)
        .filter(Embeddings.image_id.isnot(None),
                Embeddings.kind.in_(tuple(IMAGE_KINDS)))
        .all()
    }
    return [image for image in q.all() if image.id not in have]


def _contours_missing_kind(session: Session, dataset_id: int | None) -> list[Contours]:
    q = (
        session.query(Contours)
        .join(Masks, Masks.id == Contours.mask_id)
        .filter(Contours.temporary.is_(False))
    )
    if dataset_id is not None:
        q = q.join(Images, Images.id == Masks.image_id).filter(Images.dataset_id == dataset_id)
    have = {
        row[0] for row in
        session.query(Embeddings.contour_id)
        .filter(Embeddings.contour_id.isnot(None),
                Embeddings.kind.in_(tuple(REGION_KINDS)))
        .all()
    }
    return [contour for contour in q.all() if contour.id not in have]


def backfill_embeddings(
    session: Session,
    *,
    dataset_id: int | None = None,
    model_registry_key: str = EMBEDDING_MODEL_KEY,
    client: EmbedClient | None = None,
    batch_size: int = 32,
) -> dict[str, int]:
    """Embed images/contours in scope that have no embedding of a required kind yet.

    Idempotent: subjects already embedded are skipped, so re-running only fills gaps. Restrict
    to one dataset with ``dataset_id``. Returns ``{"images": n, "contours": m}`` counts. Does
    not commit -- the caller (the script) controls the transaction.
    """
    client = client or EmbeddingService()
    counts = {"images": 0, "contours": 0}

    if _required_image_kinds():
        for image in _images_missing_kind(session, dataset_id):
            embed_image(session, image, model_registry_key=model_registry_key, client=client)
            counts["images"] += 1
        session.flush()

    if _region_kinds_needed():
        missing = _contours_missing_kind(session, dataset_id)
        for chunk in _chunks(missing, batch_size):
            embed_contours(session, chunk, model_registry_key=model_registry_key, client=client)
            counts["contours"] += len(chunk)
        session.flush()

    return counts
