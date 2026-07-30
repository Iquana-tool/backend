"""Precomputed feature embeddings for images and contours.

This is the storage layer behind cross-image exemplar retrieval (find the exemplar
most similar to a target before handing it to a cross-image concept segmenter). A row
holds one dense feature vector describing *one subject*:

  * an **image** (a whole-image descriptor, e.g. a DINOv3 ``CLS`` token), or
  * a **contour** (a masked-region descriptor, e.g. the mean of an object's foreground
    patch features -- the "exemplar" unit).

Exactly one of ``image_id`` / ``contour_id`` is set per row (the CHECK enforces it),
and both carry ``ON DELETE CASCADE`` so an embedding dies with the image/contour it
describes -- the same cascade rigor the rest of the schema relies on.

Portability
-----------
Production runs on PostgreSQL with the `pgvector <https://github.com/pgvector/pgvector>`_
extension, so the ``vector`` column is a real ``vector(dim)`` and nearest-neighbour
search runs in-database via the ``<=>`` cosine operator over an HNSW index. Dev and the
whole test suite run on SQLite, which has no vector type: there the column degrades to a
JSON-encoded list of floats and :func:`search_similar` falls back to brute-force cosine
in NumPy. Both paths return the same Python types, so callers never branch on the backend.

Versioning
----------
Every row records the ``model_id`` that produced the vector (e.g.
``facebook/dinov3-vitb16-pretrain-lvd1689m``). Embeddings from different backbones are not
comparable, so retrieval always filters by a single ``model_id``; bumping the embedding
model simply writes a new generation of rows alongside the old, and the stale generation
can be dropped once nothing reads it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import getLogger
from typing import Iterable, Sequence

import numpy as np
from sqlalchemy import (
    CheckConstraint,
    Column,
    DDL,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    event,
    select,
    text,
)
from sqlalchemy.types import TEXT, TypeDecorator

from app.database import database

logger = getLogger(__name__)

# DINOv3 ViT-B/16 hidden size. The pgvector column is fixed-width so it can be HNSW
# indexed; a backbone with a different hidden size needs a schema change (a new column
# / table), which is why the width lives in one named constant.
EMBEDDING_DIM = 768

# Subject discriminators used by the data-access helpers (the table itself discriminates
# by which FK column is populated).
SUBJECT_IMAGE = "image"
SUBJECT_CONTOUR = "contour"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VectorType(TypeDecorator):
    """A fixed-dimension float vector column, portable across PostgreSQL and SQLite.

    On PostgreSQL this is a real pgvector ``vector(dim)`` -- so cosine search can run
    in-database (``<=>``) and be HNSW-indexed. On every other backend (the SQLite used in
    dev and tests) it degrades to a JSON-encoded ``list[float]`` in a ``TEXT`` column, and
    similarity is computed in Python. The value handed to/from the ORM is always a
    ``list[float]`` regardless of backend.

    ``pgvector`` is imported lazily inside :meth:`load_dialect_impl` so the package is only
    required where PostgreSQL is actually used; SQLite runs never import it.
    """

    impl = TEXT
    cache_ok = True

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        seq = value.tolist() if hasattr(value, "tolist") else list(value)
        seq = [float(x) for x in seq]
        if dialect.name == "postgresql":
            # Hand the list straight to pgvector's own bind processor (which runs after
            # this one via the resolved dialect impl) -- it renders the vector literal.
            return seq
        return json.dumps(seq)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            # pgvector already decoded the DB value to a NumPy array; normalize to list.
            return [float(x) for x in value]
        return [float(x) for x in json.loads(value)]

    class comparator_factory(TypeDecorator.Comparator):
        def cosine_distance(self, other):
            """PostgreSQL-only ``<=>`` cosine distance. Never emitted on SQLite."""
            return self.op("<=>", return_type=Float)(other)


class Embeddings(database):
    """One dense feature vector describing a single image or contour."""

    __tablename__ = "embeddings"

    id: int = Column(Integer, primary_key=True, autoincrement=True)

    # Exactly one subject FK is set (see the CHECK below). Both cascade so the embedding
    # is removed with the image/contour it describes.
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=True)
    contour_id = Column(Integer, ForeignKey("contours.id", ondelete="CASCADE"), nullable=True)

    # What the vector represents, e.g. "image_cls" or "region_mean". Different kinds live in
    # the same column (all EMBEDDING_DIM-wide); retrieval filters by (kind, model_id).
    kind = Column(String(32), nullable=False)
    # The backbone that produced the vector, e.g. "facebook/dinov3-vitb16-pretrain-lvd1689m".
    model_id = Column(String(128), nullable=False)
    dim = Column(Integer, nullable=False)
    vector = Column(VectorType(EMBEDDING_DIM), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        # A row describes an image XOR a contour -- never both, never neither.
        CheckConstraint(
            "(image_id IS NOT NULL) <> (contour_id IS NOT NULL)",
            name="ck_embeddings_one_subject",
        ),
        # One vector per (subject, kind, model). Split into two partial unique indexes
        # because a single UNIQUE over both FK columns would not deduplicate: the unused
        # FK is NULL, and NULLs compare distinct in a UNIQUE constraint. Partial indexes
        # are supported by both PostgreSQL and SQLite (>= 3.8).
        Index(
            "uq_embeddings_image_kind_model",
            "image_id", "kind", "model_id",
            unique=True,
            sqlite_where=text("image_id IS NOT NULL"),
            postgresql_where=text("image_id IS NOT NULL"),
        ),
        Index(
            "uq_embeddings_contour_kind_model",
            "contour_id", "kind", "model_id",
            unique=True,
            sqlite_where=text("contour_id IS NOT NULL"),
            postgresql_where=text("contour_id IS NOT NULL"),
        ),
    )


# --- PostgreSQL-only DDL: the pgvector extension and the HNSW index ---------- #
# ``create_all`` runs for every dialect, so these are guarded to PostgreSQL and are no-ops
# on SQLite. The extension must exist *before* a table with a vector column is created; the
# cosine HNSW index is added after. Both are IF-(NOT-)EXISTS so re-running create_all is safe.
event.listen(
    Embeddings.__table__,
    "before_create",
    DDL("CREATE EXTENSION IF NOT EXISTS vector").execute_if(dialect="postgresql"),
)
event.listen(
    Embeddings.__table__,
    "after_create",
    DDL(
        "CREATE INDEX IF NOT EXISTS ix_embeddings_vector_hnsw "
        "ON embeddings USING hnsw (vector vector_cosine_ops)"
    ).execute_if(dialect="postgresql"),
)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmbeddingHit:
    """One nearest-neighbour result: the matched subject and its cosine distance (0 = identical)."""

    subject_id: int
    distance: float
    embedding_id: int


def _subject_column(subject_type: str):
    if subject_type == SUBJECT_IMAGE:
        return Embeddings.image_id
    if subject_type == SUBJECT_CONTOUR:
        return Embeddings.contour_id
    raise ValueError(
        f"unknown subject_type {subject_type!r} (expected {SUBJECT_IMAGE!r} or {SUBJECT_CONTOUR!r})"
    )


def upsert_embedding(
    session,
    *,
    kind: str,
    model_id: str,
    vector: Sequence[float] | np.ndarray,
    image_id: int | None = None,
    contour_id: int | None = None,
) -> Embeddings:
    """Insert or replace the embedding for one (subject, kind, model_id).

    Exactly one of ``image_id`` / ``contour_id`` must be given. Idempotent: re-embedding a
    subject overwrites its vector in place (matching the partial unique indexes), so the
    lifecycle layer can safely re-run on updates without accumulating duplicates. Does not
    commit -- the caller controls the transaction.
    """
    if (image_id is None) == (contour_id is None):
        raise ValueError("pass exactly one of image_id or contour_id")

    vec = [float(x) for x in (vector.tolist() if hasattr(vector, "tolist") else vector)]
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(f"expected a {EMBEDDING_DIM}-d vector, got {len(vec)}")

    subject_col = Embeddings.image_id if image_id is not None else Embeddings.contour_id
    subject_val = image_id if image_id is not None else contour_id
    existing = (
        session.query(Embeddings)
        .filter(subject_col == subject_val,
                Embeddings.kind == kind,
                Embeddings.model_id == model_id)
        .first()
    )
    if existing is not None:
        existing.vector = vec
        existing.dim = len(vec)
        existing.created_at = _utcnow()
        session.flush()
        return existing

    row = Embeddings(
        image_id=image_id,
        contour_id=contour_id,
        kind=kind,
        model_id=model_id,
        dim=len(vec),
        vector=vec,
    )
    session.add(row)
    session.flush()
    return row


def get_embedding_vector(
    session,
    *,
    kind: str,
    model_id: str,
    image_id: int | None = None,
    contour_id: int | None = None,
) -> list[float] | None:
    """Return the stored vector for one (subject, kind, model_id), or ``None`` if absent.

    Exactly one of ``image_id`` / ``contour_id`` must be given. Used by retrieval to fetch a
    query vector (e.g. the target image's ``image_cls``) that was precomputed earlier.
    """
    if (image_id is None) == (contour_id is None):
        raise ValueError("pass exactly one of image_id or contour_id")
    subject_col = Embeddings.image_id if image_id is not None else Embeddings.contour_id
    subject_val = image_id if image_id is not None else contour_id
    row = (
        session.query(Embeddings.vector)
        .filter(subject_col == subject_val,
                Embeddings.kind == kind,
                Embeddings.model_id == model_id)
        .first()
    )
    return list(row[0]) if row is not None else None


def _dataset_scoped_ids(session, subject_type: str, dataset_id: int) -> set[int]:
    """Subject ids (image or contour) belonging to ``dataset_id``.

    Used to constrain a search to one dataset's bank. Images join straight to the dataset;
    contours reach it through their mask's image.
    """
    from app.database.images import Images

    if subject_type == SUBJECT_IMAGE:
        rows = session.query(Images.id).filter(Images.dataset_id == dataset_id).all()
        return {r[0] for r in rows}

    from app.database.contours import Contours
    from app.database.masks import Masks

    rows = (
        session.query(Contours.id)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .all()
    )
    return {r[0] for r in rows}


def search_similar(
    session,
    query_vector: Sequence[float] | np.ndarray,
    *,
    subject_type: str,
    kind: str,
    model_id: str,
    dataset_id: int | None = None,
    restrict_ids: Iterable[int] | None = None,
    exclude_ids: Iterable[int] | None = None,
    top_k: int = 10,
) -> list[EmbeddingHit]:
    """Nearest embeddings to ``query_vector`` by cosine distance, most-similar first.

    Filters to one ``subject_type`` / ``kind`` / ``model_id`` (embeddings from different
    backbones are not comparable). Optionally restricts to a dataset's bank (``dataset_id``)
    and/or an explicit candidate set (``restrict_ids`` -- e.g. the contours of one concept),
    and excludes specific subject ids (e.g. the query's own contour). On PostgreSQL the
    ordering runs in-database via pgvector's ``<=>``; on SQLite the candidate vectors are
    pulled and scored with NumPy. Returns at most ``top_k`` hits.
    """
    subject_col = _subject_column(subject_type)
    exclude = set(exclude_ids or ())

    # ``allowed`` is the intersection of every positive filter (dataset scope, explicit
    # restriction); ``None`` means "no positive filter -> all subjects of this kind".
    allowed: set[int] | None = None
    if dataset_id is not None:
        allowed = _dataset_scoped_ids(session, subject_type, dataset_id)
    if restrict_ids is not None:
        restrict = {int(x) for x in restrict_ids}
        allowed = restrict if allowed is None else (allowed & restrict)
    if allowed is not None and not allowed:
        return []

    is_postgres = session.bind is not None and session.bind.dialect.name == "postgresql"

    if is_postgres:
        distance = Embeddings.vector.cosine_distance(list(query_vector)).label("distance")
        stmt = (
            select(Embeddings.id, subject_col.label("subject_id"), distance)
            .where(subject_col.isnot(None),
                   Embeddings.kind == kind,
                   Embeddings.model_id == model_id)
        )
        if allowed is not None:
            stmt = stmt.where(subject_col.in_(allowed))
        if exclude:
            stmt = stmt.where(subject_col.notin_(exclude))
        stmt = stmt.order_by(distance).limit(top_k)
        return [
            EmbeddingHit(subject_id=row.subject_id, distance=float(row.distance),
                         embedding_id=row.id)
            for row in session.execute(stmt)
        ]

    # SQLite / other: brute-force cosine in NumPy over the candidate rows.
    rows = (
        session.query(Embeddings.id, subject_col, Embeddings.vector)
        .filter(subject_col.isnot(None),
                Embeddings.kind == kind,
                Embeddings.model_id == model_id)
        .all()
    )
    candidates = [
        (emb_id, subj_id, vec)
        for emb_id, subj_id, vec in rows
        if subj_id not in exclude and (allowed is None or subj_id in allowed)
    ]
    if not candidates:
        return []

    matrix = np.asarray([c[2] for c in candidates], dtype=np.float64)
    distances = _cosine_distances(np.asarray(query_vector, dtype=np.float64), matrix)
    order = np.argsort(distances)[:top_k]
    return [
        EmbeddingHit(subject_id=candidates[i][1], distance=float(distances[i]),
                     embedding_id=candidates[i][0])
        for i in order
    ]


def _cosine_distances(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance (``1 - cosine similarity``) from ``query`` to each row of ``matrix``.

    Mirrors pgvector's ``<=>`` so the SQLite fallback and the PostgreSQL path agree. A
    zero-norm vector yields distance 1 (maximally dissimilar) rather than a divide-by-zero.
    """
    q_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * q_norm
    sims = np.divide(matrix @ query, denom, out=np.zeros(matrix.shape[0]), where=denom > 0)
    return 1.0 - sims
