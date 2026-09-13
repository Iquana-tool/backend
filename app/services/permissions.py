"""Permission enforcement for routes.

`require()` / `require_global()` are FastAPI dependencies that replace a bare
`Depends(get_current_user)` on a route and return the same user object, so
adopting them is a one-line change per endpoint.

Most routes are keyed by `contour_id` / `mask_id` / `image_id` / `label_id` rather
than by `dataset_id`, so `require()` takes the name of the id parameter to walk up
from. The walk is contour -> mask -> image -> dataset; labels join directly.
When the dataset id only becomes known after the body is parsed, call
`ensure_permission()` from inside the handler instead.
"""
from __future__ import annotations

from logging import getLogger
from typing import Iterable, Literal

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.auth import get_current_user

logger = getLogger(__name__)

#: Which id a route carries, and therefore how to find its dataset.
IdSource = Literal["dataset_id", "mask_id", "contour_id", "image_id", "label_id"]


# -- Dataset resolution ----------------------------------------------------

def dataset_id_for_mask(mask_id: int, db: Session) -> int | None:
    return (
        db.query(Images.dataset_id)
        .join(Masks, Masks.image_id == Images.id)
        .filter(Masks.id == mask_id)
        .scalar()
    )


def dataset_id_for_contour(contour_id: int, db: Session) -> int | None:
    return (
        db.query(Images.dataset_id)
        .join(Masks, Masks.image_id == Images.id)
        .join(Contours, Contours.mask_id == Masks.id)
        .filter(Contours.id == contour_id)
        .scalar()
    )


def dataset_id_for_image(image_id: int, db: Session) -> int | None:
    return db.query(Images.dataset_id).filter(Images.id == image_id).scalar()


def dataset_id_for_label(label_id: int, db: Session) -> int | None:
    return db.query(Labels.dataset_id).filter(Labels.id == label_id).scalar()


_RESOLVERS = {
    "mask_id": dataset_id_for_mask,
    "contour_id": dataset_id_for_contour,
    "image_id": dataset_id_for_image,
    "label_id": dataset_id_for_label,
}


def resolve_dataset_id(source: IdSource, entity_id: int, db: Session) -> int | None:
    """Walk from whatever id a route carries up to the owning dataset."""
    if source == "dataset_id":
        return db.query(Datasets.id).filter(Datasets.id == entity_id).scalar()
    resolver = _RESOLVERS.get(source)
    if resolver is None:
        raise ValueError(f"No dataset resolver for id source {source!r}.")
    return resolver(entity_id, db)


# -- Imperative checks -----------------------------------------------------

def ensure_permission(user: AuthenticatedUser, dataset_id: int, permission: Permission) -> None:
    """Raise 403 unless `user` may perform `permission` on `dataset_id`."""
    if not user.has_permission(dataset_id, permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing permission '{permission.value}' on dataset {dataset_id}.",
        )


def ensure_global_permission(user: AuthenticatedUser, permission: Permission) -> None:
    """Raise 403 unless `user` holds a platform-level permission."""
    if not user.has_global_permission(permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing permission '{permission.value}'.",
        )


def ensure_permission_on_datasets(user: AuthenticatedUser,
                                  dataset_ids: Iterable[int],
                                  permission: Permission) -> None:
    """Check one permission across several datasets (for batch endpoints)."""
    for dataset_id in set(dataset_ids):
        ensure_permission(user, dataset_id, permission)


def ensure_permission_for(user: AuthenticatedUser,
                          source: IdSource,
                          entity_id: int,
                          permission: Permission,
                          db: Session) -> int:
    """Resolve the dataset behind an entity id, check `permission`, return the id."""
    dataset_id = resolve_dataset_id(source, entity_id, db)
    if dataset_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No dataset found for {source} {entity_id}.")
    ensure_permission(user, dataset_id, permission)
    return dataset_id


# -- Route dependencies ----------------------------------------------------

def _read_id(request: Request, source: IdSource) -> int:
    """Pull the entity id out of the path or query string."""
    raw = request.path_params.get(source)
    if raw is None:
        raw = request.query_params.get(source)
    if raw is None:
        # A programming error rather than a client one: the route does not carry
        # the id its permission check was told to look for.
        logger.error("Route %s has no '%s' parameter for its permission check.",
                     request.url.path, source)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route is missing the '{source}' parameter needed for its permission check.",
        )
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"'{source}' must be an integer.")


def require(permission: Permission, source: IdSource = "dataset_id"):
    """Dependency enforcing `permission` on the dataset that `source` points at.

    Returns the authenticated user, so it is a drop-in replacement for
    `Depends(get_current_user)`::

        @router.delete("/{mask_id}")
        async def delete_mask(
                mask_id: int,
                user: AuthenticatedUser = Depends(require(Permission.MASK_DELETE, "mask_id")),
        ):
            ...
    """

    async def dependency(request: Request,
                         db: Session = Depends(get_session),
                         user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        entity_id = _read_id(request, source)
        dataset_id = resolve_dataset_id(source, entity_id, db)
        if dataset_id is None:
            # Do not distinguish "missing" from "not yours" to a caller who could
            # see neither; 404 is the honest answer for both.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No dataset found for {source} {entity_id}.")
        ensure_permission(user, dataset_id, permission)
        return user

    return dependency


def require_global(permission: Permission):
    """Dependency enforcing a platform-level permission (no dataset involved)."""

    async def dependency(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        ensure_global_permission(user, permission)
        return user

    return dependency
