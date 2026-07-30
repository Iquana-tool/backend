"""Per-user, per-task favorite (default) models.

A favorite is the model preselected for a task in the annotation page. The
frontend stars a model in the Model Zoo; a model that serves several tasks is
favorited for each of its tasks (the frontend issues one PUT per task).
"""
from logging import getLogger

from fastapi import APIRouter, Depends
from iquana_toolbox.schemas.user import User
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.model_favorites import UserModelFavorites
from app.services.auth import get_current_user

logger = getLogger(__name__)
router = APIRouter(prefix="/me", tags=["model favorites"])


class SetFavoriteBody(BaseModel):
    model_registry_key: str = Field(..., description="Registry key of the model to favorite for this task.")


@router.get("/model-favorites")
async def get_my_favorites(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Return the current user's favorite model per task as ``{task: registry_key}``."""
    rows = db.query(UserModelFavorites).filter_by(username=user.username).all()
    return {"success": True, "result": {row.task: row.model_registry_key for row in rows}}


@router.put("/model-favorites/{task}")
async def set_my_favorite(
    task: str,
    body: SetFavoriteBody,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Set (upsert) the current user's favorite model for ``task``."""
    row = db.query(UserModelFavorites).filter_by(username=user.username, task=task).first()
    if row is None:
        db.add(UserModelFavorites(
            username=user.username, task=task, model_registry_key=body.model_registry_key
        ))
    else:
        row.model_registry_key = body.model_registry_key
    db.commit()
    return {"success": True, "result": {"task": task, "model_registry_key": body.model_registry_key}}


@router.delete("/model-favorites/{task}")
async def clear_my_favorite(
    task: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Clear the current user's favorite model for ``task``."""
    db.query(UserModelFavorites).filter_by(username=user.username, task=task).delete()
    db.commit()
    return {"success": True}
