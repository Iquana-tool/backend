from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from logging import getLogger
import httpx

from app.schemas.user import User
from app.services.auth import get_current_user
from paths import SEMANTIC_SEGMENTATION_BACKEND_URL as BASE_URL


logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.get("/health")
async def get_health(user: User = Depends(get_current_user)):
    """Check the health of the automatic prompted_segmentation service."""
    url = f"{BASE_URL}/health"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())
