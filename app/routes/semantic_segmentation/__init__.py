from fastapi import APIRouter
from fastapi.responses import JSONResponse
from logging import getLogger
import httpx
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL


logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.get("/health")
async def get_health():
    """Check the health of the automatic segmentation service."""
    url = f"{BASE_URL}/health"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())
