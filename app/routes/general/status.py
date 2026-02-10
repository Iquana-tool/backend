from fastapi import APIRouter, Depends
from redis.asyncio import Redis

from app.services.ai_services.completion_segmentation import CompletionService
from app.services.ai_services.prompted_segmentation import PromptedSegmentationService
from app.services.ai_services.semantic_segmentation import SemanticSegmentationService
from app.services.redis import get_redis


router = APIRouter(prefix="/status")


@router.get("/")
async def status(redis: Redis = Depends(get_redis)):
    prompted_status = await PromptedSegmentationService().check_backend()
    semantic_status = await SemanticSegmentationService().check_backend()
    completed_status = await CompletionService().check_backend()
    return {
        "success": True,
        "message": "Successfully retrieved stati",
        "result": {
            "prompted_status": prompted_status,
            "semantic_status": semantic_status,
            "completed_status": completed_status,
            "redis_status": "ok" if redis.ping() else "error",
        }
    }
