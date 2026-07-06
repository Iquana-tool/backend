from fastapi import APIRouter, Depends
from redis.asyncio import Redis

from app.services.ai_services.instance_suggestion import SuggestionService
from app.services.ai_services.prompted_segmentation import PromptedSegmentationService
from app.services.ai_services.instance_segmentation import InstanceSegmentationService
from app.services.redis import get_redis


router = APIRouter(prefix="/status")


@router.get("/")
async def status(redis: Redis = Depends(get_redis)):
    prompted_status = await PromptedSegmentationService().check_backend()
    instance_status = await InstanceSegmentationService().check_backend()
    completed_status = await SuggestionService().check_backend()
    return {
        "success": True,
        "message": "Successfully retrieved stati",
        "result": {
            "prompted_status": prompted_status,
            "semantic_status": instance_status,
            "completed_status": completed_status,
            "redis_status": "ok" if redis.ping() else "error",
        }
    }
