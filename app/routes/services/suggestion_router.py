from fastapi import APIRouter, Depends
from iquana_toolbox.schemas.user import User

from app.services.auth import get_current_user
from app.services.model_registry import list_available_models

router = APIRouter(prefix="/suggestion_segmentation", tags=["Suggestion Segmentation"])


@router.get("/models")
async def get_available_models(user: User = Depends(get_current_user)):
    """Retrieve available instance-suggestion models directly from MLflow."""
    return list_available_models("instance-suggestion")


# The POST /run endpoint was removed: nothing called it, and its request body
# carried a raw filesystem path (`image_url`) handed straight to cv2.imread, which
# made it unauthorizable — there is no dataset to resolve from a path — and an
# arbitrary-file-read on the shared volume. Suggestions run through the
# annotation-session WebSocket, which resolves the path server-side from the image
# id and holds its own SuggestionService instance.
