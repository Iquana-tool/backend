from . import router, logger
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL


@router.post("/start_training")
async def start_training():
    pass


async def send_start_training_request():
    url = f"{BASE_URL}/start_training"

