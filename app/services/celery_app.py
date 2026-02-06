from celery import Celery
from config import REDIS_URL

celery_app = Celery(
    "iquana_main_api",
    broker=f"{REDIS_URL}/0",
    backend=f"{REDIS_URL}/1",
)

# Optional: Add common Celery configurations
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True
)
