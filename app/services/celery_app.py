from celery import Celery
from config import REDIS_URL

celery_app = Celery(
    "iquana_celery",
    broker=f"{REDIS_URL}/0",
    backend=f"{REDIS_URL}/1",
    # Modules the worker imports so their @celery_app.task functions register. The embedding
    # tasks stay dormant unless EMBEDDING_LIFECYCLE_ENABLED is set (nothing enqueues them).
    include=["app.services.embedding_lifecycle"],
)

# Optional: Add common Celery configurations
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True
)
