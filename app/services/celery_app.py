from celery import Celery
from config import REDIS_URL

celery_app = Celery(
    "iquana_celery",
    broker=f"{REDIS_URL}/0",
    backend=f"{REDIS_URL}/1",
    # Modules the worker imports so their @celery_app.task functions register. The embedding
    # tasks stay dormant unless EMBEDDING_LIFECYCLE_ENABLED is set (nothing enqueues them);
    # the inference tasks are what the Batch Inference page hands its runs to.
    include=["app.services.embedding_lifecycle", "app.services.inference.tasks"],
)

#: Queue every backend task is published to and the backend worker consumes.
#
# This must NOT be Celery's default "celery" queue. The backend and the ai-service are two
# separate Celery apps sharing one Redis broker, and the ai-service worker consumes the
# default queue as a fallback alongside its own ai.training. Publishing here on the default
# queue makes it a race: if the ai-service worker picks the message up first it does not have
# these tasks registered, so it logs "Received unregistered task of type 'inference.run_next'"
# and *discards* it. Nothing retries, so the run stops dead with its job row still saying
# running -- which then also blocks every future run on that dataset.
#
# A dedicated queue removes the race entirely: the ai-service worker never sees these
# messages, whatever it is subscribed to.
BACKEND_QUEUE = "backend.jobs"

celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    task_default_queue=BACKEND_QUEUE,
)
