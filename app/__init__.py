import logging
import os
from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import get_context_session, init_db
from app.routes.general.admin import router as admin_router
from app.routes.general.annotation_queue import router as annotation_queue_router
from app.routes.general.auth import router as auth_router
from app.routes.general.calibration import router as calibration_router
from app.routes.general.contours import router as contour_router
from app.routes.general.datasets import router as dataset_router
from app.routes.general.images import router as image_router
from app.routes.general.labels import router as label_router
from app.routes.general.masks import router as mask_router
from app.routes.general.members import invite_router, router as member_router
from app.routes.general.model_favorites import router as model_favorites_router
from app.routes.general.reviews import router as review_router
from app.routes.general.pixel_scale import router as scale_router
from app.routes.general.status import router as status_router
from app.routes.general.telemetry import router as telemetry_router
from app.routes.services.suggestion_router import router as suggestion_segmentation_router
from app.routes.services.label_space_router import router as label_space_router
from app.routes.services.prompted_router import router as prompted_segmentation_router
from app.routes.services.instance_seg_router import router as instance_segmentation_router
from app.routes.services.cross_image_router import router as cross_image_router
from app.routes.websockets.image_annotation_session import router as image_annotation_session_router
from app.services.telemetry.config import get_config as get_telemetry_config
from app.services.telemetry.middleware import TelemetryMiddleware
from app.services.telemetry.recorder import recorder as telemetry_recorder
from config import *

logger = getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Application startup/shutdown.

    Currently only telemetry: the recorder's queue is drained on the way out so a
    restart during a study does not lose the last few seconds of a participant's
    session. `stop()` is a no-op when the recorder never started.
    """
    yield
    telemetry_recorder.stop()


def create_app():
    logger.setLevel(logging.DEBUG)
    logger.debug("Creating FastAPI application")
    # Load environment variables
    load_dotenv()
    
    # Get allowed origins from environment variable
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    logger.debug(f"Allowed origins: {allowed_origins}")

    # Create all necessary directories
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(THUMBNAILS_DIR, exist_ok=True)

    init_db()

    root_path = os.getenv("FASTAPI_ROOT_PATH", "").strip()

    app = FastAPI(
        title="IQUANA API",
        description="FastAPI backend for interactive coral prompted_segmentation",
        version="0.1.0",
        # Keep empty for local runs; set FASTAPI_ROOT_PATH behind reverse proxy.
        root_path=root_path,
        lifespan=_lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # User telemetry / study logging. Everything here is conditional on the
    # deployment-level USER_EVENTS_ENABLED lock: when it is off the middleware is
    # never installed and the /telemetry routes do not exist, so the feature has
    # no runtime cost and no attack surface on a deployment that opted out.
    if get_telemetry_config().enabled:
        logger.info("Telemetry enabled; installing capture middleware and routes.")
        # Resolve once here, with a session, so the stored runtime override is
        # read and cached at boot. Emit sites on the request path then hit the
        # cache instead of re-parsing the component list per event.
        with get_context_session() as telemetry_db:
            get_telemetry_config(telemetry_db)
        app.add_middleware(TelemetryMiddleware)
        app.include_router(telemetry_router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {"message": "This is the API for DFKI"}

    # Include the routers
    # General Routers
    app.include_router(status_router)
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(dataset_router)
    # Shares the /datasets prefix with the dataset router; the paths do not overlap.
    app.include_router(member_router)
    app.include_router(invite_router)
    app.include_router(review_router)
    app.include_router(annotation_queue_router)
    app.include_router(image_router)
    app.include_router(image_annotation_session_router)
    app.include_router(mask_router)
    app.include_router(contour_router)
    app.include_router(label_router)
    app.include_router(scale_router)
    # Generalises the /scale router above to every calibration kind; /scale stays
    # for the existing draw-a-line flow and its clients.
    app.include_router(calibration_router)
    app.include_router(model_favorites_router)

    # Services; Add your own service here!
    app.include_router(prompted_segmentation_router)
    app.include_router(suggestion_segmentation_router)
    app.include_router(instance_segmentation_router)
    app.include_router(label_space_router)
    app.include_router(cross_image_router)

    return app
