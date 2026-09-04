import logging
import os
from logging import getLogger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.database import init_db
from app.routes.general.admin import router as admin_router
from app.routes.general.annotation_history import router as annotation_history_router
from app.routes.general.annotation_queue import router as annotation_queue_router
from app.routes.general.auth import router as auth_router
from app.routes.general.calibration import router as calibration_router
from app.routes.general.contours import router as contour_router
from app.routes.general.datasets import router as dataset_router
from app.routes.general.image_metadata import router as image_metadata_router
from app.routes.general.images import router as image_router
from app.routes.general.instance import router as instance_router
from app.routes.general.labels import router as label_router
from app.routes.general.masks import router as mask_router
from app.routes.general.members import invite_router, router as member_router
from app.routes.general.model_favorites import router as model_favorites_router
from app.routes.general.reviews import router as review_router
from app.routes.general.pixel_scale import router as scale_router
from app.routes.general.status import router as status_router
from app.routes.services.suggestion_router import router as suggestion_segmentation_router
from app.routes.services.label_space_router import router as label_space_router
from app.routes.services.prompted_router import router as prompted_segmentation_router
from app.routes.services.instance_seg_router import router as instance_segmentation_router
from app.routes.services.inference_router import router as batch_inference_router
from app.routes.services.cross_image_router import router as cross_image_router
from app.routes.websockets.image_annotation_session import router as image_annotation_session_router
from config import *

logger = getLogger(__name__)

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
        description="FastAPI backend for IQUANA — Intelligent QUantification, ANnotation and Analysis",
        version="0.1.0",
        # Keep empty for local runs; set FASTAPI_ROOT_PATH behind reverse proxy.
        root_path=root_path,
    )

    # Compress responses.
    #
    # The API's large payloads are all JSON, and JSON of this shape compresses roughly
    # tenfold: the per-contour quantification export measured 37 MB raw against 3.7 MB
    # gzipped, which is the difference between a page that takes a minute to fill and one
    # that takes a moment. Image payloads benefit for a different reason — images are
    # returned base64-encoded inside JSON, and gzip essentially recovers the third that
    # base64 adds.
    #
    # `compresslevel` is deliberately 6 rather than Starlette's default of 9. On that same
    # 37 MB export, level 9 spends 4.1s of server CPU to save 12% over level 6's 0.6s —
    # which would simply move the delay from the network to the server. 6 is the usual
    # gzip default and the right end of that curve.
    #
    # Starlette's own `exclude_content_types` default already skips what must not be
    # touched: already-compressed formats (the dataset ZIP export), images served as
    # binary, and `text/event-stream`, which would otherwise be buffered and break
    # streaming. Anything above `thread_minimum_size` is compressed in a worker thread, so
    # a large export does not block the event loop.
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

    # Configure CORS. Added after GZip so it ends up the outer middleware — Starlette
    # builds the stack in reverse order of registration — which keeps the CORS headers on
    # the response whether or not the body below it was compressed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root endpoint
    @app.get("/")
    async def root():
        return {"message": "This is the API for DFKI"}

    # Include the routers
    # General Routers
    app.include_router(status_router)
    app.include_router(instance_router)
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(dataset_router)
    # Shares the /datasets prefix with the dataset router; the paths do not overlap.
    app.include_router(member_router)
    app.include_router(invite_router)
    app.include_router(review_router)
    app.include_router(annotation_queue_router)
    app.include_router(image_router)
    app.include_router(image_metadata_router)
    app.include_router(image_annotation_session_router)
    app.include_router(mask_router)
    app.include_router(contour_router)
    app.include_router(annotation_history_router)
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
    # Dataset-wide inference: orchestrates the services above over every image via Celery.
    app.include_router(batch_inference_router)

    return app
