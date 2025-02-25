from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.segmentation import router as segmentation_router
from app.routes.images import router as image_router


def create_app():
    app = FastAPI(
        title="Coral Segmentation API",
        description="FastAPI backend for interactive coral segmentation",
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the router
    app.include_router(segmentation_router, prefix="/api")
    app.include_router(image_router, prefix="/api")

    return app