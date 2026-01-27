import os
from logging import getLogger

import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.params import Depends
from schemas.training import SemanticTrainingRequest
from schemas.user import User
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse, StreamingResponse

from app.database import get_session
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.services.ai_services.semantic_segmentation import segment_image_with_semantic_model
from app.services.auth import get_current_user
from app.services.util import get_mask_path_from_image_path
from paths import SEMANTIC_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.post("/model={model_registry_key}&image={image_id}")
async def send_inference_job(model_registry_key: str,
                             image_id: int,
                             user: User = Depends(get_current_user),
                             db: Session = Depends(get_session)):
    """ Sends an inference job to the semantic segmentation service.

    Args:
        user: User dependency.
        model_registry_key (str): The registry key of the model you want to use.
        image_id (int): ID of the image to segment.
        db (Session): Database session dependency.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    try:
        contour_hierarchy = await segment_image_with_semantic_model(model_registry_key, image_id, db)
        return {
            "success": True,
            "message": "Inference done.",
            "contour_hierarchy": contour_hierarchy.model_dump(),
        }
    except Exception as e:
        logger.error(f"Batch prompted_segmentation failed: {e}")
        raise e


@router.get("/get_models/dataset={dataset_id}")
async def get_models(dataset_id: int,
                     user: User = Depends(get_current_user)):
    """Retrieve all available models for this dataset."""
    url = f"{BASE_URL}/models/get_models/type={'all'}&available={True}&dataset_id={dataset_id}"
    logger.info(url)
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())


@router.delete("/delete_model/{model_id}")
async def delete_model(model_id: int,
                       user: User = Depends(get_current_user)):
    """ Delete a model based on its id. """
    logger.debug(f"Deleting model with id: {model_id}")
    url = f"{BASE_URL}/models/delete_model/{model_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.delete(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())


@router.get("/get_training_status/{model_registry_key}")
async def get_training_status(
        model_registry_key: str,
        user: User = Depends(get_current_user)
):
    """ Get the status of a training job by its ID. """
    url = f"{BASE_URL}/training/get_job_status/{model_registry_key}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


@router.get("/get_training_status_stream/{model_registry_key}")
async def get_training_status_stream(
        model_registry_key: str,
        user: User = Depends(get_current_user)
):
    """ Get live updates of the status of a training job by its ID. """
    url = f"{BASE_URL}/training/get_status_stream/{model_registry_key}"
    async with httpx.AsyncClient() as client:
        # Use `stream=True` to get a streaming response
        async with client.stream("GET", url) as response:
            # Forward the stream as-is
            return StreamingResponse(
                response.aiter_bytes(),  # or response.aiter_text() for text streams
                media_type=response.headers.get("content-type", "application/octet-stream"),
                headers=dict(response.headers)
            )


@router.get("/cancel_training_of_model/{model_id}")
async def cancel_training_of_model(
        model_id: str,
        user: User = Depends(get_current_user)
):
    """ Cancel a training job by its ID."""
    try:
        url = f"{BASE_URL}/training/cancel_job/{model_id}"
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        return {
            "success": False,
            "message": f"Could not cancel job {model_id}, probably due to computational overload."
        }


@router.post("/start_training")
async def start_training(
        request: SemanticTrainingRequest,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Start training a model for automatic prompted_segmentation.
    This endpoint prepares the request with necessary parameters and forwards it to the Automatic Segmentation Service.
    Args:
        request (TrainingRequest): The training request containing dataset_id and other parameters.
        db (Session): The database session for querying labels.
        user (User): The current authenticated user.

    Returns:
        JSONResponse: The response from the Automatic Segmentation Service.
    """
    # Get the request as a dict, to add values to it
    request_dict = request.model_dump()

    # Find out the number of classes present. + 1 for background
    # Caution: This will get all labels that the user created, even if they did not annotate them anywhere.
    num_classes = db.query(Labels).filter_by(dataset_id=request.dataset_id).count() + 1
    request_dict["num_classes"] = num_classes

    # Abstract this from the frontend, user should not be concerned with this.
    request_dict["in_channels"] = 3
    request_dict["batch_size"] = 32
    request_dict["lr"] = 0.0001
    result = await send_start_training_request(request_dict)
    return JSONResponse(result)


async def send_start_training_request(request: dict):
    """ Forwards TrainingRequest to Automatic Segmentation Service."""
    logger.info(f"Start training request: {request}")
    url = f"{BASE_URL}/training/start_training"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=request)
        if not resp.is_success:
            logger.error(f"Training request failed: {resp.text}")
        resp.raise_for_status()
        return resp.json()


@router.post("/upload_file_to_dataset")
async def proxy_upload_file(
        dataset_id: int,
        is_image: bool,
        filename: str = None,
        file: UploadFile = File(...),
        user: User = Depends(get_current_user)
):
    """
    Proxies a single image/mask file upload to the prompted_segmentation training backend.
    """
    logger.debug(f"Uploading file with name {filename} to dataset {dataset_id} as {'image' if is_image else 'mask'}")
    url = f"{BASE_URL}/data/upload_file_to_dataset"
    # httpx needs 'files' and 'data' for multipart forwards
    data = {
        "dataset_id": str(dataset_id),
        "is_image": str(is_image),  # send as 0/1 for bool
    }
    if filename:
        data["filename"] = filename

    files = {"file": (file.filename if not filename else filename, await file.read(), file.content_type)}

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, data=data, files=files)
        resp.raise_for_status()
        return resp.json()


@router.post("/upload_dataset/{dataset_id}")
async def proxy_upload_dataset(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Finds all finished mask/image pairs for this dataset, and uploads them one-by-one to the prompted_segmentation backend.
    """
    image_tuples = (
        db.query(Images.file_path)
        .join(Masks, Images.id == Masks.image_id)
        .filter(Masks.finished == True, Images.dataset_id == dataset_id)
        .all()
    )
    image_paths = [row[0] for row in image_tuples]
    mask_paths = []
    paired_image_paths = []
    for img_path in image_paths:
        mask_path = get_mask_path_from_image_path(img_path)
        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            continue
        paired_image_paths.append(img_path)
        mask_paths.append(mask_path)
    if not paired_image_paths or not mask_paths:
        raise HTTPException(status_code=400, detail="No valid image/mask pairs found.")

    url = f"{BASE_URL}/data/upload_file_to_dataset"
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for img_path in paired_image_paths:
            # --- Image upload ---
            with open(img_path, "rb") as img_file:
                files = {"file": (os.path.basename(img_path), img_file, f"image/{os.path.splitext(img_path)[1]}")}
                data = {
                    "dataset_id": str(dataset_id),
                    "is_image": "1",  # backend expects int-bool
                    "filename": os.path.splitext(os.path.basename(img_path))[0]
                }
                resp = await client.post(url, data=data, files=files)
                try:
                    resp.raise_for_status()
                except Exception as e:
                    results.append({"file": img_path, "type": "image", "status": "failed", "error": str(e)})
                    continue
                results.append({"file": img_path, "type": "image", "status": "ok"})

        for mask_path in mask_paths:
            # --- Mask upload ---
            with open(mask_path, "rb") as mask_file:
                files = {"file": (os.path.basename(mask_path), mask_file, f"image/{os.path.splitext(mask_path)[1]}")}
                data = {
                    "dataset_id": str(dataset_id),
                    "is_image": "0",  # backend expects int-bool
                    "filename": os.path.splitext(os.path.basename(mask_path))[0]
                }
                resp = await client.post(url, data=data, files=files)
                print(resp.text)
                try:
                    resp.raise_for_status()
                except Exception as e:
                    results.append({"file": mask_path, "type": "mask", "status": "failed", "error": str(e)})
                    continue
                results.append({"file": mask_path, "type": "mask", "status": "ok"})

    return {
        "success": all(r["status"] == "ok" for r in results),
        "num_uploaded_images": len(paired_image_paths),
        "num_uploaded_masks": len(mask_paths),
        "results": results,
    }
