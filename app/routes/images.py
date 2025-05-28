import logging
import os.path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Literal
from app.database import get_session
from app.database.images import Images, Scans
from app.database.datasets import Datasets
from app.services.database_access import delete_image_from_disk_and_db, parse_log_file
from app.services.database_access import save_image_to_disk_and_db, load_image_as_base64_from_disk
from app.services.util import extract_numbers


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload_image")
async def upload_image(dataset_id: int, file: UploadFile = File(...), db: Session = Depends(get_session)):
    """Upload an image file"""
    try:
        image_id = await save_image_to_disk_and_db(file, dataset_id)
        if image_id is None:
            raise HTTPException(status_code=400, detail="Invalid file or upload failed")

        return {
            "success": True,
            "image_id": image_id,
            "message": f"Successfully uploaded image. Assigned id {image_id}"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_images")
async def upload_images(dataset_id: int, files: list[UploadFile] = File(...), db: Session = Depends(get_session)):
    """Upload multiple image files"""
    try:
        image_ids = []
        for file in files:
            image_id = (await upload_image(dataset_id, file, db))["image_id"]
            image_ids.append(image_id)

        return {
            "success": True,
            "image_ids": image_ids,
            "message": f"Successfully uploaded {len(files)} images. Assigned ids {image_ids}"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_image/{image_id}")
async def delete_image(image_id: int):
    try:
        delete_image_from_disk_and_db(image_id)
        return {"success": True,
                "message": f"Deleted image {image_id}."}
    except Exception as e:
        logger.error(f"Delete image error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_images/{dataset_id}")
def list_images(dataset_id: int, db: Session = Depends(get_session)):
    """List all uploaded image ids"""
    try:
        images = db.query(Images).filter_by(dataset_id=dataset_id).all()
        return {
            "success": True,
            "images": images
        }
    except Exception as e:
        logger.error(f"List images error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_image/{image_id}", response_model=dict[int, str])
async def get_image(image_id: int):
    """Get images via ids.

    Args:
        image_id: Image ID to retrieve.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    try:
        response = {}
        image = load_image_as_base64_from_disk(image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        response[image_id] = image
        return response
    except Exception as e:
        logger.error(f"Get image error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_scan")
async def upload_scan(dataset_id: int,
                      files: list[UploadFile] = File(...),
                      name: str = "Scan",
                      scan_type: Literal["CT"] = "CT",
                      description: str = "Scan description",
                      #meta_data: dict = None,
                      db: Session = Depends(get_session)):
    """Upload a scan file.
    This endpoint allows uploading multiple image files that belong to a scan.
    It creates a new scan entry in the database and associates the images with it.
    Args:
        dataset_id: ID of the dataset to which the scan belongs.
        files: List of image files to upload. The filenames must include the slice index or number. This number must
        be present as the last number in the filename to correctly associate the images with the scan. For example:
        "scan_slice_1.png", "scan_slice_2.png", etc. "Slice_1_scan_19.png" will be associated with index 19.
        name: Name of the scan.
        scan_type: Type of scan (e.g., "CT", "MRI"). Optional.
        description: Description of the scan. Optional.
        meta_data: Additional metadata about the scan. Optional.
    Returns:
        A success message with the IDs of the uploaded images.
        """
    # First create a new scan entry in the database
    # Then save each image file to disk and the database
    # and associate them with the scan entry
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    new_scan = Scans(
        dataset_id=dataset_id,
        name=name.strip(),
        type=scan_type.strip(),
        description=description.strip(),
        folder_path=os.path.join(dataset.folder_path, name),
        number_of_slices=len(files),
        #meta_data=meta_data
    )
    db.commit()
    try:
        image_ids = []
        for i, file in enumerate(files):
            # Extract numbers from the filename. The last number will be used as the index in the scan.
            numbers_in_filename = extract_numbers(file.filename)
            # If no numbers are found, use the index as a fallback
            index_in_scan = i if not numbers_in_filename else numbers_in_filename[-1]

            # Save the image to disk and the database
            image_id = await save_image_to_disk_and_db(file, dataset_id, new_scan.id, index_in_scan=index_in_scan)
            if image_id is None:
                raise HTTPException(status_code=400, detail="Invalid file or upload failed")
            image_ids.append(image_id)

        return {
            "success": True,
            "image_ids": image_ids,
            "message": f"Successfully uploaded {len(files)} images belonging to scan {new_scan.id}. "
                       f"Assigned image ids {image_ids}"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_scan_with_log_file")
async def upload_scan_with_log_file(
        dataset_id: int,
        files: list[UploadFile] = File(...),
        log_file: UploadFile = File(...),
        scan_type: str = "CT",
        description: str = "Scan description",
        db: Session = Depends(get_session)
):
    """Upload a scan file with logging."""
    # First create a new scan entry in the database
    # Then save each image file to disk and the database
    # and associate them with the scan entry
    log_data = parse_log_file(log_file.file.read())
    return await upload_scan(
        dataset_id=dataset_id,
        files=files,
        name=log_data["Filename Prefix"],
        scan_type=scan_type,
        description=description,
        #meta_data=log_data,
        db=db
    )


@router.delete("/delete_scan/{scan_id}")
async def delete_scan(scan_id: int):
    """Delete a scan and all its associated images."""
    try:
        # Get the scan and its associated images
        with get_session() as db:
            images = db.query(Images).filter(Images.scan_id == scan_id).first()
            if not images:
                raise HTTPException(status_code=404, detail="Scan not found")

            # Delete the scan and its associated images
            for image in images:
                delete_image_from_disk_and_db(image.id)

            scan = db.query(Scans).join(Datasets).filter(Scans.id == scan_id).first()
            shutil.rmtree(os.path.joi)
            db.delete(scan)
            db.commit()
        return {"success": True, "message": f"Deleted scan {scan_id} and its associated images."}
    except Exception as e:
        logger.error(f"Delete scan error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
