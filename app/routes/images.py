import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.images import Images, Scans
from app.services.database_access import delete_image_from_disk_and_db, parse_log_file
from app.services.database_access import save_image_to_disk_and_db, load_image_as_base64_from_disk

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
                      scan_type: str = "CT",
                      description: str = "Scan description",
                      number_of_slices: int = 0,
                      #meta_data: dict = None,
                      db: Session = Depends(get_session)):
    """Upload a scan file"""
    # First create a new scan entry in the database
    # Then save each image file to disk and the database
    # and associate them with the scan entry
    new_scan = Scans(
        dataset_id=dataset_id,
        name=name,
        type=scan_type,
        description=description,
        number_of_slices=number_of_slices,
        #meta_data=meta_data
    )
    db.commit()
    try:
        image_ids = []
        for i, file in enumerate(files):
            image_id = await save_image_to_disk_and_db(file, dataset_id, new_scan.id, index_in_scan=i)
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
        number_of_slices=len(files),
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

            scan = db.query(Scans).filter(Scans.id == scan_id).first()
            db.delete(scan)
            db.commit()
        return {"success": True, "message": f"Deleted scan {scan_id} and its associated images."}
    except Exception as e:
        logger.error(f"Delete scan error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
