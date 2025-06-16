import logging
import os.path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Literal
from app.database import get_session
from app.database.images import Images, Scans
from app.database.datasets import Datasets
from app.services.database_access import delete_image_from_disk_and_db, parse_log_file, get_height_width_of_image
from app.services.database_access import save_image_to_disk_and_db, load_image_as_base64_from_disk
from app.services.util import extract_numbers
import zipfile
import io


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
    image_ids = []
    failed_files = []
    for file in files:
        try:
            image_id = (await upload_image(dataset_id, file, db))["image_id"]
            image_ids.append(image_id)
        except HTTPException as e:
            logger.error(f"Failed to upload {file.filename}: {str(e)}")
            failed_files.append(file.filename)

    # Prepare response message
    if failed_files:
        if len(image_ids) > 0:
            message = f"Successfully processed {len(image_ids)} files. Failed to upload {len(failed_files)} files: {', '.join(failed_files)}"
        else:
            message = f"Failed to upload all {len(failed_files)} files: {', '.join(failed_files)}"
    else:
        message = f"Successfully processed {len(image_ids)} images. Assigned ids {image_ids}"

    return {
        "success": True,
        "image_ids": image_ids,
        "uploaded_count": len(image_ids),
        "failed_count": len(failed_files),
        "failed_files": failed_files,
        "message": message,
        }


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
    if not dataset or dataset.dataset_type != "scan":
        raise HTTPException(status_code=404, detail="Dataset not found or is not a scan dataset")
    new_scan = Scans(
        dataset_id=dataset_id,
        name=name.strip(),
        type=scan_type.strip(),
        description=description.strip(),
        folder_path=os.path.join(dataset.folder_path, name),
        number_of_slices=len(files),
        #meta_data=meta_data
    )
    db.add(new_scan)
    db.commit()
    height, width = None, None
    try:
        image_ids = []
        for i, file in enumerate(files):
            # Extract numbers from the filename. The last number will be used as the index in the scan.
            numbers_in_filename = extract_numbers(file.filename)
            # If no numbers are found, use the index as a fallback
            index_in_scan = i if not numbers_in_filename else numbers_in_filename[-1]

            # Save the image to disk and the database
            image_id = await save_image_to_disk_and_db(file, dataset_id, new_scan.id, index_in_scan=index_in_scan)
            if height is None or width is None:
                height, width = get_height_width_of_image(image_id)
            else:
                # Validate that the image dimensions match the first image
                current_height, current_width = get_height_width_of_image(image_id)
                if current_height != height or current_width != width:
                    raise HTTPException(status_code=400, detail=f"Image {file.filename} has different dimensions "
                                                                f"({current_height}x{current_width}) than the first "
                                                                f"image ({height}x{width}). All images of a scan must "
                                                                f"have the same dimensions.")
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
        raise e
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


@router.post("/upload_scan_from_zip")
async def upload_scan_from_zip(
        dataset_id: int,
        zip_file: UploadFile = File(...),
        scan_type: str = "CT",
        description: str = "Scan description",
        db: Session = Depends(get_session)
):
    """ Upload a scan from a zip file.
    This endpoint extracts images from a zip file and uploads them as a scan.
    Args:
        dataset_id: ID of the dataset to which the scan belongs.
        zip_file: The zip file containing the scan images. The zip file should contain nothing but the slices and
        optionally a log file.
        scan_type: Type of scan (e.g., "CT", "MRI"). Optional.
        description: Description of the scan. Optional.
    Returns:
        A success message with the IDs of the uploaded images.
    """

    # Create a new scan entry in the database
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset or dataset.dataset_type != "scan":
        raise HTTPException(status_code=404, detail="Dataset not found or is not a scan dataset")

    new_scan = Scans(
        dataset_id=dataset_id,
        name=os.path.splitext(zip_file.filename)[0],
        type=scan_type.strip(),
        description=description.strip(),
        folder_path=os.path.join(dataset.folder_path, os.path.splitext(zip_file.filename)[0]),
        number_of_slices=0,
        #meta_data={}
    )
    db.add(new_scan)
    db.commit()

    try:
        image_ids = []
        with zipfile.ZipFile(io.BytesIO(zip_file.file.read())) as zf:
            for i, file_info in enumerate(zf.infolist()):
                if not file_info.is_dir():
                    if file_info.filename.endswith('.log'):
                        # If a log file is found, parse it and update the scan metadata
                        log_data = parse_log_file(zf.open(file_info).read())
                        # new_scan.meta_data = log_data
                        # TO BE DONE: ADD functionality for the log file to update the scan metadata
                        db.commit()
                        continue
                    with zf.open(file_info) as file:
                        # Extract numbers from the filename. The last number will be used as the index in the scan.
                        numbers_in_filename = extract_numbers(file_info.filename)
                        index_in_scan = i if not numbers_in_filename else numbers_in_filename[-1]

                        # Save the image to disk and the database
                        image_id = await save_image_to_disk_and_db(file,
                                                                   dataset_id,
                                                                   new_scan.id,
                                                                   index_in_scan=index_in_scan)
                        if image_id is None:
                            raise HTTPException(status_code=400, detail="Invalid file or upload failed")
                        image_ids.append(image_id)

        new_scan.number_of_slices = len(image_ids)
        db.commit()

        return {
            "success": True,
            "image_ids": image_ids,
            "message": f"Successfully uploaded {len(image_ids)} images belonging to scan {new_scan.id}. "
                       f"Assigned image ids {image_ids}"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
