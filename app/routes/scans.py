import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.images import Images
from app.services.database_access import delete_image_from_disk_and_db
from app.services.database_access import save_image_to_disk_and_db, load_image_as_base64_from_disk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scans", tags=["images"])


@router.post("/upload_scan")
async def upload_scan(files: list[UploadFile] = File(...), db: Session = Depends(get_session)):
    """Upload a scan file"""
    try:
        image_ids = []
        for file in files:
            image_id = await save_image_to_disk_and_db(file)
            if image_id is None:
                raise HTTPException(status_code=400, detail="Invalid file or upload failed")
            image_ids.append(image_id)

        return {
            "success": True,
            "image_ids": image_ids,
            "message": f"Successfully uploaded {len(files)} scans. Assigned ids {image_ids}"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
