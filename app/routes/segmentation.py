from fastapi import APIRouter, UploadFile, File, HTTPException
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/segmentation/health")
def check_health():
    """Check if segmentation service is running"""
    return {"status": "ok", "service": "segmentation"}

@router.post("/segmentation")
async def segment_image(file: UploadFile = File(None)):
    """Placeholder for image segmentation endpoint"""
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    return {
        "message": "Image received for segmentation",
        "filename": file.filename,
        "content_type": file.content_type
    }