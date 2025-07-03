from fastapi import APIRouter
from logging import getLogger


logger = getLogger(__name__)
router = APIRouter("/automatic_segmentation", tags=["automatic_segmentation"])
