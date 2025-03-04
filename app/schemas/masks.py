from pydantic import BaseModel, Field, field_validator
from app.database.images import Images
from app.database import get_context_session
import numpy as np


class MaskRequest(BaseModel):
    rle_mask: str
    label: str