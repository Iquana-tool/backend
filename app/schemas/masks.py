from pydantic import BaseModel, Field, field_validator
from app.database.images import Images
from app.database import get_context_session
import numpy as np
from typing import List, Annotated


class MaskRequest(BaseModel):
    base64_mask: str
    label: Annotated[str, "A name for what the mask represents."]
    image_id: int
