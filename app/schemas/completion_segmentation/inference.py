from typing import List

import numpy as np
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from app.database.contours import Contours
from app.database.images import Images
from app.schemas.contours import Contour


class CompletionMainAPIRequest(BaseModel):
    image_id: int | None = Field(...)
    seed_contour_ids: List[int] = Field(...)
    model_key: str = Field(...)


class CompletionServiceRequest(BaseModel):
    model_key: str = Field(..., description="The key of the model.")
    user_id: str = Field(..., description="The user id of the model.")
    seeds: List[List[int]] = Field(..., description="Seeds is a list of lists of indices that specify which pixels "
                                                    "should be used as a seed. Each list in the list represents one "
                                                    "object.")
