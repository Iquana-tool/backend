from typing import List

import numpy as np
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from app.database.contours import Contours
from app.database.images import Images
from app.schemas.contours import Contour


class CompletionMainAPIRequest(BaseModel):
    image_id: int = Field(...)
    seed_contours: List[int] = Field(...)
    model_registry_key: str = Field(...)

    def to_service_request(self, db):
        height, width = db.query(Images.height, Images.width).filter_by(id=self.image_id).first()
        seeds = []
        label = None
        for contour_id in self.seed_contours:
            contour_db = db.query(Contours).filter_by(id=contour_id).first()
            if contour_db is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"Contour with id {contour_id} not found! "
                                           f"Completion request failed!")
            contour_model = Contour.from_db(contour_db)
            if label is None:
                label = contour_model.label_id
            else:
                if not label == contour_model.label_id:
                    raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                                        detail="You are trying to run completion on contours of different labels. This is "
                                               "not allowed!")
            seeds.append(np.argwhere(contour_model.to_binary_mask(height, width)).flatten())

        return CompletionServiceRequest(
            model_key=self.model_registry_key,
            user_id=self.user_id,
            seeds=seeds
        )


class CompletionServiceRequest(BaseModel):
    model_key: str = Field(..., description="The key of the model.")
    user_id: str = Field(..., description="The user id of the model.")
    seeds: List[List[int]] = Field(..., description="Seeds is a list of lists of indices that specify which pixels "
                                                    "should be used as a seed. Each list in the list represents one "
                                                    "object.")
