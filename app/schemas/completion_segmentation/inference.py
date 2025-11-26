from typing import List

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    image_id: int = Field(...)
    seed_contours: List[int] = Field(...)
    model_registry_key: str = Field(...)


class CompletionBackendRequest(BaseModel):
    model_key: str = Field(..., description="The key of the model.")
    user_id: str = Field(..., description="The user id of the model.")
    seeds: List[List[int]] = Field(..., description="Seeds is a list of lists of indices that specify which pixels "
                                                    "should be used as a seed. Each list in the list represents one "
                                                    "object.")
