from typing import List

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    image_id: int = Field(...)
    contour_ids: List[int] = Field(...)
    model_registry_key: str = Field(...)
