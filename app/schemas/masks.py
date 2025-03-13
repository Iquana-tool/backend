from typing import Annotated

from pydantic import BaseModel


class MaskRequest(BaseModel):
    base64_mask: str
    label: Annotated[str, "A name for what the mask represents."]
    image_id: int
