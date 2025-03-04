from pydantic import BaseModel, Field, field_validator
from app.database.images import Images
from app.database import get_context_session
import numpy as np


class RLEString(BaseModel):
    """Model for validating a run-length encoded string."""
    rle_string: str
    image_id: int
    starts_with_0: bool = Field(default=True, description="Whether the RLE string starts with a 0.")

    @field_validator('rle_string')
    def validate_rle_string(cls, value):
        if not isinstance(value, str):
            raise ValueError("RLE string must be a string.")
        if len(value) == 0:
            raise ValueError("RLE string must not be empty.")
        try:
            mask = cls.rle_decode(value)
            cls['mask'] = mask
        except Exception as e:
            raise ValueError(f"Could not decode RLE string: {e}")
        return value

    @staticmethod
    def rle_decode(rle_str):
        """Decodes an RLE encoded mask."""
        s = rle_str.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(max(ends), dtype=np.uint8)
        for start, end in zip(starts, ends):
            mask[start:end] = 1
        with get_context_session() as session:
            image = session.query(Images).filter_by(id=1).first()
        return mask.reshape((image.height, image.width))  # Replace height and width with actual dimensions


class MaskRequest(BaseModel):
    rle_mask: str
    label: str