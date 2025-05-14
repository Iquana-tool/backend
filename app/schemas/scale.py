from pydantic import BaseModel, Field


class ScaleInput(BaseModel):
    """
    Represents the input data required to set the pixel scale for an image.
    This model is used for validating and processing the data sent to the 
    '/set_pixel_scale_via_drawn_line' API endpoint.
    """
    image_id: int = Field(..., description="The ID of the image in the database.")
    x1: float = Field(..., description="X-coordinate of the first point in pixels.")
    y1: float = Field(..., description="Y-coordinate of the first point in pixels.")
    x2: float = Field(..., description="X-coordinate of the second point in pixels.")
    y2: float = Field(..., description="Y-coordinate of the second point in pixels.")
    known_distance: float = Field(..., description="The real-world distance between the two points (e.g., in mm).")
    unit: str = Field(..., description="The unit of measurement (default is millimeters).")

    def validate_points(self) -> None:
        """
        Validates that the two points (x1, y1) and (x2, y2) are not identical.
        This ensures a valid scale can be computed.
        """
        if self.x1 == self.x2 and self.y1 == self.y2:
            raise ValueError("The two points must not be identical to compute the scale.")
