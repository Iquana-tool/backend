"""Backend annotation state in addition to the toolbox's geometry schema."""
from pydantic import Field
from iquana_toolbox.schemas.database.contours import Contour


class AnnotationContour(Contour):
    hierarchy_enabled: bool = True
    children: list["AnnotationContour"] = Field(default_factory=list)
