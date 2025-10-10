from sqlalchemy import Column, Integer, ForeignKey, Float, JSON, Boolean, String
from sqlalchemy.orm import relationship
from app.database import database


class Contours(database):
    """Contours table to store contour information for masks."""
    __tablename__ = 'contours'
    id = Column(Integer, primary_key=True, autoincrement=True)
    mask_id = Column(Integer, ForeignKey('masks.id', ondelete='CASCADE'),
                     nullable=False)
    parent_id = Column(Integer, ForeignKey('contours.id', ondelete='CASCADE'))
    temporary = Column(Boolean, nullable=False, default=False)  # Whether a contour is temporary or not.
    added_by = Column(String(255), nullable=False)  # Who added this contour: User, SAM2, UNET, DINO etc.
    confidence_score = Column(Float, nullable=False)  # Confidence score provided by a model, for users this is set to 1
    # Allowing labels to be null, this allows contours without labels to exist, such that users can label them later.
    label = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'), nullable=True)
    area = Column(Float, nullable=False)
    perimeter = Column(Float, nullable=False)
    circularity = Column(Float, nullable=False)
    diameter = Column(Float, nullable=False)
    x = Column(JSON, nullable=False)
    y = Column(JSON, nullable=False)
    # Easy access to children, this makes accessing children much faster
    children = relationship("Contours", backref="parent", remote_side=[id], single_parent=True)