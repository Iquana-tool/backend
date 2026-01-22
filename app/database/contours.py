import json

from schemas.contours import Contour
from sqlalchemy import Column, Integer, ForeignKey, Float, JSON, Boolean, String, Table
from sqlalchemy.orm import relationship
from app.database import database


reviewer_contour_association = Table('reviewer_contour_association',
                                     database.metadata,
                                     Column('reviewer_id', Integer,
                                            ForeignKey('users.username', ondelete='CASCADE'), primary_key=True),
                                     Column('contour_id', Integer,
                                            ForeignKey('contours.id', ondelete='CASCADE'), primary_key=True),
                                     )


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
    label_id = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'), nullable=True)
    area = Column(Float, nullable=False)
    perimeter = Column(Float, nullable=False)
    circularity = Column(Float, nullable=False)
    diameter = Column(Float, nullable=False)
    x = Column(JSON, nullable=False)
    y = Column(JSON, nullable=False)
    # Easy access to children, this makes accessing children much faster
    children = relationship("Contours", backref="parent", remote_side=[id], single_parent=True)
    reviewed_by = relationship("Users", secondary=reviewer_contour_association, back_populates="reviewed_objects")

    @classmethod
    def from_schema(cls, model_schema: "Contour", mask_id: int):
        """
        Creates a Contours DB instance from a Pydantic Contour schema.
        Assumes model_schema.quantification is already populated by your validator.
        """
        # Handle quantification mapping safely
        quant = model_schema.quantification

        return cls(
            id=model_schema.id,  # SQLAlchemy handles None as autoincrement
            mask_id=mask_id,
            parent_id=model_schema.parent_id,
            added_by=model_schema.added_by,
            confidence_score=model_schema.confidence,
            label_id=model_schema.label_id,
            # Normalized coordinates stored as JSON lists
            x=model_schema.x,
            y=model_schema.y,
            # Flat mapping of the nested Quantification object
            area=quant.area if quant else 0.0,
            perimeter=quant.perimeter if quant else 0.0,
            circularity=quant.circularity if quant else 0.0,
            diameter=quant.max_diameter if quant else 0.0,
        )


def save_contour_tree(session, contour_schema: Contour, mask_id: int, parent_id=None):
    """Recursively saves a contour and all its children to the DB."""
    # 1. Convert schema to DB model
    db_contour = Contours.from_schema(contour_schema, mask_id)
    db_contour.parent_id = parent_id

    # 2. Add to session and flush to generate db_contour.id
    session.add(db_contour)
    session.flush()

    # 3. Recurse for children
    for child_schema in contour_schema.children:
        save_contour_tree(session, child_schema, mask_id, parent_id=db_contour.id)

    return db_contour
