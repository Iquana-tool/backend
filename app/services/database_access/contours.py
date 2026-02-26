from sqlalchemy.orm import Session

from app.database.contours import Contours, save_contour_tree
from logging import getLogger


logger = getLogger(__name__)


async def replace_contour(old_contour_id, new_contour_model, db: Session):
    new_contour_model.id = old_contour_id
    contour = db.query(Contours).filter_by(id=old_contour_id).first()
    if not contour:
        return False
    db.query(Contours).filter_by(id=old_contour_id).delete()
    save_contour_tree(db, new_contour_model, contour.mask_id, contour.parent_id)
    db.commit()
    return True
