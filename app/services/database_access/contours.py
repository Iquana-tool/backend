from logging import getLogger

from iquana_toolbox.schemas.contours import Contour
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database.contours import Contours, save_contour_tree, reviewer_contour_association
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users
from app.services.database_access.labels import get_label_hierarchy

logger = getLogger(__name__)


async def get_contour(
        contour_id: int,
        db: Session
) -> Contour:
    """ Get a contour by its contour id. """
    existing_contour = db.query(Contours).filter_by(id=contour_id).first()
    if not existing_contour:
        raise KeyError(f"Contour with id {contour_id} does not exist")
    return Contour.from_db(existing_contour)


async def get_contours(
        contour_ids: list[int],
        db: Session
):
    contours_db = db.query(Contours).filter(Contours.id.in_(contour_ids)).all()
    return [Contour.from_db(contour_db) for contour_db in contours_db]


async def _check_contour_label(
        contour: Contour,
        new_label_id: int,
        db: Session
):
    """ Change the label of a contour. """
    # We need to ensure the integrity of our label hierarchy here, hence this is handled separately
    # For this we need the label hierarchy of the dataset, so we first fetch the dataset id
    dataset_id = (
        db.query(Images.dataset_id)
        .join(Masks, Masks.image_id == Images.id)
        .join(Contours, Contours.mask_id == Masks.id)
        .filter(Contours.id == contour.id)
        .scalar()
    )
    label_hierarchy = await get_label_hierarchy(dataset_id, db)
    parent_contour = db.query(Contours).filter_by(id=contour.parent_id).one_or_none()
    if parent_contour is None:
        parent_label_id = None
    else:
        parent_label_id = parent_contour.label_id
    if label_hierarchy.is_label_valid(new_label_id, parent_label_id):
        # New label is valid
        contour.label_id = new_label_id
        return contour
    else:
        raise ValueError(
            f"Label with id {new_label_id} is not valid for this dataset."
        )


async def review_contour(
        contour_id: int,
        user: User,
        db: Session
):
    contour_db = db.query(Contours).filter_by(id=contour_id).first()
    user_db = db.query(Users).filter_by(username=user.username).first()
    if not contour_db:
        raise KeyError(f"Contour with id {contour_id} does not exist")
    if user not in contour_db.reviewed_by:
        contour_db.reviewed_by.append(user_db)
        db.commit()


async def delete_contour(
        contour_id: int,
        db: Session
):
    # Fetch the contour and all descendants in one query
    contour = (
        db.query(Contours)
        .filter_by(id=contour_id)
        .first()
    )
    if not contour:
        return

    # Delete the root contour (CASCADE will handle the rest)
    db.delete(contour)
    db.commit()


async def remove_review(
        contour_id: int,
        user: User,
        db: Session
):
    contour_db = db.query(Contours).filter_by(id=contour_id).first()
    user_db = db.query(Users).filter_by(username=user.username).first()
    if not contour_db:
        raise KeyError(f"Contour with id {contour_id} does not exist")
    if user in contour_db.reviewed_by:
        contour_db.reviewed_by.remove(user_db)
        db.commit()


async def modify_contour(
        contour_id: int,
        db: Session,
        **kwargs
):
    """
        Modify a contour by its contour id and kwargs. Checks whether the keyword exists, then updates the field.
        Note: This method calls the db quite a lot and should be avoided. Rather instantiate a new model schema and call
        replace_contour.
    """
    contour_db = db.query(Contours).filter_by(id=contour_id).first()
    contour = Contour.from_db(contour_db)
    for key, value in kwargs.items():
        if key in contour.__dict__:
            if key == "label_id":
                contour = await _check_contour_label(contour, value, db)
            else:
                setattr(contour, key, value)
    return await replace_contour(contour_db.id, contour, db)


async def replace_contour(
        old_contour_id,
        new_contour_model,
        db: Session
):
    """ Replace a contour with a new one. """
    new_contour_model.id = old_contour_id
    contour = db.query(Contours).filter_by(id=old_contour_id).first()
    if not contour:
        return False
    db.query(Contours).filter_by(id=old_contour_id).delete()
    save_contour_tree(db, new_contour_model, contour.mask_id, contour.parent_id)
    db.commit()
    return True
