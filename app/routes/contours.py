import json
import numpy as np
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi import APIRouter
from logging import getLogger

from app.database import get_session
from app.database.labels import Labels
from app.database.images import Images
from app.database.masks import Masks
from app.database.contours import Contours
from app.schemas.contours import ContourModel
from app.services.database_access import get_height_width_of_image
from app.services.labels import get_hierarchical_label_name
from app.services.contours import find_parent_contour, coords_to_cv_contour
from app.services.quantifications import ContourQuantifier

router = APIRouter(prefix="/contours", tags=["contours"])
logger = getLogger(__name__)


def build_hierarchical_json(mask_id, filter_labels_ids, db: Session, parent_id=None):
    """ Build a hierarchical JSON structure of contours for a given mask_id.

    Args:
        mask_id (int): The ID of the mask to filter contours.
        filter_labels_ids (list[int]): Optional list of label IDs to filter contours.
        db (Session): The database session.
        parent_id (int): Optional parent contour ID to filter contours.

    Returns:
        list: A list of contours in hierarchical JSON format."""
    query = db.query(Contours).filter_by(mask_id=mask_id, parent_id=parent_id)
    if filter_labels_ids:
        query = query.filter(Contours.label.in_(filter_labels_ids))
    contours = query.all()

    result = []
    for contour in contours:
        label_name = get_hierarchical_label_name(contour.label)
        child_contours = build_hierarchical_json(mask_id, filter_labels_ids, db, contour.id)
        diameters = json.loads(contour.diameters) if isinstance(contour.diameters, str) else contour.diameters
        coords = json.loads(contour.coords) if isinstance(contour.coords, str) else contour.coords
        result.append({
            "id": contour.id,
            "label": contour.label,
            "label_name": label_name,
            "parent_id": parent_id,
            "area": contour.area,
            "perimeter": contour.perimeter,
            "circularity": contour.circularity,
            "diameters": diameters,
            "diameter_avg": np.average(diameters) if diameters else None,
            "coords": coords,
            "center_x": np.mean(coords["x"]) if coords and "x" in coords else None,
            "center_y": np.mean(coords["y"]) if coords and "y" in coords else None,
            "children": child_contours
        })
    return result


def flatten_hierarchical_dict(hierarchical_dict, parent_id=None):
    """ Flatten a hierarchical dictionary into a list of dictionaries.

    Args:
        hierarchical_dict (list): The hierarchical dictionary to flatten.
        parent_id (int): The parent ID for the current level.

    Returns:
        list: A flattened list of dictionaries."""
    flat_list = []
    for item in hierarchical_dict:
        flat_item = {
            k: v for k, v in item.items() if k != "children"
        }
        flat_list.append(flat_item)
        if item.get("children"):
            flat_list.extend(flatten_hierarchical_dict(item["children"], item["id"]))
    return flat_list


@router.get("/get_contours_of_mask/{mask_id}&flattened={flattened}")
async def get_contours_of_mask(mask_id: int, flattened: bool = True, db: Session = Depends(get_session)):
    """ Export quantification data for the given mask_id and labels.

    Args:
        mask_id (int): The ID of the mask to export contours for.
        flattened (bool): Whether to flatten the hierarchical JSON structure. Defaults to True. If False, the
            hierarchical structure will be preserved, i.e. children contours will be nested under their
            parent contour.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.

    Returns:
        dict: A dictionary containing the success status and message if error, or a hierarchical JSON structure of
        contours for the given mask_id.
    """
    quantification = build_hierarchical_json(mask_id, [], db)
    if flattened:
        quantification = flatten_hierarchical_dict(quantification)
    return {
        "success": True,
        "message": f"Quantification data for mask {mask_id} exported successfully.",
        "quantification": quantification
    }


@router.post("/edit_contour/{contour_id}")
async def edit_contour(contour_id, db: Session = Depends(get_session), **kwargs,):
    """
    Edit a contour by updating its coordinates or label.

    Args:
        contour_id (int): The ID of the contour to edit.
        db (Session): The database session.
        **kwargs: Arbitrary keyword arguments to update the contour attributes.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the edited contour.
    """
    existing_contour = db.query(Contours).filter_by(id=contour_id).first()
    if not existing_contour:
        raise HTTPException(status_code=404, detail="Contour not found.")

    for key, value in kwargs.items():
        if hasattr(existing_contour, key):
            setattr(existing_contour, key, value)

    db.commit()
    return {
        "success": True,
        "message": "Contour edited successfully.",
        "contour_id": existing_contour.id
    }


@router.post("/edit_contour_label/{contour_id}&new_label_id={new_label_id}")
async def edit_contour_label(contour_id: int, new_label_id: int, db: Session = Depends(get_session)):
    """
    Edit the label of a contour.

    Args:
        contour_id (int): The ID of the contour to edit.
        new_label_id (int): The new label ID to set for the contour.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the edited contour.
    """
    return await edit_contour(contour_id, label=new_label_id, db=db)


@router.post("/add_contour")
async def add_contour(mask_id: int,
                      contour_to_add: ContourModel,
                      parent_contour_id: int = None,
                      db: Session = Depends(get_session)):
    """
    Add a contour to a mask in the database.

    Args:
        mask_id (int): The ID of the mask to which the contour will be added.
        contour_to_add (ContourModel): The contour data to add.
        parent_contour_id (int, optional): The ID of the parent contour. Defaults to None.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the added contour.
    """
    try:
        existing_mask = db.query(Masks).filter_by(id=mask_id).first()
        image = db.query(Images).filter_by(id=existing_mask.image_id).first()
        contour = coords_to_cv_contour(contour_to_add.x, contour_to_add.y)
        parent_label_id = (db.query(Labels.parent_id).filter_by(id=contour_to_add.label).first())[0]
        has_parent = parent_label_id is not None
        if has_parent:
            if parent_contour_id is None:
                logger.warning(f"Parent contour ID is None, but the label has parent ({parent_label_id}). Trying to find a fitting parent"
                               " contour.")
                parent_contour = find_parent_contour(parent_label_id,
                                                     mask_id,
                                                     (image.height, image.width),
                                                     contour,
                                                     db)
                if not parent_contour:
                    logger.error(f"Error adding contour: Could not find a parent contour for label {contour_to_add.label}.")
                    return {
                        "success": False,
                        "message": "Could not find a parent contour for the label.",
                        "contour_id": None
                    }
                logger.debug(f"Found parent contour with ID {parent_contour.id} for label {contour_to_add.label}.")
                parent_contour_id = parent_contour.id
            else:
                parent_contour = db.query(Contours).filter_by(id=parent_contour_id).first()
                expected_parent = db.query(Labels).filter_by(id=parent_label_id).first()
                given_parent = db.query(Labels).filter_by(id=parent_contour.label).first()
                if expected_parent.id != given_parent.id:
                    logger.error(f"Error adding contour: Parent contour does not match the expected parent label. \n"
                                 f"Given parent contour: ({given_parent.id}, {given_parent.name}) \t Expected parent: ({expected_parent.id}, {expected_parent.name})")
                    return {
                        "success": False,
                        "message": "Parent contour does not match the expected parent label.",
                        "contour_id": None
                    }

        # Quantify contour
        height, width = get_height_width_of_image(existing_mask.image_id)
        rescaled_x = [int(x * width) for x in contour_to_add.x]
        rescaled_y = [int(y * height) for y in contour_to_add.y]
        quantifier = ContourQuantifier().from_coordinates(rescaled_x, rescaled_y)
        new_contour = Contours(
            mask_id=mask_id,
            parent_id=parent_contour_id,
            coords=json.dumps({"x": contour_to_add.x, "y": contour_to_add.y}),
            label=contour_to_add.label,
            area=quantifier.area,
            perimeter=quantifier.perimeter,
            circularity=quantifier.circularity,
            diameters=json.dumps(quantifier.get_diameters()),
        )

        # Add contour to the database
        db.add(new_contour)
        db.commit()

        return {
            "success": True,
            "message": "Contour added successfully.",
            "contour_id": new_contour.id
        }
    except Exception as e:
        logger.error(f"Error adding contour: {e}")
        raise e
        raise HTTPException(status_code=500, detail="Error adding contour.")


@router.delete("/delete_contour/{contour_id}")
async def delete_contour(contour_id: int, db: Session = Depends(get_session)):
    """
    Delete a contour and its child contours from the database.

    Args:
        contour_id (int): The ID of the contour to delete.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    try:
        # Check if contour exists
        existing_contour = db.query(Contours).filter_by(id=contour_id).first()
        if not existing_contour:
            raise HTTPException(status_code=404, detail="Contour not found.")
        # Check if the contour has child contours
        child_contours = db.query(Contours).filter_by(parent_id=contour_id).all()
        for child_contour in child_contours:
            # Recursively delete child contours
            await delete_contour(child_contour.id, db)
        # Delete the contour
        db.delete(existing_contour)
        db.commit()
        return {
            "success": True,
            "message": "Contour deleted successfully."
        }
    except Exception as e:
        logger.error(f"Error deleting contour: {e}")
        raise HTTPException(status_code=500, detail="Error deleting contour.")


@router.post("/add_contours")
async def add_contours(mask_id: int,
                       contours_to_add: list[ContourModel],
                       parent_contour_id: int = None,
                       db: Session = Depends(get_session)):
    """
    Add multiple contours to a mask in the database. Internally calls `add_contour` for each contour.

    Args:
        mask_id (int): The ID of the mask to which the contours will be added.
        contours_to_add (list[ContourModel]): A list of contour data to add.
        parent_contour_id (int, optional): The ID of the parent contour. Defaults to None.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and lists of added and failed contour IDs.
    """
    failed = []
    added_ids = []
    for contour_to_add in contours_to_add:
        logger.info(f"Added {len(added_ids)} / {len(contours_to_add)} contours. Failed {len(failed)}")
        result = await add_contour(mask_id, contour_to_add, parent_contour_id, db)
        if not result["success"]:
            failed.append({
                "contour": contour_to_add,
                "error": result["message"]
            })
        else:
            added_ids.append(result["contour_id"])
    if failed:
        return {
            "success": False,
            "message": f"Added {len(added_ids)} contours. Failed to add {len(failed)} contours.",
            "mask_id": mask_id,
            "failed": failed,
            "added_ids": added_ids
        }
    else:
        return {
            "success": True,
            "message": "All contours added successfully.",
            "mask_id": mask_id,
            "added_ids": added_ids,
            "failed": []
        }


@router.delete("/delete_all_contours_of_mask/{mask_id}")
async def delete_all_contours_of_mask(mask_id: int, db: Session = Depends(get_session)):
    """ Deletes all contours of a mask. """
    try:
        contours = db.query(Contours).filter_by(mask_id=mask_id).delete()
        mask = db.query(Masks).filter_by(id=mask_id).first()
        mask.generated = False
        mask.finished = False
        db.commit()
        return {
            "success": True,
            "message": f"Deleted all contours of mask {mask_id}"
        }
    except Exception as e:
        logger.error(e)
        raise HTTPException(500, e)