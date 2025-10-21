import json
from logging import getLogger

import numpy as np
from fastapi import APIRouter
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.database import get_session
from app.database.contours import Contours
from app.database.labels import Labels
from app.database.masks import Masks
from app.schemas.contours import Contour
from app.services.labels import get_hierarchical_label_name

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


@router.post("/modify_contour/{contour_id}")
async def modify_contour(contour_id, db: Session = Depends(get_session), **kwargs):
    """
    Edit a contour by updating its coordinates or label.

    Args:
        contour_id (int): The ID of the contour to edit.
        db (Session): The database session.
        **kwargs: Arbitrary keyword arguments to update the contour attributes.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the edited contour.
    """
    try:
        existing_contour = db.query(Contours).filter_by(id=contour_id).first()
        if not existing_contour:
            raise HTTPException(status_code=404, detail="Contour not found.")

        for key, value in kwargs.items():
            if hasattr(existing_contour, key):
                if key == "label":
                    # The user wants to change the label of a contour. Here we need to check its children and its parents,
                    # whether that change is possible.
                    pass
                setattr(existing_contour, key, value)

        db.commit()
        return {
            "success": True,
            "message": "Contour edited successfully.",
            "contour_id": existing_contour.id
        }
    except Exception as e:
        logger.error(f"Error modifying contour: {e}")
        db.rollback()
        raise e


@router.post("/change_contour_label/{contour_id}&new_label_id={new_label_id}")
async def change_contour_label(contour_id: int, new_label_id: int, db: Session = Depends(get_session)):
    """
    Edit the label of a contour.

    Args:
        contour_id (int): The ID of the contour to edit.
        new_label_id (int): The new label ID to set for the contour.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the edited contour.
    """
    return await modify_contour(contour_id, label=new_label_id, db=db)


@router.get("/finalise/{contour_id}")
async def finalise(contour_id: int, db: Session = Depends(get_session)):
    """ Mark a temporary contour as not temporary."""
    contour = db.query(Contours).filter_by(id=contour_id).first()
    contour.temporary = False
    db.commit()
    return {
        "success": True,
        "message": f"Contour {contour_id} finalised successfully.",
    }


@router.post("/add_contour")
async def add_contour(mask_id: int,
                      contour_to_add: Contour,
                      db: Session = Depends(get_session)):
    """
    Add a contour to a mask in the database.

    Args:
        mask_id (int): The ID of the mask to which the contour will be added.
        contour_to_add (Contour): The contour data to add.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the added contour.
    """
    try:
        parent_contour_id = contour_to_add.parent_id
        expected_parent_label = (db.query(Labels.parent_id).filter_by(id=contour_to_add.label).first())[0]
        should_have_parent = expected_parent_label is not None
        if should_have_parent and parent_contour_id is None:
            # Contour should have a parent but none was given.
            logger.error(f"Parent contour ID is None, but the label expects a parent ({expected_parent_label}).")
            return {
                "success": False,
                "message": f"Parent contour ID is None, but the label expects a parent ({expected_parent_label}).",
                "contour_id": None
            }
        elif should_have_parent and parent_contour_id is not None:
            # Contour should have a parent and one is given one
            parent_contour_label = db.query(Contours.label).filter_by(id=parent_contour_id).first()
            if expected_parent_label != parent_contour_label:
                logger.error(f"Error adding contour: Parent contour does not match the expected parent label."
                             f"\nGiven label of parent contour: ({parent_contour_label})"
                             f"\tExpected label of parent contour: ({expected_parent_label})")
                return {
                    "success": False,
                    "message": "Parent contour does not match the expected parent label.",
                    "contour_id": None
                }
        else:
            logger.error("Contour with label should not have a parent but has a parent contour id given.")
            return {
                "success": False,
                "message": "Contour with label should not have a parent but has a parent contour id given.",
                "contour_id": None
            }

        # Add contour to the database
        entry = contour_to_add.to_db_entry(mask_id)
        db.add(entry)
        db.commit()
        contour_to_add.id = entry.id

        return {
            "success": True,
            "message": "Contour added successfully.",
            "added_contour": contour_to_add.model_dump_json(),
        }
    except Exception as e:
        logger.error(f"Error adding contour: {e}")
        db.rollback()
        raise e


@router.delete("/delete_contour/{contour_id}")
async def delete_contour(contour_id: int, db: Session = Depends(get_session)):
    """
    Delete a contour and all its descendants (via CASCADE).
    Returns the list of deleted contour IDs.
    """
    try:
        # Fetch the contour and all descendants in one query
        contour = (
            db.query(Contours)
            .options(joinedload(Contours.children))
            .filter_by(id=contour_id)
            .first()
        )
        if not contour:
            raise HTTPException(status_code=404, detail="Contour not found.")

        # Collect all descendant IDs (including the root)
        deleted_ids = []
        stack = [contour]
        while stack:
            current = stack.pop()
            deleted_ids.append(current.id)
            stack.extend(current.children)  # Add children to the stack

        # Delete the root contour (CASCADE will handle the rest)
        db.delete(contour)
        db.commit()

        return {
            "success": True,
            "message": "Contour and descendants deleted successfully.",
            "deleted_contours": deleted_ids,
        }
    except Exception as e:
        logger.error(f"Error deleting contour: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Error deleting contour.")


@router.post("/add_contours")
async def add_contours(mask_id: int,
                       contours_to_add: list[Contour],
                       added_by: str,
                       temporary_list: list[bool],
                       db: Session = Depends(get_session)):
    """
    Add multiple contours to a mask in the database. Internally calls `add_contour` for each contour.

    Args:
        mask_id (int): The ID of the mask to which the contours will be added.
        contours_to_add (list[Contour]): A list of contour data to add.
        parent_contour_id (int, optional): The ID of the parent contour. Defaults to None.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and lists of added and failed contour IDs.
    """
    failed = []
    added_ids = []
    for contour_to_add, temporary in zip(contours_to_add, temporary_list):
        logger.info(f"Added {len(added_ids)} / {len(contours_to_add)} contours. Failed {len(failed)}")
        result = await add_contour(mask_id, contour_to_add, added_by, temporary, db)
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