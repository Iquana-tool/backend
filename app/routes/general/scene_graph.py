"""HTTP routes for the scene graph: given attributes and named relationships.

Everything sits under ``/scene-graph/dataset/{dataset_id}`` so the permission check
runs on the dataset in the path, and the service layer then refuses any id that
belongs to another dataset.

* Defining attributes and relationship names is curation: ``label.manage``.
* Setting values and drawing relationships follows the rules for editing a label:
  ``annotation.edit_own`` on your own objects, ``annotation.edit_any`` on anyone's.
* Reading needs ``annotation.read``.

Edits are REST rather than annotation-socket messages so that they work the same
from the annotation page, the dataset table and scripts. They still join the
annotator's undo history for the image(s) they touch.
"""
from contextlib import contextmanager
from logging import getLogger
from typing import Any

import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette import status

from app.database import get_session
from app.database.contours import Contours
from app.database.scene_graph import ObjectRelationships, SOURCE_USER, STATUS_CONFIRMED
from app.exceptions import InvalidSceneGraphError, SceneGraphNotFoundError
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.database_access import annotation_history as history_db
from app.services.database_access import scene_graph as graph_db
from app.services.label_to_attribute import convert_labels_to_attribute
from app.services.object_query import ObjectQuery, run_query
from app.services.permissions import ensure_permission, require

router = APIRouter(prefix="/scene-graph", tags=["scene graph"])
logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class AttributeOption(BaseModel):
    value: str
    code: str | None = Field(default=None, description="External identifier, e.g. an AphiaID or IRI.")
    description: str | None = None


class AttributeCreate(BaseModel):
    name: str
    value_type: str = Field(default="categorical",
                            description="One of text | categorical | number | date | boolean.")
    unit: str | None = None
    options: list[AttributeOption | str] = Field(
        default_factory=list, description="Allowed values of a categorical attribute. Empty = open.")
    description: str | None = None
    label_ids: list[int] = Field(default_factory=list,
                                 description="Labels the attribute applies to. Empty = every label.")


class AttributeUpdate(BaseModel):
    name: str | None = None
    value_type: str | None = None
    unit: str | None = None
    options: list[AttributeOption | str] | None = None
    description: str | None = None
    label_ids: list[int] | None = None
    rename_options: dict[str, str] | None = Field(
        default=None, description="{old: new}: renames a value in the vocabulary and on every object.")


class AttributeValueSet(BaseModel):
    contour_ids: list[int] = Field(..., min_length=1)
    value: str | float | bool | None = Field(
        ..., description="The value; null means 'looked at, cannot tell'.")
    source: str = SOURCE_USER
    status: str = STATUS_CONFIRMED
    model_key: str | None = None
    confidence: float | None = None


class ContourSelection(BaseModel):
    contour_ids: list[int] = Field(..., min_length=1)


class RelationshipTypeCreate(BaseModel):
    name: str
    symmetric: bool = False
    description: str | None = None


class RelationshipTypeUpdate(BaseModel):
    name: str | None = None
    symmetric: bool | None = None
    description: str | None = None
    merge_into: int | None = Field(default=None,
                                   description="Fold this type into another one; other fields are ignored.")


class RelationshipCreate(BaseModel):
    subject_id: int
    object_ids: list[int] = Field(..., min_length=1)
    type_id: int | None = None
    type_name: str | None = Field(default=None, description="Picked up if the dataset uses it, created if not.")
    source: str = SOURCE_USER
    status: str = STATUS_CONFIRMED
    model_key: str | None = None
    confidence: float | None = None


class RelationshipSelection(BaseModel):
    relationship_ids: list[int] = Field(..., min_length=1)


class LabelsToAttributeRequest(BaseModel):
    label_ids: list[int] = Field(..., min_length=1, description="Sibling is-a labels, e.g. species.")
    attribute_name: str = Field(..., description="Categorical attribute that takes their names.")
    target_label_id: int | None = Field(default=None, description="An existing sibling label ...")
    target_label_name: str | None = Field(default=None, description="... or a new one's name.")
    dry_run: bool = Field(default=True, description="Report only. Send false to convert.")


class TransferRequest(BaseModel):
    from_ids: list[int] = Field(..., min_length=1)
    to_id: int
    move_relationships: bool = Field(
        default=False, description="True for a merge (edges follow), false for a split.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(exc: Exception):
    if isinstance(exc, SceneGraphNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


def _ensure_may_edit_objects(db: Session, dataset_id: int, contour_ids,
                             user: AuthenticatedUser) -> None:
    """Refuse when any object belongs to someone else and the caller may only edit their own."""
    if user.has_permission(dataset_id, Permission.ANNOTATION_EDIT_ANY):
        return
    others = [cid for (cid, author) in
              db.query(Contours.id, Contours.author_username)
              .filter(Contours.id.in_(list(contour_ids))).all()
              if author not in (None, user.username)]
    if others:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Object(s) {', '.join(map(str, sorted(others)[:10]))} were created by someone "
                   f"else and you may only edit your own.",
        )


@contextmanager
def _graph_errors(db: Session):
    """Turn the service's domain errors into HTTP errors, rolling the session back."""
    try:
        yield
    except (InvalidSceneGraphError, SceneGraphNotFoundError) as exc:
        db.rollback()
        _fail(exc)


# ---------------------------------------------------------------------------
# Attribute definitions
# ---------------------------------------------------------------------------

@router.get("/dataset/{dataset_id}/attributes")
async def list_attributes(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ)),
):
    """Every attribute of the dataset, with its type, vocabulary and label scope."""
    return {"success": True,
            "attributes": [d.to_dict() for d in graph_db.list_attributes(db, dataset_id)]}


@router.post("/dataset/{dataset_id}/attributes", status_code=status.HTTP_201_CREATED)
async def create_attribute(
        dataset_id: int,
        body: AttributeCreate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE)),
):
    with _graph_errors(db):
        definition = graph_db.create_attribute(
            db, dataset_id, body.name, value_type=body.value_type, unit=body.unit,
            options=[o if isinstance(o, str) else o.model_dump() for o in body.options],
            description=body.description, label_ids=body.label_ids, username=user.username,
        )
    db.commit()
    return {"success": True, "attribute": definition.to_dict()}


@router.patch("/dataset/{dataset_id}/attributes/{attribute_id}")
async def update_attribute(
        dataset_id: int,
        attribute_id: int,
        body: AttributeUpdate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE)),
):
    """Rename, retype, change the vocabulary or scope. Stored values are re-validated,
    and the change is refused, naming the values that would not fit."""
    with _graph_errors(db):
        definition = graph_db.update_attribute(
            db, dataset_id, attribute_id, name=body.name, value_type=body.value_type,
            unit=body.unit,
            options=(None if body.options is None
                     else [o if isinstance(o, str) else o.model_dump() for o in body.options]),
            description=body.description, label_ids=body.label_ids,
            rename_options=body.rename_options,
        )
    db.commit()
    return {"success": True, "attribute": definition.to_dict()}


@router.delete("/dataset/{dataset_id}/attributes/{attribute_id}")
async def delete_attribute(
        dataset_id: int,
        attribute_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE)),
):
    """Delete an attribute and every value of it."""
    with _graph_errors(db):
        removed = graph_db.delete_attribute(db, dataset_id, attribute_id)
    db.commit()
    return {"success": True, "values_removed": removed}


# ---------------------------------------------------------------------------
# Attribute values
# ---------------------------------------------------------------------------

@router.post("/dataset/{dataset_id}/attributes/{attribute_id}/values")
async def set_attribute_values(
        dataset_id: int,
        attribute_id: int,
        body: AttributeValueSet,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN)),
):
    """Set one attribute to one value on one or many objects (one undo step)."""
    _ensure_may_edit_objects(db, dataset_id, body.contour_ids, user)
    with _graph_errors(db):
        changes = graph_db.set_attribute_values(
            db, dataset_id, attribute_id, body.contour_ids, body.value, username=user.username,
            source=body.source, status=body.status, model_key=body.model_key,
            confidence=body.confidence,
        )
    db.commit()
    history_db.record_attribute_changes(db, user.username, changes)
    return {"success": True, "changed": [c["contour_id"] for c in changes]}


@router.post("/dataset/{dataset_id}/attributes/{attribute_id}/values/clear")
async def clear_attribute_values(
        dataset_id: int,
        attribute_id: int,
        body: ContourSelection,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN)),
):
    """Unset an attribute on objects (back to 'not annotated')."""
    _ensure_may_edit_objects(db, dataset_id, body.contour_ids, user)
    with _graph_errors(db):
        changes = graph_db.clear_attribute_values(db, dataset_id, attribute_id, body.contour_ids)
    db.commit()
    history_db.record_attribute_changes(db, user.username, changes)
    return {"success": True, "cleared": [c["contour_id"] for c in changes]}


@router.post("/dataset/{dataset_id}/attributes/{attribute_id}/values/confirm")
async def confirm_attribute_values(
        dataset_id: int,
        attribute_id: int,
        body: ContourSelection,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN)),
):
    """Accept model-suggested values on objects."""
    _ensure_may_edit_objects(db, dataset_id, body.contour_ids, user)
    with _graph_errors(db):
        changes = graph_db.confirm_attribute_values(db, dataset_id, attribute_id,
                                                    body.contour_ids, user.username)
    db.commit()
    history_db.record_attribute_changes(db, user.username, changes)
    return {"success": True, "confirmed": [c["contour_id"] for c in changes]}


@router.post("/dataset/{dataset_id}/objects/attributes")
async def get_object_attributes(
        dataset_id: int,
        body: ContourSelection,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ)),
):
    """The given attributes of some objects, as ``{contour_id: [values]}``."""
    with _graph_errors(db):
        graph_db.contours_in_dataset(db, dataset_id, body.contour_ids)
    values = graph_db.attributes_of_contours(db, body.contour_ids)
    return {"success": True, "attributes": {str(cid): values.get(cid, []) for cid in body.contour_ids}}


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

@router.get("/dataset/{dataset_id}/relationship-types")
async def list_relationship_types(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ)),
):
    """Every relationship name of the dataset, with how often it is used (the canvas
    ranks its autocomplete by that)."""
    counts = graph_db.relationship_type_usage(db, dataset_id)
    return {"success": True,
            "relationship_types": [{**t.to_dict(), "usage_count": counts.get(t.id, 0)}
                                   for t in graph_db.list_relationship_types(db, dataset_id)]}


@router.post("/dataset/{dataset_id}/relationship-types", status_code=status.HTTP_201_CREATED)
async def create_relationship_type(
        dataset_id: int,
        body: RelationshipTypeCreate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE)),
):
    with _graph_errors(db):
        rel_type = graph_db.create_relationship_type(db, dataset_id, body.name, body.symmetric,
                                                     body.description, username=user.username)
    db.commit()
    return {"success": True, "relationship_type": rel_type.to_dict()}


@router.patch("/dataset/{dataset_id}/relationship-types/{type_id}")
async def update_relationship_type(
        dataset_id: int,
        type_id: int,
        body: RelationshipTypeUpdate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE)),
):
    """Rename, make (a)symmetric, or merge into another type."""
    with _graph_errors(db):
        if body.merge_into is not None:
            rel_type = graph_db.merge_relationship_types(db, dataset_id, type_id, body.merge_into)
        else:
            rel_type = graph_db.update_relationship_type(db, dataset_id, type_id, name=body.name,
                                                         symmetric=body.symmetric,
                                                         description=body.description)
    db.commit()
    return {"success": True, "relationship_type": rel_type.to_dict()}


@router.delete("/dataset/{dataset_id}/relationship-types/{type_id}")
async def delete_relationship_type(
        dataset_id: int,
        type_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE)),
):
    with _graph_errors(db):
        removed = graph_db.delete_relationship_type(db, dataset_id, type_id)
    db.commit()
    return {"success": True, "relationships_removed": removed}


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

@router.post("/dataset/{dataset_id}/relationships")
async def add_relationships(
        dataset_id: int,
        body: RelationshipCreate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE)),
):
    """Relate one object to one or more others: ``subject <type> object`` for each."""
    with _graph_errors(db):
        edges, created = graph_db.add_relationships(
            db, dataset_id, body.subject_id, body.object_ids, type_id=body.type_id,
            type_name=body.type_name, username=user.username, source=body.source,
            status=body.status, model_key=body.model_key, confidence=body.confidence,
        )
        result = [graph_db.edge_to_dict(edge) for edge in edges]
    db.commit()
    history_db.record_relationship_changes(db, user.username, created=created, deleted=[])
    return {"success": True, "relationships": result,
            "created": [state["id"] for state in created]}


@router.post("/dataset/{dataset_id}/relationships/delete")
async def delete_relationships(
        dataset_id: int,
        body: RelationshipSelection,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN)),
):
    """Delete relationships. Someone else's need ``annotation.edit_any``."""
    if not user.has_permission(dataset_id, Permission.ANNOTATION_EDIT_ANY):
        others = [rid for (rid, author) in
                  db.query(ObjectRelationships.id, ObjectRelationships.author_username)
                  .filter(ObjectRelationships.id.in_(body.relationship_ids)).all()
                  if author not in (None, user.username)]
        if others:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Some of these relationships were drawn by someone else.")
    with _graph_errors(db):
        deleted = graph_db.delete_relationships(db, dataset_id, body.relationship_ids)
    db.commit()
    history_db.record_relationship_changes(db, user.username, created=[], deleted=deleted)
    return {"success": True, "deleted": [state["id"] for state in deleted]}


@router.post("/dataset/{dataset_id}/relationships/confirm")
async def confirm_relationships(
        dataset_id: int,
        body: RelationshipSelection,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN)),
):
    """Accept model-suggested relationships."""
    with _graph_errors(db):
        edges = graph_db.confirm_relationships(db, dataset_id, body.relationship_ids, user.username)
        result = [graph_db.edge_to_dict(edge) for edge in edges]
    db.commit()
    return {"success": True, "relationships": result}


@router.post("/dataset/{dataset_id}/objects/transfer")
async def transfer_graph(
        dataset_id: int,
        body: TransferRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN)),
):
    """Carry attributes (and, for a merge, relationships) over to another object.

    Call it for a split (from the original to the new half) and before deleting the
    absorbed objects of a merge (from them to the survivor, ``move_relationships``).
    """
    _ensure_may_edit_objects(db, dataset_id, [*body.from_ids, body.to_id], user)
    with _graph_errors(db):
        result = graph_db.transfer(db, dataset_id, body.from_ids, body.to_id,
                                   body.move_relationships)
    db.commit()
    return {"success": True, **result}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@router.get("/image/{image_id}")
async def get_image_scene_graph(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ, "image_id")),
) -> dict[str, Any]:
    """One image as a scene graph: scene metadata, objects with their attributes and
    metrics, and every relationship (part-of and named)."""
    with _graph_errors(db):
        graph = graph_db.image_scene_graph(db, image_id)
    return {"success": True, **graph}


@router.post("/dataset/{dataset_id}/objects/query")
async def query_objects(
        dataset_id: int,
        body: ObjectQuery,
        format: str = "json",
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ)),
):
    """Find objects by label, given attributes, image metadata, measured metrics and
    relationships.

    Returns one row per object: image, label, parent, ``meta_*`` (image metadata),
    ``attr_*`` (given attributes), ``rel_*`` (relationships) and the requested
    metrics, in the same units as the quantification page. Metric thresholds are read
    in that unit too; ``scale_status`` says which it is (the ``X-Iquana-Unit`` header
    for CSV). ``format=csv`` downloads the table (needs ``export.quantification``).
    """
    if format not in ("json", "csv"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="format must be 'json' or 'csv'.")
    if format == "csv":
        ensure_permission(user, dataset_id, Permission.EXPORT_QUANTIFICATION)
    with _graph_errors(db):
        df = run_query(db, dataset_id, body)
    if format == "csv":
        buffer = io.StringIO()
        df.drop(columns=[c for c in ("coords_x", "coords_y") if c in df.columns]).to_csv(
            buffer, index=False)
        return StreamingResponse(
            iter([buffer.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="objects_dataset_{dataset_id}.csv"',
                     "X-Iquana-Unit": df.attrs["scale_status"]["display_unit"]},
        )
    return {"success": True, "count": len(df), "scale_status": df.attrs["scale_status"],
            "columns": list(df.columns),
            "rows": df.astype(object).where(df.notna(), None).to_dict(orient="records")}


@router.post("/dataset/{dataset_id}/labels/convert-to-attribute")
async def labels_to_attribute(
        dataset_id: int,
        body: LabelsToAttributeRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE)),
):
    """Fold is-a labels (``Acropora``, ``Pocillopora``) into one label (``Coral
    fragment``) and an attribute value each (``species``).

    Objects keep their ids, outlines, children and reviews. A dry run (the default)
    reports what would change: objects relabelled, values written, objects whose
    existing value disagrees, child labels, model mappings and profiles re-pointed.
    """
    with _graph_errors(db):
        report = convert_labels_to_attribute(
            db, dataset_id, body.label_ids, body.attribute_name,
            target_label_id=body.target_label_id, target_label_name=body.target_label_name,
            dry_run=body.dry_run, username=user.username,
        )
    if body.dry_run:
        db.rollback()
    else:
        db.commit()
    return {"success": True, **report.to_dict()}
