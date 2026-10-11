"""Propagation: carry objects drawn on some slices of a stack to the others (#121).

The model work happens in the ai-service (``/propagation``, SAM 2's video mode);
this module is the part around it:

1. **Build the request** (:func:`build_propagation_request`). Each object is one or
   more contours on frames of one stack -- the slices a person drew it on. They
   become masks on frame indices; every frame of the stack is sent by file path,
   as every other AI request sends its image.
2. **Write the answer** (:func:`save_propagated`). Each outline the model returns
   becomes an ordinary object on its frame: the seed's label, unreviewed, logged
   as an AI suggestion (source ``propagation``) with the model's presence score as
   its confidence. So propagated objects flow through review like any suggestion.

Objects are still per frame (there is no track linking an object across slices
yet), so a propagated object is a new object on each slice it reaches. What keeps
a re-run from piling up copies is the same rule batch inference uses: an outline
that overlaps an object already on that slice is dropped, never the other way round.

The toolbox request schemas are imported lazily: they exist only once the toolbox
pin includes them, and this module is on the app's import path.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

from fastapi import HTTPException, status
from iquana_toolbox.inference import best_parent, nms
from iquana_toolbox.schemas.database.contours import Contour
from sqlalchemy.orm import Session

from app.database.ai_suggestions import SuggestionSource
from app.database.contours import Contours, save_contour_tree
from app.database.images import Frames
from app.database.masks import Masks
from app.services import provenance
from app.services.database_access.contours import invalidate_metrics_for_new_contours
from app.services.database_access.stacks import get_stack

if TYPE_CHECKING:
    from iquana_toolbox.schemas.networking.http.services import PropagationRequest

logger = getLogger(__name__)

#: Outlines whose presence score is below this are written, but reported as the
#: slices a person should look at first.
CHECK_BELOW = 0.8
#: Overlap above which a propagated outline counts as an object already there.
DUPLICATE_IOU = 0.5


@dataclass
class PropagationSummary:
    """What a propagation run did, for the response and the frontend's check step."""

    created: int = 0
    contour_ids: list[int] = field(default_factory=list)
    #: Frame index -> number of objects written there.
    frames: dict[int, int] = field(default_factory=dict)
    #: Frames where something was written with a presence below :data:`CHECK_BELOW`.
    to_check: list[int] = field(default_factory=list)
    #: Frames the model judged an object absent on (the object ended there).
    absent: list[int] = field(default_factory=list)
    #: Outlines dropped because an object was already there.
    duplicates: int = 0
    #: Outlines of child objects with no parent object on their slice to nest under.
    unparented: int = 0


def _frames_of(db: Session, stack_id: int) -> list[Frames]:
    return (db.query(Frames).filter(Frames.stack_id == stack_id)
            .order_by(Frames.frame_index).all())


def build_propagation_request(
    db: Session,
    *,
    stack_id: int,
    objects: list[list[int]],
    model_key: str,
    user_id: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
    direction: str = "both",
    parameters: dict | None = None,
) -> tuple["PropagationRequest", dict[int, Contours]]:
    """Turn seed contours into an ai-service request.

    Args:
        objects: One list of contour ids per object -- the outlines of that object
            on the slices it was drawn on. One id is the common case; several give
            the model more to go on.

    Returns:
        ``(request, seeds)`` where ``seeds`` maps each request ``object_id`` to the
        contour that defines it (label and parent) when results are written. The
        object id is the first contour id of the object.

    Raises:
        404 for a missing stack; 400 for an empty object, an unknown contour, or a
        contour that is not on a frame of this stack.
    """
    from iquana_toolbox.schemas.networking.http.services import (
        FrameMask,
        PropagationObject,
        PropagationRequest,
    )

    get_stack(db, stack_id)
    frames = _frames_of(db, stack_id)
    index_of_image = {frame.id: frame.frame_index for frame in frames}

    request_objects = []
    seeds: dict[int, Contours] = {}
    first_drawn: int | None = None
    for contour_ids in objects:
        if not contour_ids:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "An object needs at least one outline.")
        frame_masks = []
        for contour_id in contour_ids:
            row = (db.query(Contours, Masks.image_id)
                   .join(Masks, Masks.id == Contours.mask_id)
                   .filter(Contours.id == contour_id).first())
            if row is None:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Object {contour_id} was not found.")
            contour, image_id = row
            if image_id not in index_of_image:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"Object {contour_id} is not on a slice of stack {stack_id}.")
            frame = frames[index_of_image[image_id]]
            shape = Contour(x=list(contour.x), y=list(contour.y), label_id=contour.label_id,
                            confidence=contour.confidence_score)
            frame_masks.append(FrameMask(
                frame_index=frame.frame_index,
                mask=shape.to_binary_mask_model(frame.height, frame.width),
            ))
            first_drawn = frame.frame_index if first_drawn is None else min(first_drawn, frame.frame_index)
        object_id = contour_ids[0]
        seeds[object_id] = db.get(Contours, object_id)
        request_objects.append(PropagationObject(object_id=object_id, frame_masks=frame_masks))

    request = PropagationRequest(
        image_url=frames[first_drawn].file_path,
        user_id=user_id,
        model_registry_key=model_key,
        parameters=parameters or {},
        frame_urls=[frame.file_path for frame in frames],
        objects=request_objects,
        start_frame=start_frame,
        end_frame=end_frame,
        direction=direction,
    )
    return request, seeds


def _shape(row: Contours) -> Contour:
    return Contour(id=row.id, label_id=row.label_id, parent_id=row.parent_id,
                   x=list(row.x or []), y=list(row.y or []), confidence=row.confidence_score)


def save_propagated(
    db: Session,
    *,
    stack_id: int,
    results: list[dict],
    seeds: dict[int, Contours],
    model_key: str,
    username: str,
) -> PropagationSummary:
    """Write the ai-service's propagated outlines onto their frames. Commits.

    Args:
        results: The ``/propagation/inference`` ``result`` entries,
            ``{object_id, frame_index, presence, contour}``.
    """
    frames = _frames_of(db, stack_id)
    masks = {mask.image_id: mask for mask in
             db.query(Masks).filter(Masks.image_id.in_([frame.id for frame in frames])).all()}
    seed_parent_labels = {
        object_id: db.query(Contours.label_id).filter(Contours.id == seed.parent_id).scalar()
        for object_id, seed in seeds.items() if seed.parent_id is not None
    }

    summary = PropagationSummary()
    to_check: set[int] = set()
    absent: set[int] = set()
    by_frame: dict[int, list[tuple[Contour, int]]] = defaultdict(list)
    for entry in results:
        index = entry["frame_index"]
        seed = seeds.get(entry["object_id"])
        if seed is None or not 0 <= index < len(frames):
            continue
        if entry.get("contour") is None:
            absent.add(index)
            continue
        contour = Contour.model_validate(entry["contour"])
        contour.label_id = seed.label_id
        contour.confidence = float(entry.get("presence", contour.confidence or 0))
        contour.added_by = model_key
        contour.parent_id = None
        contour.children = []
        by_frame[index].append((contour, entry["object_id"]))

    saved_rows: list[Contours] = []
    for index, candidates in sorted(by_frame.items()):
        mask = masks.get(frames[index].id)
        if mask is None:
            continue
        existing = db.query(Contours).filter(Contours.mask_id == mask.id,
                                             Contours.temporary.is_(False)).all()
        shapes = [contour for contour, _ in candidates]
        decision = nms(shapes, iou_threshold=DUPLICATE_IOU, existing=[_shape(row) for row in existing])
        summary.duplicates += len(decision.suppressed)
        for position in decision.kept:
            contour, object_id = candidates[position]
            parent_id = None
            if object_id in seed_parent_labels:
                # A child object nests under the parent object it falls inside on
                # this slice; with none there it is dropped, as batch inference does.
                parents = [row for row in existing if row.label_id == seed_parent_labels[object_id]]
                winner = best_parent(contour, [_shape(row) for row in parents])
                if winner is None:
                    summary.unparented += 1
                    continue
                parent_id = parents[winner].id
            saved = save_contour_tree(db, contour, mask.id, parent_id=parent_id,
                                      author_username=username, invalidate_metrics=False)
            saved_rows.append(saved)
            summary.contour_ids.append(saved.id)
            summary.frames[index] = summary.frames.get(index, 0) + 1
            if contour.confidence < CHECK_BELOW:
                to_check.add(index)

    invalidate_metrics_for_new_contours(db, saved_rows)
    provenance.record_suggestions(
        db, summary.contour_ids, source=SuggestionSource.PROPAGATION,
        model_key=model_key, username=username, commit=False,
    )
    db.commit()
    summary.created = len(summary.contour_ids)
    summary.to_check = sorted(to_check)
    summary.absent = sorted(absent - set(summary.frames))
    return summary
