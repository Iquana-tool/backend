"""Ask a trained model how unsure it is about an image, without saving any objects.

Used by the active-learning annotation queue's Uncertainty ordering. The model runs at a
low threshold so that the doubtful detections, the ones normal inference throws away,
count towards the score. An image is as uncertain as its most borderline detection (see
app.services.al_ranking.image_uncertainty).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Iterable, Protocol

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from sqlalchemy.orm import Session

from app.database.image_scores import ImageScores
from app.database.images import Images
from app.services.al_ranking import image_uncertainty

#: Keep detections down to this confidence; normal inference discards them at 0.5.
SCORING_THRESHOLD = 0.1


class ScoreClient(Protocol):
    def segment(self, request: InstanceSegmentationRequest) -> object: ...


class HttpScoreClient:
    """The real client: the instance-segmentation surface of the ai-service."""

    def segment(self, request: InstanceSegmentationRequest) -> object:
        from app.services.ai_services.instance_segmentation import InstanceSegmentationService

        return asyncio.run(InstanceSegmentationService().inference(request))


def _contours(response: object) -> list[Contour]:
    # The annotation-session route wraps results in {success, message, result}; accept a
    # bare list too, as the batch inference path does.
    payload = response.get("result", []) if isinstance(response, dict) else response
    return [item if isinstance(item, Contour) else Contour.model_validate(item)
            for item in (payload or [])]


def _confidences(contours: Iterable[Contour]) -> list[float]:
    values: list[float] = []
    for contour in contours:
        values.append(float(contour.confidence))
        values.extend(_confidences(contour.children))
    return values


def score_image(
        db: Session,
        image: Images,
        model_key: str,
        username: str,
        client: ScoreClient | None = None,
) -> ImageScores:
    """Score one image with one model and upsert its row. Does not commit."""
    client = client or HttpScoreClient()
    request = InstanceSegmentationRequest(
        image_url=str(image.file_path),
        user_id=username,
        model_registry_key=model_key,
        label=None,
        parameters={"threshold": SCORING_THRESHOLD},
    )
    confidences = _confidences(_contours(client.segment(request)))

    row = (db.query(ImageScores)
           .filter(ImageScores.image_id == image.id, ImageScores.model_key == model_key)
           .one_or_none())
    if row is None:
        row = ImageScores(image_id=image.id, model_key=model_key)
        db.add(row)
    row.score = image_uncertainty(confidences)
    row.num_detections = len(confidences)
    row.confidences = confidences
    row.scored_at = datetime.now(timezone.utc)
    db.flush()
    return row
