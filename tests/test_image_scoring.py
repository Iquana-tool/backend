"""Scoring one image: request shape, nested detections, the no-detection case, upsert."""
import pytest

from app.database.image_scores import ImageScores
from app.services import image_scoring
from app.services.al_ranking import binary_entropy
from tests.test_queue_prep import FakeScoreClient, db, make_dataset, make_image  # noqa: F401


def test_score_image_counts_nested_detections_and_upserts(db):  # noqa: F811
    ds = make_dataset(db)
    img = make_image(db, ds, "a.png", state="none")
    client = FakeScoreClient({"result": [
        {"confidence": 0.9, "children": [{"confidence": 0.3}]},
        {"confidence": 0.6},
    ]})

    row = image_scoring.score_image(db, img, "m1", "ann", client=client)
    db.commit()

    request = client.requests[0]
    assert request.model_registry_key == "m1"
    assert request.label is None
    assert request.parameters == {"threshold": image_scoring.SCORING_THRESHOLD}
    assert row.num_detections == 3
    assert row.confidences == [0.9, 0.3, 0.6]
    # The most borderline detection (0.6) decides.
    assert row.score == pytest.approx(binary_entropy(0.6))

    client.response = []
    image_scoring.score_image(db, img, "m1", "ann", client=client)
    db.commit()
    rows = db.query(ImageScores).all()
    assert len(rows) == 1
    assert rows[0].score == 0.0
    assert rows[0].num_detections == 0
    assert rows[0].confidences == []
