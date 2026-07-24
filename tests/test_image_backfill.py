"""The image-dimension backfill: repairs rows hit by the thumbnail-mutation
ingest bug and re-derives the geometry metrics computed from the wrong size."""
import asyncio

import pytest
from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users
from app.services.database_access.images import backfill_image_dimensions


@pytest.fixture
def ctx(tmp_path):
    """One image whose DB row carries thumbnail dimensions while the file on
    disk is the 400x300 original, plus a contour with metrics derived from the
    wrong size."""
    engine = create_engine(f"sqlite:///{tmp_path / 'backfill.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    db.add(Users(username="ann", hashed_password="x"))
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path=str(tmp_path), created_by="ann")
    db.add(ds)
    db.flush()

    file_path = tmp_path / "a.png"
    PILImage.new("RGB", (400, 300), (10, 20, 30)).save(file_path)

    img = Images(dataset_id=ds.id, file_name="a.png", file_path=str(file_path),
                 thumbnail_file_path=str(tmp_path / "t.png"),
                 # The bug: thumbnail dimensions in the row, original on disk.
                 width=200, height=150, color_mode="RGB")
    db.add(img)
    db.flush()
    mask = Masks(image_id=img.id, fully_annotated=True, file_path="/tmp/m.png")
    db.add(mask)
    db.flush()
    contour = Contours(mask_id=mask.id, added_by="manual", author_username="ann",
                       confidence_score=1.0, area=1.0, perimeter=1.0,
                       circularity=1.0, diameter=1.0,
                       x=[0.1, 0.5, 0.5, 0.1], y=[0.1, 0.1, 0.5, 0.5])
    db.add(contour)
    db.commit()

    yield {"db": db, "dataset_id": ds.id, "image_id": img.id,
           "contour_id": contour.id}
    db.close()


def test_backfill_fixes_dimensions_and_recomputes_geometry(ctx):
    db = ctx["db"]
    result = asyncio.run(backfill_image_dimensions(db, dataset_id=ctx["dataset_id"]))
    assert result["corrected"] == [ctx["image_id"]]
    assert result["missing"] == []
    assert result["recomputed_contours"] == 1

    image = db.query(Images).filter_by(id=ctx["image_id"]).one()
    assert (image.width, image.height) == (400, 300)

    # 0.4 x 0.4 of a 400x300 image = a 160x120 px rectangle.
    contour = db.query(Contours).filter_by(id=ctx["contour_id"]).one()
    assert contour.area == pytest.approx(160 * 120, rel=0.01)

    area_row = (db.query(ContourMetrics)
                .filter_by(contour_id=ctx["contour_id"], metric_key="area").one())
    assert area_row.value == pytest.approx(160 * 120, rel=0.01)
    assert area_row.stale is False


def test_backfill_is_idempotent(ctx):
    db = ctx["db"]
    asyncio.run(backfill_image_dimensions(db, dataset_id=ctx["dataset_id"]))
    again = asyncio.run(backfill_image_dimensions(db, dataset_id=ctx["dataset_id"]))
    assert again["corrected"] == []
    assert again["recomputed_contours"] == 0
