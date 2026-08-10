"""Tests for the three-phase image status (Calibrate -> Annotate -> Review).

The property that matters most here is the one the old single lifecycle could not
express: the three phases are *independent*. An image can be reviewed but never
calibrated, or calibrated but never annotated, and the overall status has to stay
``in_progress`` in both directions rather than collapsing to one axis.

Covered:
  * the combination rule (finished only when all three are, not started only while
    none are),
  * the Calibrate phase against the real calibration registry,
  * the Annotate/Review hybrids against a real database, including the reset a
    reviewer's send-back causes,
  * the dataset roll-up, which counts images and so must include images that have
    no mask row at all.
"""
import asyncio

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.contours  # noqa: F401
import app.database.dataset_calibration_defaults  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.image_calibrations  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contours import save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.schemas.review import RejectionCreate, RejectionReason
from app.services import image_status
from app.services.calibration import registry
from app.services.database_access import datasets as datasets_db
from app.services.database_access import members as members_db
from app.services.database_access import rejections as rejections_db

from iquana_toolbox.schemas.database.contours import Contour

WIDTH, HEIGHT = 100, 100

BLOCKED = image_status.BLOCKED
NOT_STARTED = image_status.NOT_STARTED
IN_PROGRESS = image_status.IN_PROGRESS
FINISHED = image_status.FINISHED


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'status.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


@pytest.fixture
def dataset(session, tmp_path):
    session.add(Users(username="owner", hashed_password="x"))
    session.flush()
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path=str(tmp_path), created_by="owner")
    session.add(ds)
    session.flush()
    members_db.ensure_owner_membership(ds.id, "owner", session)
    session.add(Labels(dataset_id=ds.id, parent_id=None, name="blob", value=1))
    session.commit()
    return ds


def _image(session, dataset, name="a.png"):
    img = Images(dataset_id=dataset.id, file_name=name, file_path=f"/tmp/{name}",
                 thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.commit()
    return img


def _mask(session, image, fully_annotated=False):
    mask = Masks(image_id=image.id, fully_annotated=fully_annotated,
                 file_path=f"/tmp/m{image.id}.png")
    session.add(mask)
    session.commit()
    return mask


def _contour(session, dataset, mask, reviewed_by=()):
    label = session.query(Labels).filter_by(dataset_id=dataset.id).first()
    entry = save_contour_tree(session, Contour(
        x=[0.1, 0.6, 0.6, 0.1], y=[0.1, 0.1, 0.6, 0.6],
        label_id=label.id, added_by="owner", reviewed_by=list(reviewed_by),
    ), mask.id)
    session.commit()
    return entry


def _calibrate_all(session, image):
    """Set every registered calibration kind on an image, the cheap way.

    Writes rows directly rather than going through the service: what this module
    tests is how *many* kinds are set, not whether any given kind's parameters are
    valid — that is ``test_calibration.py``'s job.
    """
    from app.database.image_calibrations import ImageCalibrations
    from app.services.calibration.registry import CalibrationSource

    for kind in registry.all_kinds():
        if kind.key == "scale":
            # Scale is read back out of the mirror columns, so the row alone is
            # not enough to make it read as calibrated.
            image.scale_x = image.scale_y = 0.01
            image.unit = "mm"
        session.add(ImageCalibrations(image_id=image.id, kind=kind.key,
                                      params={"set": True},
                                      source=CalibrationSource.MANUAL,
                                      created_by="owner"))
    session.commit()


# ---------------------------------------------------------------------------
# The combination rule
# ---------------------------------------------------------------------------

class TestCombine:
    def test_all_finished_is_finished(self):
        assert image_status.combine(FINISHED, FINISHED, FINISHED) == FINISHED

    def test_all_untouched_is_not_started(self):
        assert image_status.combine(NOT_STARTED, NOT_STARTED, NOT_STARTED) == NOT_STARTED

    def test_a_blocked_review_still_reads_as_untouched(self):
        """The real shape of a fresh image: nothing done, review blocked."""
        assert image_status.combine(NOT_STARTED, NOT_STARTED, BLOCKED) == NOT_STARTED

    def test_a_blocked_review_cannot_be_finished(self):
        assert image_status.combine(FINISHED, FINISHED, BLOCKED) == IN_PROGRESS

    @pytest.mark.parametrize("phases", [
        (FINISHED, FINISHED, NOT_STARTED),   # reviewed nothing yet
        (NOT_STARTED, FINISHED, FINISHED),   # annotated and reviewed, never calibrated
        (FINISHED, NOT_STARTED, NOT_STARTED),  # calibrated only
        (IN_PROGRESS, NOT_STARTED, NOT_STARTED),
    ])
    def test_any_mixture_is_in_progress(self, phases):
        """One finished phase must not make the image finished, and one untouched
        phase must not make it untouched. This is the conflation the old five-value
        status could not avoid."""
        assert image_status.combine(*phases) == IN_PROGRESS

    def test_no_registered_kinds_does_not_block_finishing(self):
        """A deployment with no calibration kinds must still be able to finish."""
        assert image_status.calibrate_status_from_counts(0, 0) == FINISHED

    @pytest.mark.parametrize("calibrated,expected", [
        (0, NOT_STARTED), (1, IN_PROGRESS), (2, FINISHED),
    ])
    def test_calibrate_state_from_counts(self, calibrated, expected):
        assert image_status.calibrate_status_from_counts(calibrated, 2) == expected


# ---------------------------------------------------------------------------
# Per-image phases
# ---------------------------------------------------------------------------

class TestImagePhases:
    def test_fresh_image_has_nothing_started(self, session, dataset):
        image = _image(session, dataset)
        state = image_status.status_for_image(session, image)
        # Review is blocked, not "not started": there is nothing drawn to review.
        assert state["phases"] == {"calibrate": NOT_STARTED,
                                   "annotate": NOT_STARTED,
                                   "review": BLOCKED}
        # A blocked review must not stop the image reading as untouched overall.
        assert state["status"] == NOT_STARTED
        assert state["mask_id"] is None

    def test_review_unblocks_as_soon_as_anything_is_drawn(self, session, dataset):
        """The boundary is "any annotation exists", not "annotation is finished".

        The review queue defaults to only offering submitted masks, but a reviewer
        can turn that off and sweep work in progress -- so a drawn-but-unsubmitted
        mask really is reviewable, and calling it blocked would be wrong.
        """
        image = _image(session, dataset)
        mask = _mask(session, image)
        assert image_status.status_for_image(session, image)["phases"]["review"] == BLOCKED

        _contour(session, dataset, mask)
        session.refresh(mask)
        phases = image_status.status_for_image(session, image)["phases"]
        assert phases["annotate"] == IN_PROGRESS
        assert phases["review"] == NOT_STARTED

    def test_emptying_a_mask_blocks_review_again(self, session, dataset):
        image = _image(session, dataset)
        mask = _mask(session, image)
        contour = _contour(session, dataset, mask)

        session.delete(contour)
        session.commit()
        session.refresh(mask)
        assert image_status.status_for_image(session, image)["phases"]["review"] == BLOCKED

    def test_partial_calibration_is_in_progress(self, session, dataset):
        image = _image(session, dataset)
        image.scale_x = image.scale_y = 0.01
        image.unit = "mm"
        session.commit()

        state = image_status.status_for_image(session, image)
        assert state["phases"]["calibrate"] == IN_PROGRESS
        assert state["status"] == IN_PROGRESS

    def test_calibration_alone_never_finishes_the_image(self, session, dataset):
        image = _image(session, dataset)
        _calibrate_all(session, image)

        state = image_status.status_for_image(session, image)
        assert state["phases"]["calibrate"] == FINISHED
        assert state["status"] == IN_PROGRESS

    def test_full_workflow_finishes(self, session, dataset):
        image = _image(session, dataset)
        _calibrate_all(session, image)
        mask = _mask(session, image)
        _contour(session, dataset, mask, reviewed_by=["owner"])
        mask.fully_annotated = True
        session.commit()

        state = image_status.status_for_image(session, image)
        assert state["phases"] == {"calibrate": FINISHED,
                                   "annotate": FINISHED,
                                   "review": FINISHED}
        assert state["status"] == FINISHED

    def test_review_without_calibration_stays_in_progress(self, session, dataset):
        """The case the old model got wrong: it reported this image as finished."""
        image = _image(session, dataset)
        mask = _mask(session, image)
        _contour(session, dataset, mask, reviewed_by=["owner"])
        mask.fully_annotated = True
        session.commit()

        state = image_status.status_for_image(session, image)
        assert state["phases"]["review"] == FINISHED
        assert state["phases"]["calibrate"] == NOT_STARTED
        assert state["status"] == IN_PROGRESS

    def test_send_back_resets_annotate_and_review(self, session, dataset):
        image = _image(session, dataset)
        _calibrate_all(session, image)
        mask = _mask(session, image)
        contour = _contour(session, dataset, mask, reviewed_by=["owner"])
        mask.fully_annotated = True
        session.commit()
        assert image_status.status_for_image(session, image)["status"] == FINISHED

        asyncio.run(rejections_db.reject(
            mask.id,
            RejectionCreate(reason=RejectionReason.BAD_OUTLINE, contour_id=contour.id),
            username="owner", db=session))
        session.refresh(mask)

        state = image_status.status_for_image(session, image)
        assert state["phases"] == {"calibrate": FINISHED,      # untouched by a send-back
                                   "annotate": IN_PROGRESS,
                                   "review": IN_PROGRESS}
        assert state["status"] == IN_PROGRESS

    def test_status_for_mask_answers_for_the_whole_image(self, session, dataset):
        image = _image(session, dataset)
        mask = _mask(session, image)
        _contour(session, dataset, mask)

        state = image_status.status_for_mask(session, mask)
        assert state["phases"]["annotate"] == IN_PROGRESS
        assert state["phases"]["calibrate"] == NOT_STARTED


# ---------------------------------------------------------------------------
# Dataset roll-up
# ---------------------------------------------------------------------------

class TestDatasetProgress:
    def test_counts_images_without_masks(self, session, dataset):
        """Untouched images used to vanish from the totals: the roll-up joined
        masks, and an image nobody has opened has no mask row."""
        _image(session, dataset, "untouched.png")
        image = _image(session, dataset, "worked.png")
        mask = _mask(session, image)
        _contour(session, dataset, mask)

        counts, total = asyncio.run(
            datasets_db.get_annotation_progress_of_dataset(dataset.id, session))

        assert total == 2
        assert counts["annotate"] == {NOT_STARTED: 1, IN_PROGRESS: 1, FINISHED: 0}
        assert counts["calibrate"] == {NOT_STARTED: 2, IN_PROGRESS: 0, FINISHED: 0}
        assert counts["overall"] == {NOT_STARTED: 1, IN_PROGRESS: 1, FINISHED: 0}
        # Only review carries a blocked bucket, and the untouched image is in it.
        assert counts["review"] == {BLOCKED: 1, NOT_STARTED: 1, IN_PROGRESS: 0,
                                    FINISHED: 0}
        assert BLOCKED not in counts["calibrate"]
        assert BLOCKED not in counts["annotate"]
        assert BLOCKED not in counts["overall"]

    def test_every_phase_row_sums_to_the_image_count(self, session, dataset):
        for index in range(3):
            _image(session, dataset, f"{index}.png")

        counts, total = asyncio.run(
            datasets_db.get_annotation_progress_of_dataset(dataset.id, session))
        for phase, states in counts.items():
            assert sum(states.values()) == total, phase

    def test_listing_filters_by_phase(self, session, dataset):
        untouched = _image(session, dataset, "untouched.png")
        worked = _image(session, dataset, "worked.png")
        mask = _mask(session, worked)
        _contour(session, dataset, mask)

        rows = asyncio.run(datasets_db.get_image_and_mask_ids_of_dataset(
            dataset.id, session, filter_for_status=IN_PROGRESS,
            filter_for_phase="annotate"))
        assert [row["image_id"] for row in rows] == [worked.id]

        rows = asyncio.run(datasets_db.get_image_and_mask_ids_of_dataset(
            dataset.id, session, filter_for_status=NOT_STARTED,
            filter_for_phase="calibrate"))
        assert {row["image_id"] for row in rows} == {untouched.id, worked.id}

        # Without a phase the filter is on the combined status.
        rows = asyncio.run(datasets_db.get_image_and_mask_ids_of_dataset(
            dataset.id, session, filter_for_status=NOT_STARTED))
        assert [row["image_id"] for row in rows] == [untouched.id]

    def test_listing_includes_images_without_a_mask(self, session, dataset):
        image = _image(session, dataset)
        rows = asyncio.run(
            datasets_db.get_image_and_mask_ids_of_dataset(dataset.id, session))
        assert rows == [{
            "image_id": image.id,
            "mask_id": None,
            "status": NOT_STARTED,
            "phases": {"calibrate": NOT_STARTED, "annotate": NOT_STARTED,
                       "review": BLOCKED},
        }]
