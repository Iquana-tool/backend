"""Active-learning queue: Diversity, Density, Uncertainty and the job that fills their data."""
import numpy as np
import pytest
from iquana_toolbox.schemas.networking.http.services import EmbeddingVector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, import_models
from app.database.annotation_queues import AnnotationQueues
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.embeddings import EMBEDDING_DIM, upsert_embedding
from app.database.image_scores import ImageScores
from app.database.images import Frames, Images
from app.database.masks import Masks
from app.database.queue_prep_jobs import QueuePrepJobs
from app.database.stacks import Stacks
from app.database.users import Users
from app.services import queue_prep
from app.services.annotation_queue import build_and_save_queue
from config import EMBEDDING_MODEL_ID

import_models()


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'queue_prep.db'}")
    database.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([Users(username="ann", hashed_password="x"),
                     Users(username="bob", hashed_password="x")])
    session.flush()
    yield session
    session.close()


def make_dataset(db):
    ds = Datasets(name=f"ds{db.query(Datasets).count()}", description="", dataset_type="image",
                  folder_path="/tmp/ds", created_by="ann")
    db.add(ds)
    db.flush()
    return ds


def make_image(db, ds, name, *, state="not_started", author="ann", stack=None, frame_index=None):
    """state: 'none' (never opened, no mask), 'not_started', 'in_progress' or 'finished'."""
    # Frames are the polymorphic Frames subclass; the images table checks the pairing.
    image_cls = Frames if stack else Images
    img = image_cls(dataset_id=ds.id, file_name=name, file_path=f"/tmp/{name}",
                    thumbnail_file_path=f"/tmp/t_{name}", width=10, height=10, color_mode="RGB",
                    stack_id=stack.id if stack else None, frame_index=frame_index)
    db.add(img)
    db.flush()
    if state == "none":
        return img
    mask = Masks(image_id=img.id, fully_annotated=(state == "finished"), file_path=f"/tmp/m_{name}")
    db.add(mask)
    db.flush()
    if state in ("in_progress", "finished"):
        db.add(Contours(mask_id=mask.id, added_by="User", author_username=author,
                        confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0,
                        diameter=1.0, x=[0.1, 0.5, 0.5], y=[0.1, 0.1, 0.5]))
        db.flush()
    return img


def make_stack(db, ds):
    stack = Stacks(dataset_id=ds.id, name="scan", kind="oct", axis="z", frame_count=0,
                   thumbnail_file_path="/tmp/s.png", source_format="vol",
                   source_file_name="scan.vol")
    db.add(stack)
    db.flush()
    return stack


def add_vector(db, image, vec):
    full = np.zeros(EMBEDDING_DIM)
    full[:len(vec)] = vec
    upsert_embedding(db, image_id=image.id, kind="image_cls", model_id=EMBEDDING_MODEL_ID,
                     vector=full.tolist())


class FakeScoreClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def segment(self, request):
        self.requests.append(request)
        return self.response


class FakeEmbedClient:
    """Returns one image_cls vector per request, tagged with the configured model id."""

    def request_embeddings(self, request):
        return [EmbeddingVector(kind="image_cls", region_id=None, model_id=EMBEDDING_MODEL_ID,
                                dim=EMBEDDING_DIM, vector=[1.0] + [0.0] * (EMBEDDING_DIM - 1))]


def test_tables_exist_and_round_trip(db):
    ds = make_dataset(db)
    img = make_image(db, ds, "a.png")
    db.add(ImageScores(image_id=img.id, model_key="m1", score=0.7, num_detections=3,
                       confidences=[0.9, 0.5, 0.3]))
    db.add(QueuePrepJobs(dataset_id=ds.id, created_by="ann", strategy="uncertainty",
                         model_key="m1", image_ids=[img.id]))
    db.add(AnnotationQueues(dataset_id=ds.id, username="ann", strategy="uncertainty", model_key="m1",
                            image_order=[img.id], details={str(img.id): {"reason": "unsure"}}))
    db.commit()
    job = db.query(QueuePrepJobs).one()
    assert job.status == "pending"
    assert job.next_index == 0
    assert job.failed_image_ids == []
    assert db.query(ImageScores).one().confidences == [0.9, 0.5, 0.3]
    assert db.query(AnnotationQueues).one().details[str(img.id)]["reason"] == "unsure"


# -- Orderings -------------------------------------------------------------------

def score(db, image, confidences, model_key="m1"):
    db.add(ImageScores(image_id=image.id, model_key=model_key, score=0.0,
                       num_detections=len(confidences), confidences=confidences))


def test_uncertainty_queue_puts_most_borderline_first_with_tails_and_reasons(db):
    ds = make_dataset(db)
    sure = make_image(db, ds, "sure.png")
    torn = make_image(db, ds, "torn.png")
    empty = make_image(db, ds, "empty.png")
    unscored = make_image(db, ds, "unscored.png", state="none")
    others = make_image(db, ds, "others.png", state="in_progress", author="bob")
    mine = make_image(db, ds, "mine.png", state="in_progress", author="ann")
    done = make_image(db, ds, "done.png", state="finished")
    score(db, sure, [0.95])
    score(db, torn, [0.97, 0.5])        # one 50/50 object makes the image most torn
    score(db, empty, [])                # nothing found: not uncertain at all
    score(db, mine, [0.7])
    score(db, others, [0.5])
    db.commit()

    queue = build_and_save_queue(ds.id, "ann", "uncertainty", db, model_key="m1")

    assert queue.image_ids == [torn.id, mine.id, sure.id, empty.id, unscored.id,
                               others.id, done.id]
    assert queue.model_key == "m1"
    assert queue.details[str(torn.id)] == {"reason": "unsure", "score": 1.0}
    assert queue.details[str(unscored.id)]["reason"] == "not_scored"
    assert queue.details[str(others.id)]["reason"] == "started_by_other"
    assert queue.details[str(done.id)]["reason"] == "finished"


def test_stack_is_ranked_as_one_unit_by_its_most_unsure_frame(db):
    ds = make_dataset(db)
    plain = make_image(db, ds, "plain.png")
    stack = make_stack(db, ds)
    f1 = make_image(db, ds, "f1.png", stack=stack, frame_index=1)
    f0 = make_image(db, ds, "f0.png", stack=stack, frame_index=0)
    score(db, plain, [0.9])
    score(db, f0, [0.99])
    score(db, f1, [0.6])
    db.commit()

    queue = build_and_save_queue(ds.id, "ann", "uncertainty", db, model_key="m1")

    assert queue.image_ids == [f0.id, f1.id, plain.id]
    assert queue.details[str(f0.id)]["score"] == pytest.approx(0.971, abs=1e-3)


def test_diversity_queue_starts_far_from_finished_images(db):
    ds = make_dataset(db)
    done = make_image(db, ds, "done.png", state="finished")
    near = make_image(db, ds, "near.png")
    far = make_image(db, ds, "far.png")
    no_vec = make_image(db, ds, "novec.png")
    add_vector(db, done, [1, 0])
    add_vector(db, near, [0.99, 0.01])
    add_vector(db, far, [0, 1])
    db.commit()

    queue = build_and_save_queue(ds.id, "ann", "diversity", db)

    assert queue.image_ids == [far.id, near.id, no_vec.id, done.id]
    assert queue.details[str(far.id)]["reason"] == "varied"
    assert queue.details[str(no_vec.id)]["reason"] == "no_vector"


def test_density_queue_puts_typical_images_first(db):
    ds = make_dataset(db)
    odd = make_image(db, ds, "odd.png")
    common_a = make_image(db, ds, "a.png")
    common_b = make_image(db, ds, "b.png")
    common_c = make_image(db, ds, "c.png")
    done = make_image(db, ds, "done.png", state="finished")
    add_vector(db, odd, [0, 1])
    add_vector(db, common_a, [1, 0])
    add_vector(db, common_b, [0.99, 0.01])
    add_vector(db, common_c, [0.98, 0.02])
    add_vector(db, done, [1, 0])
    db.commit()

    queue = build_and_save_queue(ds.id, "ann", "density", db)

    assert set(queue.image_ids[:3]) == {common_a.id, common_b.id, common_c.id}
    assert queue.image_ids[3:] == [odd.id, done.id]
    assert queue.details[str(odd.id)]["reason"] == "typical"


# -- Preparation job ---------------------------------------------------------------

def test_missing_work_by_strategy(db):
    ds = make_dataset(db)
    open_img = make_image(db, ds, "open.png")
    done = make_image(db, ds, "done.png", state="finished")
    add_vector(db, done, [1, 0])
    score(db, open_img, [0.6])
    old = make_image(db, ds, "old.png")
    # A row from before confidences were kept counts as not scored.
    db.add(ImageScores(image_id=old.id, model_key="m1", score=0.4, num_detections=1))
    db.commit()

    # Vectors: everything without one. Scores: only unfinished images.
    assert queue_prep.missing_work(db, ds.id, "diversity", None) == [open_img.id, old.id]
    assert queue_prep.missing_work(db, ds.id, "density", None) == [open_img.id, old.id]
    assert queue_prep.missing_work(db, ds.id, "uncertainty", "m1") == [old.id]
    assert queue_prep.missing_work(db, ds.id, "uncertainty", "m2") == [open_img.id, old.id]
    assert queue_prep.missing_work(db, ds.id, "as_uploaded", None) == []


def test_start_prep_returns_none_when_ready_and_reuses_a_running_job(db):
    ds = make_dataset(db)
    make_image(db, ds, "a.png")
    db.commit()
    assert queue_prep.start_prep(db, ds.id, "ann", "as_uploaded", None, enqueue=False) is None

    first = queue_prep.start_prep(db, ds.id, "ann", "uncertainty", "m1", enqueue=False)
    second = queue_prep.start_prep(db, ds.id, "bob", "uncertainty", "m1", enqueue=False)
    assert first is not None
    assert second.id == first.id


def test_step_processes_each_image_then_finishes(db):
    ds = make_dataset(db)
    a = make_image(db, ds, "a.png", state="none")
    b = make_image(db, ds, "b.png", state="none")
    db.commit()
    job = queue_prep.start_prep(db, ds.id, "ann", "uncertainty", "m1", enqueue=False)
    score_client = FakeScoreClient({"result": [{"confidence": 0.6}]})

    states = [queue_prep.step(db, job.id, score_client=score_client) for _ in range(3)]

    assert states == ["running", "running", "succeeded"]
    assert db.query(ImageScores).filter_by(model_key="m1").count() == 2
    assert queue_prep.missing_work(db, ds.id, "uncertainty", "m1") == []
    assert sorted(job.image_ids) == sorted([a.id, b.id])

    vectors = queue_prep.start_prep(db, ds.id, "ann", "density", None, enqueue=False)
    for _ in range(3):
        queue_prep.step(db, vectors.id, embed_client=FakeEmbedClient())
    assert queue_prep.missing_work(db, ds.id, "density", None) == []


def test_step_records_failures_and_ends_partial(db):
    ds = make_dataset(db)
    make_image(db, ds, "a.png", state="none")
    db.commit()
    job = queue_prep.start_prep(db, ds.id, "ann", "uncertainty", "m1", enqueue=False)

    class Broken:
        def segment(self, request):
            raise ConnectionError("ai-service down")

    assert queue_prep.step(db, job.id, score_client=Broken()) == "running"
    assert queue_prep.step(db, job.id, score_client=Broken()) == "partial"
    assert len(db.get(QueuePrepJobs, job.id).failed_image_ids) == 1


def test_cancel_stops_the_job(db):
    ds = make_dataset(db)
    make_image(db, ds, "a.png", state="none")
    db.commit()
    job = queue_prep.start_prep(db, ds.id, "ann", "uncertainty", "m1", enqueue=False)
    queue_prep.cancel(db, job)
    assert queue_prep.step(db, job.id) == "cancelled"
    assert db.get(QueuePrepJobs, job.id).finished_at is not None
