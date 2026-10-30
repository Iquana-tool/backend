"""Active-learning queue: the vector and model methods, filters, round robin, and the job
that prepares their data."""
import pytest
from iquana_toolbox.schemas.networking.http.services import EmbeddingVector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, import_models
from app.database.annotation_queues import AnnotationQueues
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.image_scores import ImageScores
from app.database.image_vectors import ImageVectors
from app.database.images import Frames, Images
from app.database.masks import Masks
from app.database.queue_prep_jobs import QueuePrepJobs
from app.database.stacks import Stacks
from app.database.users import Users
from app.schemas.annotation_queue import QueueSettings
from app.services import queue_prep
from app.services.annotation_queue import build_and_save_queue

import_models()

DINO = {"embedding_model": "dinov3"}
M1 = {"trained_model": "m1"}


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


def vector(db, image, values, model_key="dinov3"):
    db.add(ImageVectors(image_id=image.id, model_key=model_key, vector=list(values)))


def score(db, image, confidences, model_key="m1"):
    db.add(ImageScores(image_id=image.id, model_key=model_key, score=0.0,
                       num_detections=len(confidences), confidences=confidences))


def build(db, ds, method, **kwargs):
    return build_and_save_queue(ds.id, "ann", QueueSettings(method=method, **kwargs), db)


class FakeScoreClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def segment(self, request):
        self.requests.append(request)
        return self.response


class FakeEmbedClient:
    """Returns a 3-number image_cls vector: embedding models may have any length."""

    def __init__(self):
        self.requests = []

    def request_embeddings(self, request):
        self.requests.append(request)
        return [EmbeddingVector(kind="image_cls", region_id=None, model_id="x", dim=3,
                                vector=[1.0, 0.0, 0.0])]


def test_tables_round_trip(db):
    ds = make_dataset(db)
    img = make_image(db, ds, "a.png")
    vector(db, img, [0.5] * 1280, model_key="ijepa")
    db.add(QueuePrepJobs(dataset_id=ds.id, created_by="ann", needs=M1, image_ids=[img.id]))
    db.add(AnnotationQueues(dataset_id=ds.id, username="ann", strategy="uncertainty",
                            options={"method": "uncertainty"}, image_order=[img.id]))
    db.commit()
    job = db.query(QueuePrepJobs).one()
    assert (job.status, job.next_index, job.failed_image_ids) == ("pending", 0, [])
    assert len(db.query(ImageVectors).one().vector) == 1280
    assert db.query(AnnotationQueues).one().show_builder is True


# -- Methods --------------------------------------------------------------------------

def test_uncertainty_puts_most_borderline_first_with_tails_and_reasons(db):
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

    queue = build(db, ds, "uncertainty", trained_model="m1")

    assert queue.image_ids == [torn.id, mine.id, sure.id, empty.id, unscored.id,
                               others.id, done.id]
    assert queue.details[str(torn.id)] == {"reason": "uncertainty", "score": 1.0}
    assert queue.details[str(unscored.id)]["reason"] == "no_data"
    assert queue.details[str(others.id)]["reason"] == "started_by_other"
    assert queue.details[str(done.id)]["reason"] == "finished"


def test_ascending_flips_a_method(db):
    ds = make_dataset(db)
    sure = make_image(db, ds, "sure.png")
    torn = make_image(db, ds, "torn.png")
    score(db, sure, [0.95])
    score(db, torn, [0.5])
    db.commit()
    queue = build(db, ds, "uncertainty", trained_model="m1", direction="ascending")
    assert queue.image_ids == [sure.id, torn.id]


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

    queue = build(db, ds, "uncertainty", trained_model="m1")

    assert queue.image_ids == [f0.id, f1.id, plain.id]
    assert queue.details[str(f0.id)]["score"] == pytest.approx(0.971, abs=1e-3)


def test_diversity_starts_far_from_finished_images(db):
    ds = make_dataset(db)
    done = make_image(db, ds, "done.png", state="finished")
    near = make_image(db, ds, "near.png")
    far = make_image(db, ds, "far.png")
    no_vec = make_image(db, ds, "novec.png")
    vector(db, done, [1, 0])
    vector(db, near, [0.99, 0.01])
    vector(db, far, [0, 1])
    db.commit()

    queue = build(db, ds, "diversity", embedding_model="dinov3")

    assert queue.image_ids == [far.id, near.id, no_vec.id, done.id]
    assert queue.details[str(no_vec.id)]["reason"] == "no_data"


def test_density_puts_typical_images_first_with_any_vector_length(db):
    ds = make_dataset(db)
    odd = make_image(db, ds, "odd.png")
    commons = [make_image(db, ds, f"{n}.png") for n in "abc"]
    vector(db, odd, [0, 1, 0, 0, 0], model_key="ijepa")
    for image, x in zip(commons, (1.0, 0.99, 0.98)):
        vector(db, image, [x, 1 - x, 0, 0, 0], model_key="ijepa")
    db.commit()

    queue = build(db, ds, "density", embedding_model="ijepa")

    assert set(queue.image_ids[:3]) == {c.id for c in commons}
    assert queue.image_ids[3] == odd.id


def test_round_robin_alternates_two_methods(db):
    ds = make_dataset(db)
    a, b, c, d = (make_image(db, ds, f"{n}.png") for n in "abcd")
    for image, confidences in ((a, [0.5]), (b, [0.6]), (c, [0.9]), (d, [0.95])):
        score(db, image, confidences)
    db.commit()

    # Uncertainty picks a, then As uploaded (ascending) picks the first not yet taken: b...
    queue = build(db, ds, "uncertainty", trained_model="m1", round_robin_with="as_uploaded")
    assert queue.image_ids == [a.id, b.id, c.id, d.id]
    # ...and with As uploaded descending first, the two lists meet in the middle.
    queue = build(db, ds, "as_uploaded", direction="descending",
                  round_robin_with="uncertainty", trained_model="m1")
    assert queue.image_ids == [d.id, a.id, c.id, b.id]
    assert queue.details[str(a.id)]["reason"] == "uncertainty"


def test_filters_keep_matching_images_only(db, monkeypatch):
    ds = make_dataset(db)
    low, high, other = (make_image(db, ds, f"{n}.png") for n in ("low", "high", "other"))
    score(db, low, [0.3])
    score(db, high, [0.9])
    score(db, other, [0.2])
    db.commit()
    import app.services.annotation_queue as aq
    monkeypatch.setattr(aq, "filter_image_ids", lambda _db, _ds, _f: [low.id, high.id])

    queue = build(db, ds, "as_uploaded", trained_model="m1", max_confidence=0.5,
                  metadata_filters={"site": ["reef_a"]})

    assert queue.image_ids == [low.id]


# -- Preparation job ---------------------------------------------------------------

def test_missing_work_by_need(db):
    ds = make_dataset(db)
    open_img = make_image(db, ds, "open.png")
    done = make_image(db, ds, "done.png", state="finished")
    vector(db, done, [1, 0])
    score(db, open_img, [0.6])
    old = make_image(db, ds, "old.png")
    # A row from before confidences were kept counts as not scored.
    db.add(ImageScores(image_id=old.id, model_key="m1", score=0.4, num_detections=1))
    db.commit()

    assert queue_prep.missing_work(db, ds.id, DINO) == [open_img.id, old.id]
    assert queue_prep.missing_work(db, ds.id, {"embedding_model": "ijepa"}) == [
        open_img.id, done.id, old.id]
    assert queue_prep.missing_work(db, ds.id, M1) == [old.id]
    assert queue_prep.missing_work(db, ds.id, {}) == []


def test_start_prep_returns_none_when_ready_and_reuses_a_running_job(db):
    ds = make_dataset(db)
    make_image(db, ds, "a.png")
    db.commit()
    assert queue_prep.start_prep(db, ds.id, "ann", {}, enqueue=False) is None

    first = queue_prep.start_prep(db, ds.id, "ann", M1, enqueue=False)
    second = queue_prep.start_prep(db, ds.id, "bob", M1, enqueue=False)
    third = queue_prep.start_prep(db, ds.id, "bob", DINO, enqueue=False)
    assert second.id == first.id
    assert third.id != first.id


def test_step_fills_vectors_and_scores_then_finishes(db):
    ds = make_dataset(db)
    a = make_image(db, ds, "a.png", state="none")
    b = make_image(db, ds, "b.png", state="none")
    db.commit()
    job = queue_prep.start_prep(db, ds.id, "ann", {**DINO, **M1}, enqueue=False)
    embed_client = FakeEmbedClient()
    score_client = FakeScoreClient({"result": [{"confidence": 0.6}]})

    states = [queue_prep.step(db, job.id, embed_client=embed_client, score_client=score_client)
              for _ in range(3)]

    assert states == ["running", "running", "succeeded"]
    assert embed_client.requests[0].model_registry_key == "dinov3"
    assert db.query(ImageVectors).filter_by(model_key="dinov3").count() == 2
    assert db.query(ImageScores).filter_by(model_key="m1").count() == 2
    assert queue_prep.missing_work(db, ds.id, {**DINO, **M1}) == []
    assert sorted(job.image_ids) == sorted([a.id, b.id])


def test_step_records_failures_and_ends_partial(db):
    ds = make_dataset(db)
    make_image(db, ds, "a.png", state="none")
    db.commit()
    job = queue_prep.start_prep(db, ds.id, "ann", M1, enqueue=False)

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
    job = queue_prep.start_prep(db, ds.id, "ann", M1, enqueue=False)
    queue_prep.cancel(db, job)
    assert queue_prep.step(db, job.id) == "cancelled"
    assert db.get(QueuePrepJobs, job.id).finished_at is not None


# -- Review fixes: cancel mid-image, stale jobs, ownership, balance by metadata --------

def test_cancel_landing_mid_image_is_not_overwritten(db, monkeypatch):
    """The worker must not write "running" back over a cancel the API set meanwhile."""
    from sqlalchemy.orm import sessionmaker
    ds = make_dataset(db)
    imgs = [make_image(db, ds, f"c{i}.png") for i in range(3)]
    job = QueuePrepJobs(dataset_id=ds.id, created_by="ann", needs=DINO,
                        image_ids=[i.id for i in imgs], status="running")
    db.add(job)
    db.commit()
    api = sessionmaker(bind=db.get_bind())()

    def unit_while_cancelled(session, j, image_id, *_):
        queue_prep.cancel(api, api.get(QueuePrepJobs, j.id))

    monkeypatch.setattr(queue_prep, "_run_unit", unit_while_cancelled)
    queue_prep.step(db, job.id)
    api.expire_all()
    assert api.get(QueuePrepJobs, job.id).status == "cancelling"
    assert queue_prep.step(db, job.id) == "cancelled"
    api.close()


def test_stale_job_is_given_up_and_restarted(db):
    from datetime import datetime, timedelta, timezone
    ds = make_dataset(db)
    make_image(db, ds, "s.png")
    old = datetime.now(timezone.utc) - queue_prep.STALE_AFTER - timedelta(minutes=1)
    stuck = QueuePrepJobs(dataset_id=ds.id, created_by="ann", needs=DINO, image_ids=[1],
                          status="running", created_at=old, heartbeat_at=old)
    db.add(stuck)
    db.commit()
    fresh = queue_prep.start_prep(db, ds.id, "bob", DINO, enqueue=False)
    assert fresh.id != stuck.id
    db.refresh(stuck)
    assert stuck.status == "failed" and "stopped responding" in stuck.error


def test_live_job_is_shared_and_only_its_starter_may_cancel(db):
    ds = make_dataset(db)
    make_image(db, ds, "l.png")
    first = queue_prep.start_prep(db, ds.id, "ann", DINO, enqueue=False)
    assert queue_prep.start_prep(db, ds.id, "bob", DINO, enqueue=False).id == first.id
    assert queue_prep.can_cancel(first, "ann")
    assert not queue_prep.can_cancel(first, "bob")


def test_balance_by_metadata_takes_turns_between_values(db):
    from app.database.image_metadata import ImageMetadata
    ds = make_dataset(db)
    species = ["apal", "apal", "apal", "ofav", "ofav", "ssid"]
    imgs = [make_image(db, ds, f"b{i}.png", state="none") for i in range(len(species))]
    for img, value in zip(imgs, species):
        db.add(ImageMetadata(image_id=img.id, key="species", value=value))
    db.commit()
    settings = QueueSettings(method="as_uploaded", balance_by="species")
    queue = build_and_save_queue(ds.id, "ann", settings, db)
    by_id = {img.id: value for img, value in zip(imgs, species)}
    assert [by_id[i] for i in queue.image_ids] == ["apal", "ofav", "ssid", "apal", "ofav", "apal"]
    # Within a value the method's order holds.
    assert [i for i in queue.image_ids if by_id[i] == "apal"] == [imgs[0].id, imgs[1].id, imgs[2].id]
