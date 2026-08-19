"""Tests for conditioning dispatcher and resolvers (Stage 4)."""
import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.contours  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.users  # noqa: F401
import app.database.embeddings  # noqa: F401
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.embeddings import EMBEDDING_DIM, upsert_embedding
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.schemas.inference import ResolvedStep
from app.services.inference.conditioning_dispatcher import (
    dispatch_conditioning,
    is_image_fully_annotated,
    resolve_concept_text,
    resolve_embedding_conditioning,
    resolve_instance_conditioning,
    resolve_reference_images,
)
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.training import HyperParameter
from config import EMBEDDING_MODEL_ID

MODEL = EMBEDDING_MODEL_ID


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


def _axis(*components):
    v = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    for i, val in components:
        v[i] = val
    return v.tolist()


def _image(session, dataset_id, name, fully_annotated=False):
    img = Images(
        dataset_id=dataset_id, file_name=f"{name}.png", file_path=f"/data/{name}.png",
        thumbnail_file_path="/tmp/t.png", width=100, height=100, color_mode="RGB",
        scale_x=1.0, scale_y=1.0, unit="px"
    )
    session.add(img)
    session.flush()
    mask = Masks(image_id=img.id, fully_annotated=fully_annotated, file_path=f"/tmp/{name}_m.png")
    session.add(mask)
    session.flush()
    return img, mask


def _contour(session, mask_id, label_id=None):
    c = Contours(
        mask_id=mask_id, added_by="u", confidence_score=1.0, label_id=label_id,
        area=0.0, perimeter=0.0, circularity=0.0, diameter=0.0,
        x=[0.1, 0.6, 0.6], y=[0.1, 0.1, 0.6]
    )
    session.add(c)
    session.flush()
    return c


@pytest.fixture
def world(session):
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.flush()
    ds = Datasets(name="A", description="", dataset_type="image", folder_path="/tmp/A", created_by="u")
    session.add(ds)
    session.flush()
    label = Labels(dataset_id=ds.id, parent_id=None, name="coral", value=1)
    session.add(label)
    session.flush()

    target, target_mask = _image(session, ds.id, "target", fully_annotated=False)
    near, near_mask = _image(session, ds.id, "near", fully_annotated=True)
    mid, mid_mask = _image(session, ds.id, "mid", fully_annotated=False)
    far, far_mask = _image(session, ds.id, "far", fully_annotated=True)

    upsert_embedding(session, image_id=target.id, kind="image_cls", model_id=MODEL, vector=_axis((0, 1.0)))
    upsert_embedding(session, image_id=near.id, kind="image_cls", model_id=MODEL, vector=_axis((0, 1.0), (1, 0.1)))
    upsert_embedding(session, image_id=mid.id, kind="image_cls", model_id=MODEL, vector=_axis((0, 1.0), (1, 0.2)))
    upsert_embedding(session, image_id=far.id, kind="image_cls", model_id=MODEL, vector=_axis((1, 1.0)))

    return dict(
        session=session, ds=ds, label=label, target=target, target_mask=target_mask,
        near=near, near_mask=near_mask, mid=mid, mid_mask=mid_mask, far=far, far_mask=far_mask
    )


def test_is_image_fully_annotated(world):
    s = world["session"]
    assert is_image_fully_annotated(s, world["near"].id) is True
    assert is_image_fully_annotated(s, world["mid"].id) is False
    assert is_image_fully_annotated(s, world["target"].id) is False


def test_resolve_reference_images_enforces_completeness_and_max_units(world):
    s = world["session"]
    # Seed contours
    c_near1 = _contour(s, world["near_mask"].id, label_id=world["label"].id)
    c_near2 = _contour(s, world["near_mask"].id, label_id=world["label"].id)
    c_mid = _contour(s, world["mid_mask"].id, label_id=world["label"].id)
    c_far = _contour(s, world["far_mask"].id, label_id=world["label"].id)

    # 1. With complete annotation required and max_images=1 (SAM 3 contract):
    exemplars, matches = resolve_reference_images(
        s,
        target_image_id=world["target"].id,
        dataset_id=world["ds"].id,
        strategy="global_scene",
        concept_label_id=world["label"].id,
        max_images=1,
        requires_complete_annotation=True,
        model_id=MODEL,
    )
    # Only `near` image is selected (since mid is incomplete, and far is second choice)
    # With max_images=1, only the top exemplar from the selected reference image is materialized
    assert len(exemplars) == 1
    assert exemplars[0].image_url == "/data/near.png"
    assert len(matches) == 1
    assert matches[0].contour_id in {c_near1.id, c_near2.id}

    # 2. With complete annotation NOT required:
    exemplars_any, matches_any = resolve_reference_images(
        s,
        target_image_id=world["target"].id,
        dataset_id=world["ds"].id,
        strategy="global_scene",
        concept_label_id=world["label"].id,
        max_images=2,
        requires_complete_annotation=False,
        model_id=MODEL,
    )
    # Both near and mid selected
    urls = {ex.image_url for ex in exemplars_any}
    assert urls == {"/data/near.png", "/data/mid.png"}


def test_resolve_instances(world):
    s = world["session"]
    c1 = _contour(s, world["near_mask"].id, label_id=world["label"].id)
    c2 = _contour(s, world["far_mask"].id, label_id=world["label"].id)

    cids, positive_exemplars, cross_image_exemplars = resolve_instance_conditioning(
        s,
        dataset_id=world["ds"].id,
        concept_label_id=world["label"].id,
        max_instances=1,
    )
    assert len(cids) == 1
    assert len(positive_exemplars) == 1
    assert len(cross_image_exemplars) == 1
    assert cids[0] in {c1.id, c2.id}
    assert cross_image_exemplars[0].image_url in {"/data/near.png", "/data/far.png"}
    assert cross_image_exemplars[0].mask is not None


def test_resolve_embeddings(world):
    s = world["session"]
    vec = resolve_embedding_conditioning(
        s,
        target_image_id=world["target"].id,
        kind="image_cls",
        model_id=MODEL,
    )
    assert vec is not None
    assert len(vec) == EMBEDDING_DIM
    assert vec[0] == 1.0


def test_resolve_concept_text():
    # Provided directly in inputs
    assert resolve_concept_text(step_inputs={"conditioning": {"concept_text": "custom text"}}) == "custom text"
    # Fallback to label
    lbl = Labels(id=1, name="coral_label", value=1)
    assert resolve_concept_text(step_inputs={}, label=lbl) == "coral_label"


def test_dispatch_conditioning_all_kinds(world):
    s = world["session"]
    c_near = _contour(s, world["near_mask"].id, label_id=world["label"].id)

    # 1. Kind: none
    step_none = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="m2f",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="Mask2Former",
        inputs={"conditioning": {"count": 0, "strategy": None, "concept_text": None}, "parameters": {"threshold": 0.5}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
            parameters=[HyperParameter(key="threshold", label="T", type="float", default_value=0.5)],
        ),
    )
    res_none = dispatch_conditioning(s, step_none, world["target"])
    assert res_none == {"kind": "none"}

    # 2. Kind: concept_text
    step_text = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="clipseg",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="CLIPSeg",
        inputs={"conditioning": {"count": 0, "strategy": None, "concept_text": "vibrant coral"}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(kind="concept_text", user_selectable_count=False),
            parameters=[],
        ),
    )
    res_text = dispatch_conditioning(s, step_text, world["target"])
    assert res_text["kind"] == "concept_text"
    assert res_text["concept_text"] == "vibrant coral"

    # 3. Kind: reference_images
    step_ref = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="sam3",
        task="cross-image-suggestion",
        level=0,
        label_name="coral",
        model_name="SAM 3",
        inputs={"conditioning": {"count": 1, "strategy": "global_scene", "concept_text": None}, "parameters": {"threshold": 0.3}},
        input_contract=InputContract(
            task="cross-image-suggestion",
            conditioning=ConditioningSpec(
                kind="reference_images", unit="image", min_units=1, max_units=1,
                requires_complete_annotation=True, user_selectable_count=False,
            ),
            parameters=[HyperParameter(key="threshold", label="T", type="float", default_value=0.3)],
        ),
    )
    res_ref = dispatch_conditioning(s, step_ref, world["target"])
    assert res_ref["kind"] == "reference_images"
    assert len(res_ref["exemplars"]) == 1
    assert res_ref["exemplars"][0].image_url == "/data/near.png"

    # 4. Kind: instances
    step_inst = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="sam2",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="SAM 2",
        inputs={"conditioning": {"count": 2, "strategy": None, "concept_text": None}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(kind="instances", unit="instance", min_units=1, max_units=5, user_selectable_count=True),
            parameters=[],
        ),
    )
    res_inst = dispatch_conditioning(s, step_inst, world["target"])
    assert res_inst["kind"] == "instances"
    assert len(res_inst["contour_ids"]) == 1
    assert res_inst["contour_ids"][0] == c_near.id

    # 5. Kind: embeddings
    step_emb = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="embedder",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="Embedder",
        inputs={"conditioning": {"count": 0, "strategy": None, "concept_text": None}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(kind="embeddings", unit="vector", embedding_kinds=["image_cls"], user_selectable_count=False),
            parameters=[],
        ),
    )
    res_emb = dispatch_conditioning(s, step_emb, world["target"])
    assert res_emb["kind"] == "embeddings"
    assert res_emb["vector"] is not None


def test_dispatch_conditioning_unsupported_kind_raises(world):
    s = world["session"]
    step_bad = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="m",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="M",
        inputs={"conditioning": {}, "parameters": {}},
        input_contract=InputContract.model_construct(
            task="instance-segmentation",
            conditioning=ConditioningSpec.model_construct(kind="unsupported_kind", user_selectable_count=False),
            parameters=[],
        ),
    )
    with pytest.raises(ValueError, match="Unsupported conditioning kind 'unsupported_kind'"):
        dispatch_conditioning(s, step_bad, world["target"])


def test_dispatch_conditioning_multi_vector_embeddings(world):
    s = world["session"]
    # Seed image_cls on image, and region_mean on a contour on the target image (production schema behavior)
    c = _contour(s, world["target_mask"].id, label_id=world["label"].id)
    upsert_embedding(s, contour_id=c.id, kind="region_mean", model_id=MODEL, vector=_axis((1, 0.5)))

    step_multi_emb = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="multi_embedder",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="MultiEmbedder",
        inputs={"conditioning": {"count": 0, "strategy": None, "concept_text": None}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(
                kind="embeddings",
                unit="vector",
                embedding_kinds=["image_cls", "region_mean"],
                user_selectable_count=False,
            ),
            parameters=[],
        ),
    )
    res = dispatch_conditioning(s, step_multi_emb, world["target"])
    assert res["kind"] == "embeddings"
    assert "image_cls" in res["vectors"]
    assert "region_mean" in res["vectors"]
    assert len(res["vectors"]["image_cls"]) == EMBEDDING_DIM
    assert len(res["vectors"]["region_mean"]) == EMBEDDING_DIM


def test_dispatch_conditioning_region_mean_explicit_query_contour_id(world):
    """Explicit query_contour_id selects precisely that contour's embedding vector."""
    s = world["session"]
    c1 = _contour(s, world["target_mask"].id, label_id=world["label"].id)
    c2 = _contour(s, world["target_mask"].id, label_id=world["label"].id)
    v1 = _axis((1, 0.2))
    v2 = _axis((1, 0.8))
    upsert_embedding(s, contour_id=c1.id, kind="region_mean", model_id=MODEL, vector=v1)
    upsert_embedding(s, contour_id=c2.id, kind="region_mean", model_id=MODEL, vector=v2)

    step = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="emb_model",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="EmbModel",
        inputs={"conditioning": {"count": 0, "strategy": None, "concept_text": None, "query_contour_id": c2.id}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(
                kind="embeddings",
                unit="vector",
                embedding_kinds=["region_mean"],
                user_selectable_count=False,
            ),
            parameters=[],
        ),
    )
    res = dispatch_conditioning(s, step, world["target"])
    assert res["vectors"]["region_mean"] == v2


def test_dispatch_conditioning_query_contour_outside_dataset_rejected(world):
    """A query_contour_id from another dataset is rejected and not resolved."""
    s = world["session"]
    other_ds = Datasets(name="B", description="", dataset_type="image", folder_path="/tmp/B", created_by="u")
    s.add(other_ds)
    s.flush()
    other_img, other_mask = _image(s, dataset_id=other_ds.id, name="other")
    c_foreign = _contour(s, other_mask.id, label_id=world["label"].id)
    upsert_embedding(s, contour_id=c_foreign.id, kind="region_mean", model_id=MODEL, vector=_axis((1, 0.5)))

    # Attempt to dispatch on world["target"] (which is dataset_id=1) using c_foreign.id
    step = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="emb_model",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="EmbModel",
        inputs={"conditioning": {"count": 0, "strategy": None, "concept_text": None, "query_contour_id": c_foreign.id}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(
                kind="embeddings",
                unit="vector",
                embedding_kinds=["region_mean"],
                user_selectable_count=False,
            ),
            parameters=[],
        ),
    )
    with pytest.raises(ValueError, match="missing required embedding.*region_mean"):
        dispatch_conditioning(s, step, world["target"])



def test_dispatch_conditioning_region_mean_deterministic_ordering_fallback(world):
    """When query_contour_id is omitted, deterministic ordering selects the highest priority contour."""
    s = world["session"]
    c_older = _contour(s, world["target_mask"].id, label_id=world["label"].id)
    c_newer = _contour(s, world["target_mask"].id, label_id=world["label"].id)
    v_older = _axis((1, 0.1))
    v_newer = _axis((1, 0.9))
    upsert_embedding(s, contour_id=c_older.id, kind="region_mean", model_id=MODEL, vector=v_older)
    upsert_embedding(s, contour_id=c_newer.id, kind="region_mean", model_id=MODEL, vector=v_newer)

    step = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="emb_model",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="EmbModel",
        inputs={"conditioning": {"count": 0, "strategy": None, "concept_text": None}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(
                kind="embeddings",
                unit="vector",
                embedding_kinds=["region_mean"],
                user_selectable_count=False,
            ),
            parameters=[],
        ),
    )
    res = dispatch_conditioning(s, step, world["target"])
    # Deterministic order by id / created_at selects c_newer
    assert res["vectors"]["region_mean"] == v_newer


def test_dispatch_conditioning_missing_embedding_raises(world):
    s = world["session"]
    # Request a kind that is not present on target
    step_missing_emb = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="custom_embedder",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="CustomEmbedder",
        inputs={"conditioning": {"count": 0, "strategy": None, "concept_text": None}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(
                kind="embeddings",
                unit="vector",
                embedding_kinds=["non_existent_kind"],
                user_selectable_count=False,
            ),
            parameters=[],
        ),
    )
    with pytest.raises(ValueError, match="missing required embedding.*non_existent_kind"):
        dispatch_conditioning(s, step_missing_emb, world["target"])


def test_dispatch_conditioning_post_resolution_min_units_reference_images_raises(world):
    s = world["session"]
    # Seed 1 eligible contour on `near` (which is fully_annotated). `mid` is not fully annotated.
    _contour(s, world["near_mask"].id, label_id=world["label"].id)

    # Contract requires min_units=2 unique reference images, but only 1 eligible exists
    step_ref_min2 = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="sam3",
        task="cross-image-suggestion",
        level=0,
        label_name="coral",
        model_name="SAM 3",
        inputs={"conditioning": {"count": 2, "strategy": "global_scene", "concept_text": None}, "parameters": {"threshold": 0.3}},
        input_contract=InputContract(
            task="cross-image-suggestion",
            conditioning=ConditioningSpec(
                kind="reference_images", unit="image", min_units=2, max_units=5,
                requires_complete_annotation=True, user_selectable_count=True,
            ),
            parameters=[HyperParameter(key="threshold", label="T", type="float", default_value=0.3)],
        ),
    )
    with pytest.raises(ValueError, match="Resolved 1 unique reference image.*requires at least 2 min_units"):
        dispatch_conditioning(s, step_ref_min2, world["target"])


def test_dispatch_conditioning_post_resolution_min_units_instances_raises(world):
    s = world["session"]
    # Seed 1 contour
    _contour(s, world["near_mask"].id, label_id=world["label"].id)

    step_inst_min3 = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="sam2",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="SAM 2",
        inputs={"conditioning": {"count": 3, "strategy": None, "concept_text": None}, "parameters": {}},
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(kind="instances", unit="instance", min_units=3, max_units=5, user_selectable_count=True),
            parameters=[],
        ),
    )
    with pytest.raises(ValueError, match="Resolved 1 instance exemplar.*requires at least 3 min_units"):
        dispatch_conditioning(s, step_inst_min3, world["target"])


def test_predict_cross_image_with_concept_text_and_parameters(world, monkeypatch):
    """Worker _predict_cross_image forwards concept_text, parameters, and conditioning to AI service."""
    from app.services.inference.execution import _predict_cross_image

    captured_requests = []

    class MockCrossImageService:
        async def inference(self, request):
            captured_requests.append(request)
            return {"result": []}

    monkeypatch.setattr(
        "app.services.ai_services.cross_image.CrossImageService",
        lambda: MockCrossImageService(),
    )

    s = world["session"]

    # 1. Concept text model on cross-image-suggestion with normalized parameters
    step_text = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="text_seg",
        task="cross-image-suggestion",
        level=0,
        label_name="coral",
        model_name="TextSeg",
        inputs={
            "conditioning": {"count": 0, "strategy": None, "concept_text": "sea coral"},
            "parameters": {"threshold": 0.45, "mask_threshold": 0.6},
        },
        input_contract=InputContract(
            task="cross-image-suggestion",
            conditioning=ConditioningSpec(kind="concept_text", user_selectable_count=False),
            parameters=[
                HyperParameter(key="threshold", label="T", type="float", default_value=0.3),
                HyperParameter(key="mask_threshold", label="M", type="float", default_value=0.5),
            ],
        ),
    )
    res_text = _predict_cross_image(s, step_text, world["target"], "u")
    assert res_text == []
    assert len(captured_requests) == 1
    req = captured_requests[-1]
    assert req.exemplars == []
    assert req.concept.name == "sea coral"
    assert req.parameters == {"threshold": 0.45, "mask_threshold": 0.6}


def test_predict_instance_segmentation_forwards_parameters_and_conditioning(world, monkeypatch):
    """Worker _predict_instance_segmentation dispatches conditioning and forwards parameters/contour_ids/embeddings."""
    from app.services.inference.execution import _predict_instance_segmentation

    captured_requests = []

    class MockInstanceSegmentationService:
        async def inference(self, request):
            captured_requests.append(request)
            return {"result": []}

    monkeypatch.setattr(
        "app.services.ai_services.instance_segmentation.InstanceSegmentationService",
        lambda: MockInstanceSegmentationService(),
    )

    s = world["session"]
    c = _contour(s, world["near_mask"].id, label_id=world["label"].id)

    # 1. Instance-segmentation with instance conditioning and hyperparameters
    step_inst = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="sam2",
        task="instance-segmentation",
        level=0,
        label_name="coral",
        model_name="SAM 2",
        inputs={
            "conditioning": {"count": 1, "strategy": None, "concept_text": None},
            "parameters": {"threshold": 0.75, "min_target_frac": 0.2},
        },
        input_contract=InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(kind="instances", unit="instance", min_units=1, max_units=5, user_selectable_count=True),
            parameters=[
                HyperParameter(key="threshold", label="T", type="float", default_value=0.5),
                HyperParameter(key="min_target_frac", label="F", type="float", default_value=0.5),
            ],
        ),
    )
    res = _predict_instance_segmentation(s, step_inst, world["target"], "u")
    assert res == []
    assert len(captured_requests) == 1
    req = captured_requests[-1]
    assert req.parameters == {"threshold": 0.75, "min_target_frac": 0.2}
    assert req.contour_ids == [c.id]
    assert req.model_registry_key == "sam2"
    assert len(req.positive_exemplars) == 1


def test_resolve_reference_images_progressive_search_exhaustion(world, monkeypatch):
    """Progressive search continues past 1024 candidates until retrieval is exhausted or enough images are found."""
    from app.services.exemplar_retrieval import ExemplarMatch

    s = world["session"]
    calls = []

    # Mock retrieve_exemplars to simulate candidate 1200 being the first fully annotated image
    img_far = world["far"]
    c_far = _contour(s, world["far_mask"].id, label_id=world["label"].id)

    def mock_retrieve_exemplars(session, strategy, query, model_id=None):
        calls.append(query.top_k)
        # For small top_k, return non-fully-annotated candidate images
        # When top_k reaches 2048, include the far image which is fully annotated
        if query.top_k < 2048:
            # 128 incomplete images
            return [
                ExemplarMatch(contour_id=c_far.id, image_id=world["mid"].id, score=0.9 - i * 0.001)
                for i in range(min(query.top_k, 1500))
            ]
        else:
            matches = [
                ExemplarMatch(contour_id=c_far.id, image_id=world["mid"].id, score=0.9 - i * 0.001)
                for i in range(1200)
            ]
            matches.append(ExemplarMatch(contour_id=c_far.id, image_id=img_far.id, score=0.1))
            return matches

    monkeypatch.setattr(
        "app.services.inference.conditioning_dispatcher.retrieve_exemplars",
        mock_retrieve_exemplars,
    )

    exemplars, matches = resolve_reference_images(
        s,
        target_image_id=world["target"].id,
        dataset_id=world["ds"].id,
        strategy="global_scene",
        concept_label_id=world["label"].id,
        max_images=1,
        requires_complete_annotation=True,
        model_id=MODEL,
    )
    assert len(exemplars) == 1
    assert exemplars[0].image_url == "/data/far.png"
    # Verify search expanded past 1024
    assert any(k > 1024 for k in calls)


def test_predict_cross_image_forwards_source_aware_exemplars_and_positive_exemplars(world, monkeypatch):
    """_predict_cross_image forwards exemplars, positive_exemplars, and contour_ids."""
    from app.services.inference.execution import _predict_cross_image

    captured_requests = []

    class MockCrossImageService:
        async def inference(self, request):
            captured_requests.append(request)
            return {"result": []}

    monkeypatch.setattr(
        "app.services.ai_services.cross_image.CrossImageService",
        lambda: MockCrossImageService(),
    )

    s = world["session"]
    c = _contour(s, world["near_mask"].id, label_id=world["label"].id)

    step = ResolvedStep(
        label_id=world["label"].id,
        model_registry_key="cross_model",
        task="cross-image-suggestion",
        level=0,
        label_name="coral",
        model_name="CrossModel",
        inputs={
            "conditioning": {"count": 1, "strategy": "global_scene", "concept_text": None},
            "parameters": {},
        },
        input_contract=InputContract(
            task="cross-image-suggestion",
            conditioning=ConditioningSpec(
                kind="instances", unit="instance", min_units=1, max_units=5, user_selectable_count=True
            ),
            parameters=[],
        ),
    )
    res = _predict_cross_image(s, step, world["target"], "u")
    assert res == []
    assert len(captured_requests) == 1
    req = captured_requests[-1]
    assert len(req.exemplars) == 1
    assert req.exemplars[0].image_url == "/data/near.png"
    assert req.exemplars[0].mask is not None
    assert len(req.positive_exemplars) == 1
    assert req.contour_ids == [c.id]
