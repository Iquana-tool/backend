"""Derived metrics: formulas a dataset declares over its direct measurements.

The two examples that motivated them are the fixture:

  * an OCT-style hierarchy -- Choroid objects with Vessel children -- for the choroidal
    vascularity index
    ``sum(area of Vessel children) / area``,
  * a Coral with Polyp children for polyp density ``count(Polyp children) / area``.

What has to hold:

  * the numbers are right, including the empty case (a choroid with no vessels has a
    vascularity of 0, not a missing value), and a label filter means exactly that label --
    the label tree is containment, so "Choroid" does not also match the Vessels under it,
  * units follow from the formula -- a ratio is unitless, a count per area is "/mm²" --
    and formulas whose units make no sense are refused,
  * derived values only see the objects the page's filters let through,
  * they reach the summary, the per-object export and the grouped summary,
  * definitions are checked as a set: cycles, unknown inputs and deleted labels are
    reported, and the HTTP surface enforces who may change them.
"""
import asyncio

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session
import app.database.dataset_members  # noqa: F401
import app.database.dataset_metadata_keys  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.derived_metrics  # noqa: F401
import app.database.image_metadata  # noqa: F401
import app.database.images  # noqa: F401
import app.database.quantification_profiles  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.routes.general.datasets import router as datasets_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole
from app.services import derived_metrics as dm
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import derived_metrics as derived_db
from app.services.database_access import image_metadata as meta
from app.services.quantification import GEOMETRY_METRIC_KEYS


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def _cvi(choroid, vessel):
    return dm.DerivedMetricDefinition(
        key="cvi", name="Choroidal vascularity index", label_ids=[choroid],
        expression={"type": "binary", "op": "/",
                    "left": {"type": "rollup", "scope": "children", "agg": "sum",
                             "label_ids": [vessel], "metric": "area"},
                    "right": {"type": "metric", "metric": "area"}},
    )


def _density(coral, polyp):
    return dm.DerivedMetricDefinition(
        key="polyp_density", name="Polyp density", label_ids=[coral],
        expression={"type": "binary", "op": "/",
                    "left": {"type": "rollup", "scope": "children", "agg": "count",
                             "label_ids": [polyp]},
                    "right": {"type": "metric", "metric": "area"}},
    )


@pytest.fixture
def ctx(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'derived.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    curator = Users(username="curator", hashed_password="x")
    db.add_all([curator, Users(username="viewer", hashed_password="x")])
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path=str(tmp_path), created_by="curator")
    other = Datasets(name="other", description="", dataset_type="image",
                     folder_path=str(tmp_path), created_by="curator")
    db.add_all([ds, other])
    db.flush()
    for dataset in (ds, other):
        db.add(DatasetMembers(dataset_id=dataset.id, username="curator",
                              role=DatasetRole.CURATOR.value,
                              extra_permissions=[], denied_permissions=[]))
    db.add(DatasetMembers(dataset_id=ds.id, username="viewer", role=DatasetRole.VIEWER.value,
                          extra_permissions=[], denied_permissions=[]))

    choroid = Labels(dataset_id=ds.id, name="Choroid", value=1)
    coral = Labels(dataset_id=ds.id, name="Coral", value=2)
    db.add_all([choroid, coral])
    db.flush()
    vessel = Labels(dataset_id=ds.id, name="Vessel", value=3, parent_id=choroid.id)
    polyp = Labels(dataset_id=ds.id, name="Polyp", value=4, parent_id=coral.id)
    db.add_all([vessel, polyp])
    db.flush()

    images, masks = [], []
    for index in range(3):
        image = Images(dataset_id=ds.id, file_name=f"img_{index}.png",
                       file_path=str(tmp_path / f"img_{index}.png"),
                       thumbnail_file_path=str(tmp_path / f"t_{index}.png"),
                       width=100, height=80, color_mode="RGB")
        db.add(image)
        db.flush()
        mask = Masks(image_id=image.id, fully_annotated=True, file_path=str(tmp_path / f"m_{index}.png"))
        db.add(mask)
        db.flush()
        images.append(image)
        masks.append(mask)

    def add(mask, label, area, parent=None, reviewed=True):
        contour = Contours(mask_id=mask.id, label_id=label.id, parent_id=parent.id if parent else None,
                           added_by="manual", author_username="curator", confidence_score=1.0,
                           area=area, perimeter=1.0, circularity=0.5, diameter=1.0,
                           x=[0.1, 0.5, 0.5, 0.1], y=[0.1, 0.1, 0.5, 0.5])
        if reviewed:
            contour.reviewed_by.append(curator)
        db.add(contour)
        db.flush()
        # Every geometry metric stored and fresh, so the routes' lazy compute leaves the
        # fixture's areas alone instead of re-measuring the placeholder outline.
        for key in GEOMETRY_METRIC_KEYS:
            db.add(ContourMetrics(contour_id=contour.id, metric_key=key, component=0,
                                  value=area if key == "area" else 1.0, unit="", stale=False))
        return contour

    # Image 0: vascularity (20 + 10) / 100 = 0.3.
    choroid_a = add(masks[0], choroid, 100.0)
    add(masks[0], vessel, 20.0, parent=choroid_a)
    add(masks[0], vessel, 10.0, parent=choroid_a)
    # Image 1: a choroid without vessels -> 0, plus an unreviewed vessel elsewhere.
    choroid_b = add(masks[1], choroid, 50.0)
    choroid_c = add(masks[1], choroid, 40.0)
    add(masks[1], vessel, 8.0, parent=choroid_c, reviewed=False)
    # Image 2: a coral of area 200 with four polyps -> 0.02 per px².
    coral_a = add(masks[2], coral, 200.0)
    for _ in range(4):
        add(masks[2], polyp, 1.0, parent=coral_a)

    db.add(Labels(dataset_id=other.id, name="Choroid", value=1))
    db.flush()
    db.add(Labels(dataset_id=other.id, name="Vessel", value=2))
    db.commit()

    yield {"db": db, "Session": Session, "ds": ds.id, "other": other.id, "images": [i.id for i in images],
           "choroid": choroid.id, "vessel": vessel.id, "coral": coral.id,
           "polyp": polyp.id, "choroid_a": choroid_a.id, "choroid_b": choroid_b.id,
           "choroid_c": choroid_c.id, "coral_a": coral_a.id}
    db.close()
    engine.dispose()


def _summary(ctx, exclude_unreviewed=True, **kwargs):
    return asyncio.run(datasets_db.get_quantification_summary(
        ctx["ds"], True, exclude_unreviewed, ctx["db"], **kwargs))


def _stats(block, label_id, key):
    return block[str(label_id)][key]["components"][0]


# ---------------------------------------------------------------------------
# The engine on its own
# ---------------------------------------------------------------------------

def _resolve(*definitions, labels=frozenset({1, 2, 3})):
    return dm.resolve_definitions([(None, d) for d in definitions], set(labels))


def _def(key, expression, label_ids=None):
    return dm.DerivedMetricDefinition(key=key, name=key, label_ids=label_ids, expression=expression)


AREA = {"type": "metric", "metric": "area"}
PERIMETER = {"type": "metric", "metric": "perimeter"}


def test_units_follow_from_the_formula():
    everything, _ = _resolve(
        _def("ratio", {"type": "binary", "op": "/", "left": AREA, "right": AREA}),
        _def("per_area", {"type": "binary", "op": "/", "left": {"type": "constant", "value": 1}, "right": AREA}),
        _def("side", {"type": "func", "fn": "sqrt", "arg": AREA}),
        _def("compact", {"type": "binary", "op": "/",
                         "left": {"type": "binary", "op": "*", "left": PERIMETER, "right": PERIMETER},
                         "right": AREA}),
    )
    assert [item.exponent for item in everything] == [0.0, -2.0, 1.0, 0.0]
    assert dm.unit_for_exponent(-2.0, "mm") == "/mm²"
    assert dm.unit_for_exponent(2.0, "mm") == "mm²"
    assert dm.unit_for_exponent(0.0, "mm") == ""
    assert dm.unit_for_exponent(0.5, "mm") == "mm^0.5"


def test_formulas_without_a_unit_are_refused():
    everything, order = _resolve(
        _def("nonsense", {"type": "binary", "op": "+", "left": AREA, "right": PERIMETER}),
        _def("log_area", {"type": "func", "fn": "log", "arg": AREA}),
    )
    assert not order
    assert "different units" in everything[0].problems[0]
    assert "unitless" in everything[1].problems[0]


def test_cycles_and_unknown_inputs_are_reported():
    a = _def("a", {"type": "binary", "op": "*", "left": {"type": "metric", "metric": "b"}, "right": AREA})
    b = _def("b", {"type": "metric", "metric": "a"})
    c = _def("c", {"type": "metric", "metric": "a"})
    unknown = _def("d", {"type": "metric", "metric": "volume"})
    everything, order = _resolve(a, b, c, unknown)
    assert not order
    assert any("Circular" in p for p in everything[0].problems)
    assert any("Circular" in p for p in everything[1].problems)
    assert any("has a problem" in p for p in everything[2].problems)
    assert "Unknown measurement 'volume'." in everything[3].problems


def test_a_derived_metric_can_build_on_another():
    double = _def("double_area", {"type": "binary", "op": "*", "left": {"type": "constant", "value": 2}, "right": AREA})
    # Defined first, but evaluated after what it reads.
    quad = _def("quad_area", {"type": "binary", "op": "*", "left": {"type": "constant", "value": 2},
                              "right": {"type": "metric", "metric": "double_area"}})
    _, order = _resolve(quad, double)
    assert [item.definition.key for item in order] == ["double_area", "quad_area"]

    objects = pd.DataFrame({"label_id": [1], "parent_id": [None], "mask_id": [1]}, index=pd.Index([7]))
    frame = dm.EvaluationFrame(objects, {7: None}, {("area", 0): pd.Series({7: 3.0})})
    assert dm.evaluate(order, frame)["quad_area"][7] == 12.0


def test_descendants_reach_past_a_level_children_do_not():
    # 1 -> 2 -> 3: the grandchild 3 is a descendant of 1 but not its child.
    objects = pd.DataFrame({"label_id": [1, 2, 2], "parent_id": [None, 1, 2], "mask_id": [1, 1, 1]},
                           index=pd.Index([1, 2, 3]))
    frame = dm.EvaluationFrame(objects, {1: None, 2: 1, 3: 2}, {})
    count = lambda scope: _def(scope, {"type": "rollup", "scope": scope, "agg": "count", "label_ids": [2]})
    _, order = _resolve(count("children"), count("descendants"), count("image"), count("parent"))
    values = dm.evaluate(order, frame)
    assert values["children"].tolist() == [1.0, 1.0, 0.0]
    assert values["descendants"].tolist() == [2.0, 1.0, 0.0]
    assert values["image"].tolist() == [2.0, 2.0, 2.0]
    assert values["parent"].tolist() == [0.0, 0.0, 1.0]


def test_division_by_zero_is_missing_not_infinite():
    objects = pd.DataFrame({"label_id": [1], "parent_id": [None], "mask_id": [1]}, index=pd.Index([1]))
    frame = dm.EvaluationFrame(objects, {1: None}, {("area", 0): pd.Series({1: 0.0})})
    _, order = _resolve(_def("inv", {"type": "binary", "op": "/", "left": {"type": "constant", "value": 1}, "right": AREA}))
    assert pd.isna(dm.evaluate(order, frame)["inv"][1])


# ---------------------------------------------------------------------------
# Against the database
# ---------------------------------------------------------------------------

def test_vascularity_index_in_the_summary(ctx):
    derived_db.create(ctx["db"], ctx["ds"], _cvi(ctx["choroid"], ctx["vessel"]), "curator")
    stats = _stats(_summary(ctx)["metrics"], ctx["choroid"], "cvi")
    # 0.3, 0 (no vessels) and 0 (its only vessel is unreviewed, so the default filters
    # leave it out).
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(0.1)
    assert stats["max"] == pytest.approx(0.3)
    assert _summary(ctx)["metrics"][str(ctx["choroid"])]["cvi"]["unit"] == ""
    # Reported only on the label it is defined for.
    assert "cvi" not in _summary(ctx)["metrics"].get(str(ctx["vessel"]), {})


def test_the_filters_apply_to_what_a_rollup_counts(ctx):
    derived_db.create(ctx["db"], ctx["ds"], _cvi(ctx["choroid"], ctx["vessel"]), "curator")
    stats = _stats(_summary(ctx, exclude_unreviewed=False)["metrics"], ctx["choroid"], "cvi")
    assert stats["max"] == pytest.approx(0.3)
    assert stats["mean"] == pytest.approx((0.3 + 0.0 + 8.0 / 40.0) / 3)


def test_density_in_pixels(ctx):
    derived_db.create(ctx["db"], ctx["ds"], _density(ctx["coral"], ctx["polyp"]), "curator")
    entry = _summary(ctx)["metrics"][str(ctx["coral"])]["polyp_density"]
    assert entry["unit"] == "/px²"
    assert entry["components"][0]["mean"] == pytest.approx(4 / 200)


def test_density_unit_is_per_area(ctx):
    for image_id in ctx["images"]:
        image = ctx["db"].get(Images, image_id)
        image.scale_x, image.scale_y, image.unit = 0.5, 0.5, "mm"
    ctx["db"].commit()
    derived_db.create(ctx["db"], ctx["ds"], _density(ctx["coral"], ctx["polyp"]), "curator")
    derived_db.create(ctx["db"], ctx["ds"], _cvi(ctx["choroid"], ctx["vessel"]), "curator")
    metrics = _summary(ctx)["metrics"]
    density = metrics[str(ctx["coral"])]["polyp_density"]
    assert density["unit"] == "/mm²"
    # 4 polyps over 200 px² = 50 mm².
    assert density["components"][0]["mean"] == pytest.approx(4 / 50)
    # A ratio of areas does not change with the scale.
    assert _stats(metrics, ctx["choroid"], "cvi")["max"] == pytest.approx(0.3)


def test_derived_metrics_are_columns_of_the_export(ctx):
    derived_db.create(ctx["db"], ctx["ds"], _cvi(ctx["choroid"], ctx["vessel"]), "curator")
    df = asyncio.run(datasets_db.get_dataset_as_df(
        ctx["ds"], True, True, ctx["db"], metric_scoping={"area": None}))
    by_id = df.set_index("contour_id")["cvi"]
    assert by_id[ctx["choroid_a"]] == pytest.approx(0.3)
    assert by_id[ctx["choroid_b"]] == 0.0
    assert pd.isna(by_id[ctx["coral_a"]])


def test_grouped_summary_carries_derived_metrics(ctx):
    derived_db.create(ctx["db"], ctx["ds"], _cvi(ctx["choroid"], ctx["vessel"]), "curator")
    meta.set_metadata_for_images(ctx["db"], [ctx["images"][0]], {"eye": "OD"})
    meta.set_metadata_for_images(ctx["db"], [ctx["images"][1]], {"eye": "OS"})
    summary = _summary(ctx, group_by_key="eye")
    assert _stats(summary["groups"]["OD"], ctx["choroid"], "cvi")["mean"] == pytest.approx(0.3)
    assert _stats(summary["groups"]["OS"], ctx["choroid"], "cvi")["mean"] == 0.0


def test_a_deleted_label_breaks_the_definition_visibly(ctx):
    derived_db.create(ctx["db"], ctx["ds"], _cvi(ctx["choroid"], ctx["vessel"]), "curator")
    ctx["db"].delete(ctx["db"].get(Labels, ctx["vessel"]))
    ctx["db"].commit()
    listed = derived_db.list_serialized(ctx["db"], ctx["ds"])
    assert "no longer exist" in listed[0]["problems"][0]
    assert "cvi" not in _summary(ctx)["metrics"].get(str(ctx["choroid"]), {})


def test_copy_matches_labels_by_name(ctx):
    derived_db.create(ctx["db"], ctx["ds"], _cvi(ctx["choroid"], ctx["vessel"]), "curator")
    copied = derived_db.copy_from_dataset(ctx["db"], ctx["ds"], ctx["other"], None, "curator")
    target = {name: lid for lid, name in ctx["db"].query(Labels.id, Labels.name)
              .filter(Labels.dataset_id == ctx["other"]).all()}
    assert copied[0]["label_ids"] == [target["Choroid"]]
    assert copied[0]["expression"]["left"]["label_ids"] == [target["Vessel"]]
    assert not copied[0]["problems"]


def test_copy_refuses_labels_the_target_lacks(ctx):
    derived_db.create(ctx["db"], ctx["ds"], _density(ctx["coral"], ctx["polyp"]), "curator")
    with pytest.raises(dm.DerivedMetricError, match="Coral, Polyp"):
        derived_db.copy_from_dataset(ctx["db"], ctx["ds"], ctx["other"], None, "curator")


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

def _client(ctx, username):
    Session = ctx["Session"]
    app = FastAPI()
    app.include_router(datasets_router)

    def _session_override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def _user_override():
        session = Session()
        try:
            return AuthenticatedUser.from_query(session.query(Users).filter_by(username=username).one())
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override
    return TestClient(app)


def test_crud_over_http(ctx):
    client = _client(ctx, "curator")
    base = f"/datasets/{ctx['ds']}/quantification/derived-metrics"
    body = _cvi(ctx["choroid"], ctx["vessel"]).model_dump()

    created = client.post(base, json=body)
    assert created.status_code == 200, created.text
    metric_id = created.json()["derived_metric"]["id"]

    assert client.post(base, json=body).status_code == 422  # key taken
    builtin = dict(body, key="area")
    assert "built-in" in client.post(base, json=builtin).json()["detail"]

    renamed = dict(body, key="other_key")
    assert client.put(f"{base}/{metric_id}", json=renamed).status_code == 422

    doubled = dict(body, key="cvi_pct", expression={
        "type": "binary", "op": "*", "left": {"type": "constant", "value": 100},
        "right": {"type": "metric", "metric": "cvi"}})
    assert client.post(base, json=doubled).status_code == 200
    assert client.delete(f"{base}/{metric_id}").status_code == 409  # cvi_pct reads it

    listed = client.get(base).json()["derived_metrics"]
    assert [m["key"] for m in listed] == ["cvi", "cvi_pct"]

    summary = client.get(f"/datasets/{ctx['ds']}/quantification/summary").json()
    assert summary["metrics"][str(ctx["choroid"])]["cvi_pct"]["components"][0]["max"] == pytest.approx(30.0)


def test_preview_reports_problems_as_data(ctx):
    client = _client(ctx, "curator")
    base = f"/datasets/{ctx['ds']}/quantification/derived-metrics/preview"
    good = client.post(base, json={"definition": _cvi(ctx["choroid"], ctx["vessel"]).model_dump()}).json()
    assert good["ok"] is True
    assert good["objects_in_scope"] == 3
    assert good["summary"]["max"] == pytest.approx(0.3)
    assert len(good["samples"]) == 3
    assert {s["label"] for s in good["samples"]} == {"Choroid"}

    bad = _def("bad", {"type": "binary", "op": "+", "left": AREA, "right": PERIMETER}).model_dump()
    response = client.post(base, json={"definition": bad})
    assert response.status_code == 200
    assert response.json()["ok"] is False

    # A new definition reusing a saved key is refused, not previewed in the saved one's place.
    derived_db.create(ctx["db"], ctx["ds"], _cvi(ctx["choroid"], ctx["vessel"]), "curator")
    clash = client.post(base, json={"definition": _density(ctx["coral"], ctx["polyp"]).model_dump() | {"key": "cvi"}})
    assert clash.json()["ok"] is False
    assert "already has" in clash.json()["problems"][0]


def test_viewers_cannot_change_definitions(ctx):
    client = _client(ctx, "viewer")
    base = f"/datasets/{ctx['ds']}/quantification/derived-metrics"
    assert client.post(base, json=_cvi(ctx["choroid"], ctx["vessel"]).model_dump()).status_code == 403
    assert client.get(base).status_code == 200
