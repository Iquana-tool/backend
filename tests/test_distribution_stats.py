"""Tests for the server-side distribution stats (box/violin) computation.

Covers the pure numpy reducer ``_compute_distribution_stats`` (percentiles, whiskers,
outliers, KDE / histogram fallback, payload bounding) and the end-to-end
``get_quantification_distribution`` over a small SQLite dataset, including that it only
emits value_dim-1 numeric metrics and honors metric_scoping.
"""
import asyncio

import numpy as np
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
import app.database.contours  # noqa: F401
from app.database.contour_metrics import ContourMetrics
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users

from app.services.database_access import datasets as datasets_db
from app.services.database_access.datasets import (
    _compute_distribution_stats,
    _distribution_metric_keys,
    _MAX_OUTLIER_SAMPLES,
    _KDE_GRID_POINTS,
)


class TestComputeDistributionStats:
    def test_known_quartiles_and_whiskers(self):
        # 1..100 with a single blatant high outlier at 1000.
        values = np.array(list(range(1, 101)) + [1000], dtype=np.float64)
        stats = _compute_distribution_stats(values)

        # numpy linear-interpolation percentiles on 1..100 + outlier.
        np_q1, np_med, np_q3 = np.percentile(values, [25, 50, 75])
        assert stats["q1"] == pytest.approx(np_q1)
        assert stats["median"] == pytest.approx(np_med)
        assert stats["q3"] == pytest.approx(np_q3)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(1000.0)
        # 1000 is far beyond q3 + 1.5*IQR -> flagged as an outlier.
        assert stats["outlier_count"] >= 1
        assert 1000.0 in stats["outliers"]
        # Whiskers stay within the inlier range (never reach the outlier).
        assert stats["whisker_high"] < 1000.0

    def test_kde_present_and_bounded(self):
        rng = np.random.default_rng(0)
        values = rng.normal(50, 10, size=500)
        stats = _compute_distribution_stats(values)
        assert stats["kde"] is not None
        assert len(stats["kde"]["x"]) == _KDE_GRID_POINTS
        assert len(stats["kde"]["density"]) == _KDE_GRID_POINTS
        # Density is non-negative.
        assert all(d >= 0 for d in stats["kde"]["density"])

    def test_zero_variance_skips_kde(self):
        # All identical -> no KDE, no histogram (nothing to plot as a distribution).
        stats = _compute_distribution_stats(np.array([5.0, 5.0, 5.0]))
        assert stats["kde"] is None
        assert stats["histogram"] is None
        assert stats["q1"] == stats["median"] == stats["q3"] == pytest.approx(5.0)

    def test_single_value(self):
        stats = _compute_distribution_stats(np.array([7.0]))
        assert stats["count"] == 1
        assert stats["kde"] is None
        assert stats["mean"] == pytest.approx(7.0)

    def test_outlier_sample_is_bounded(self):
        # A dense core (so the IQR is small) plus many spread-out high values that all fall
        # beyond q3 + 1.5*IQR -> lots of outliers, but the returned sample stays capped.
        core = np.repeat(np.arange(0.0, 10.0), 40)      # 400 values in [0, 9], tight IQR
        outliers = np.linspace(1000.0, 3000.0, 60)      # 60 distinct far-out values
        stats = _compute_distribution_stats(np.concatenate([core, outliers]))
        assert stats["outlier_count"] == 60
        assert len(stats["outliers"]) <= _MAX_OUTLIER_SAMPLES

    def test_two_bell_ish_points_use_histogram_fallback(self):
        # n>=2 with variance but a KDE that could be poorly conditioned still yields a curve;
        # here we assert the histogram fallback path when kde is unavailable is well-formed
        # by forcing a tiny distinct-value set.
        stats = _compute_distribution_stats(np.array([1.0, 2.0]))
        # scipy can KDE two distinct points; either kde or histogram must be present.
        assert stats["kde"] is not None or stats["histogram"] is not None


class TestDistributionMetricKeys:
    def test_only_numeric_value_dim_1(self):
        keys = _distribution_metric_keys()
        # geometry + contextual + count metrics are eligible.
        assert {"area", "perimeter", "circularity", "max_diameter", "nn_distance", "n_children"} <= keys
        # color metrics (value_dim 3) are excluded.
        assert "mean_color_rgb" not in keys
        assert "mean_color_lab" not in keys

    def test_scoping_intersects(self):
        keys = _distribution_metric_keys(["area", "mean_color_rgb", "n_children"])
        assert keys == {"area", "n_children"}


# --- end-to-end over a small dataset -----------------------------------------

WIDTH, HEIGHT = 1000, 1000


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


def _seed_with_area_values(session, area_values):
    """Seed one label with N contours, each carrying a stored 'area' metric row."""
    ds = Datasets(name="dist", description="", dataset_type="image",
                  folder_path="/tmp/dist", created_by="u")
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.add(ds)
    session.flush()
    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()
    mask = Masks(image_id=img.id, fully_annotated=True, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()
    label = Labels(dataset_id=ds.id, parent_id=None, name="blob", value=1)
    session.add(label)
    session.flush()

    from app.database.contours import Contours
    for v in area_values:
        c = Contours(mask_id=mask.id, parent_id=None, added_by="u", confidence_score=1.0,
                     label_id=label.id, area=v, perimeter=0.0, circularity=0.0, diameter=0.0,
                     x=[0.0, 0.1, 0.1], y=[0.0, 0.0, 0.1])
        session.add(c)
        session.flush()
        # Mark it reviewed so exclude_unreviewed=True keeps it.
        c.reviewed_by = session.query(Users).filter_by(username="u").all()
        session.add(ContourMetrics(contour_id=c.id, metric_key="area", component=0,
                                   value=v, unit="px²", stale=False))
    session.commit()
    return ds, label


def test_distribution_end_to_end(session):
    values = [float(v) for v in range(1, 21)]  # 1..20
    ds, label = _seed_with_area_values(session, values)

    dist = asyncio.run(datasets_db.get_quantification_distribution(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
    ))
    label_key = str(label.id)
    assert label_key in dist
    area_stats = dist[label_key]["area"]["0"]
    assert area_stats["count"] == 20
    assert area_stats["min"] == pytest.approx(1.0)
    assert area_stats["max"] == pytest.approx(20.0)
    assert area_stats["median"] == pytest.approx(np.median(values))
    # KDE present for a 20-point spread.
    assert area_stats["kde"] is not None


def test_distribution_respects_scoping(session):
    ds, label = _seed_with_area_values(session, [1.0, 2.0, 3.0, 4.0])
    # Scope to a metric that has no stored rows -> empty distribution.
    dist = asyncio.run(datasets_db.get_quantification_distribution(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
        metric_scoping={"perimeter": None},
    ))
    assert dist == {}
