"""Tests for app/services/scale_computation.py.

Covers the main service functions: get_image_scale, set_image_scale,
set_scale_from_drawn_line, compute_pixel_scale_from_points, and the stale-metric
marking behaviour after a scale change.
"""
import math
import pytest
from unittest.mock import MagicMock, patch, call

from app.exceptions import ImageNotFoundError, InvalidScaleError
from app.services.scale_computation import (
    compute_pixel_scale_from_points,
    get_image_scale,
    set_image_scale,
    set_scale_from_drawn_line,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(scale_x=1.0, scale_y=1.0, unit="px"):
    """Return a mock Images ORM row."""
    img = MagicMock()
    img.id = 42
    img.scale_x = scale_x
    img.scale_y = scale_y
    img.unit = unit
    return img


def _make_db(image=None):
    """Return a mock SQLAlchemy Session with query().filter_by().first() set up."""
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = image
    return db


# ---------------------------------------------------------------------------
# compute_pixel_scale_from_points
# ---------------------------------------------------------------------------

class TestComputePixelScaleFromPoints:
    def test_horizontal_line(self):
        scale = compute_pixel_scale_from_points((0, 0), (100, 0), 10.0)
        assert math.isclose(scale, 0.1, rel_tol=1e-9)

    def test_diagonal_line(self):
        # 3-4-5 right triangle: pixel_distance = 5
        scale = compute_pixel_scale_from_points((0, 0), (3, 4), 5.0)
        assert math.isclose(scale, 1.0, rel_tol=1e-9)

    def test_identical_points_raises(self):
        with pytest.raises(InvalidScaleError, match="identical points"):
            compute_pixel_scale_from_points((5, 5), (5, 5), 10.0)


# ---------------------------------------------------------------------------
# get_image_scale
# ---------------------------------------------------------------------------

class TestGetImageScale:
    def test_returns_scale_dict(self):
        db = _make_db(_make_image(scale_x=0.5, scale_y=0.5, unit="mm"))
        result = get_image_scale(db, 42)
        assert result == {"scale_x": 0.5, "scale_y": 0.5, "unit": "mm"}

    def test_raises_if_not_found(self):
        db = _make_db(image=None)
        with pytest.raises(ImageNotFoundError):
            get_image_scale(db, 999)


# ---------------------------------------------------------------------------
# set_image_scale
# ---------------------------------------------------------------------------

class TestSetImageScale:
    def test_updates_image_and_commits(self):
        img = _make_image(scale_x=1.0, scale_y=1.0, unit="px")
        db = _make_db(img)

        with patch("app.services.scale_computation._mark_scale_dependent_metrics_stale") as mock_stale:
            result = set_image_scale(db, 42, 0.25, 0.25, "mm")

        assert img.scale_x == 0.25
        assert img.scale_y == 0.25
        assert img.unit == "mm"
        db.commit.assert_called_once()
        mock_stale.assert_called_once_with(db, 42)
        assert result == {"scale_x": 0.25, "scale_y": 0.25, "unit": "mm"}

    def test_no_commit_if_unchanged(self):
        img = _make_image(scale_x=0.25, scale_y=0.25, unit="mm")
        db = _make_db(img)
        with patch("app.services.scale_computation._mark_scale_dependent_metrics_stale") as mock_stale:
            set_image_scale(db, 42, 0.25, 0.25, "mm")
        db.commit.assert_not_called()
        mock_stale.assert_not_called()

    def test_raises_on_zero_scale(self):
        db = _make_db(_make_image())
        with pytest.raises(InvalidScaleError, match="positive"):
            set_image_scale(db, 42, 0.0, 1.0, "mm")

    def test_raises_on_negative_scale(self):
        db = _make_db(_make_image())
        with pytest.raises(InvalidScaleError, match="positive"):
            set_image_scale(db, 42, -1.0, 1.0, "mm")

    def test_raises_on_empty_unit(self):
        db = _make_db(_make_image())
        with pytest.raises(InvalidScaleError, match="non-empty"):
            set_image_scale(db, 42, 1.0, 1.0, "")

    def test_raises_if_image_not_found(self):
        db = _make_db(image=None)
        with pytest.raises(ImageNotFoundError):
            set_image_scale(db, 999, 1.0, 1.0, "mm")


# ---------------------------------------------------------------------------
# set_scale_from_drawn_line
# ---------------------------------------------------------------------------

class TestSetScaleFromDrawnLine:
    def test_correct_scale_computed(self):
        img = _make_image()
        db = _make_db(img)
        with patch("app.services.scale_computation._mark_scale_dependent_metrics_stale"):
            result = set_scale_from_drawn_line(db, 42, (0, 0), (100, 0), 10.0, "mm")
        # 10 mm / 100 px = 0.1 mm/px
        assert math.isclose(result["scale_x"], 0.1, rel_tol=1e-9)
        assert math.isclose(result["pixel_distance"], 100.0, rel_tol=1e-9)
        assert result["unit"] == "mm"

    def test_raises_on_zero_known_distance(self):
        db = _make_db(_make_image())
        with pytest.raises(InvalidScaleError, match="positive"):
            set_scale_from_drawn_line(db, 42, (0, 0), (100, 0), 0.0, "mm")

    def test_raises_on_identical_points(self):
        db = _make_db(_make_image())
        with pytest.raises(InvalidScaleError, match="identical"):
            set_scale_from_drawn_line(db, 42, (5, 5), (5, 5), 10.0, "mm")
