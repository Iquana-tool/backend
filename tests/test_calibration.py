"""Tests for the image-calibration system.

Four layers, tested where each one's risk actually lives:

  * reference card profiles — the target values a physical card should read, which
    is where the original MATLAB utility was quietly wrong,
  * the strategies' estimation maths, which are pure functions,
  * the service against a real SQLite database — set/read/clear/propagate, dataset
    defaults, the mirror into the legacy ``images`` scale columns, and the
    staleness marking that keeps stored measurements honest,
  * the pixel pipeline end to end, including the property that makes re-calibration
    safe (a re-sample must not see the calibration it is about to replace).
"""
import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# Import the shared declarative base and every model module so metadata is complete.
from app.database import database
import app.database.dataset_calibration_defaults  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.image_calibrations  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
import app.database.contours  # noqa: F401  (also pulls in contour_metrics)
from app.database.contour_metrics import ContourMetrics
from app.database.contours import save_contour_tree
from app.database.datasets import Datasets
from app.database.image_calibrations import ImageCalibrations
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users

from app.exceptions import (
    InvalidCalibrationError,
    InvalidScaleError,
    UnknownCalibrationKindError,
)
from app.services.calibration import cards, registry, service, store, strategies
from app.services.calibration.registry import CalibrationSource

from iquana_toolbox.schemas.database.contours import Contour

WIDTH, HEIGHT = 40, 30

#: A synthetic wedge reading: linear in patch index, so a straight-line fit
#: recovers it exactly and the anchors map each sample onto its card target.
WEDGE_OBSERVED = [200.0 - 9.0 * i for i in range(20)]      # 200 down to 29


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'calibration.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()


def _write_image(path, rgb):
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[:, :] = rgb
    PILImage.fromarray(array).save(path)
    return str(path)


@pytest.fixture
def image(session, tmp_path):
    """A dataset with one solid image carrying a warm colour cast."""
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    ds = Datasets(name="cal", description="", dataset_type="image",
                  folder_path=str(tmp_path), created_by="u")
    session.add(ds)
    session.flush()

    path = _write_image(tmp_path / "img.png", (180, 160, 120))
    img = Images(dataset_id=ds.id, file_name="img.png", file_path=path,
                 thumbnail_file_path=path, width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.commit()
    return img


@pytest.fixture
def image_with_contour(session, image):
    """The same image, plus one contour carrying appearance + geometry metrics."""
    mask = Masks(image_id=image.id, fully_annotated=True, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()
    label = Labels(dataset_id=image.dataset_id, parent_id=None, name="blob", value=1)
    session.add(label)
    session.flush()

    save_contour_tree(session, Contour(
        x=[0.1, 0.6, 0.6, 0.1], y=[0.1, 0.1, 0.6, 0.6],
        label_id=label.id, added_by="u", reviewed_by=["u"],
    ), mask.id)
    session.commit()
    return image


def _wedge_samples(observed=None):
    """One neutral RGB triple per card patch, in card order."""
    return [[value, value, value] for value in (observed or WEDGE_OBSERVED)]


# ---------------------------------------------------------------------------
# Reference cards
# ---------------------------------------------------------------------------

class TestReferenceCards:
    def test_kodak_has_the_twenty_printed_steps(self):
        card = cards.get_card("kodak_q13")
        assert len(card.neutral_patches) == 20
        names = [patch.name for patch in card.neutral_patches]
        # A, M and B are printed markers occupying positions 0, 7 and 16.
        assert names[0] == "A" and names[7] == "M" and names[16] == "B"

    def test_physical_targets_come_from_the_published_densities(self):
        """The card steps in equal density, which is not equal pixel value."""
        card = cards.get_card("kodak_q13")
        patches = card.neutral_patches
        assert patches[0].density == pytest.approx(0.05)
        assert patches[19].density == pytest.approx(1.95)
        # 10**-D through the sRGB encoding, not a straight line.
        assert patches[0].target_rgb[0] == pytest.approx(242.4, abs=0.5)
        assert patches[7].target_rgb[0] == pytest.approx(117.0, abs=0.5)
        assert patches[19].target_rgb[0] == pytest.approx(27.4, abs=0.5)

    def test_legacy_targets_reproduce_the_matlab_ramp(self):
        """256 - 13i, clamped at the top to a value 8-bit output can take."""
        patches = cards.get_card("kodak_q13_legacy").neutral_patches
        assert [patch.target_rgb[0] for patch in patches[:3]] == [255.0, 243.0, 230.0]
        assert patches[19].target_rgb[0] == 9.0

    def test_the_two_profiles_disagree_most_in_the_midtones(self):
        """The reason both ship: this gap is the MATLAB utility's systematic error."""
        physical = cards.get_card("kodak_q13").neutral_patches[7].target_rgb[0]
        legacy = cards.get_card("kodak_q13_legacy").neutral_patches[7].target_rgb[0]
        assert legacy - physical > 40

    def test_colour_spots_are_declared_but_carry_no_target(self):
        """Declaring them keeps the card honest; guessing their values would not."""
        card = cards.get_card("kodak_q13")
        chromatic = [p for p in card.patches if p.role == cards.PATCH_CHROMATIC]
        assert [p.name for p in chromatic] == ["C", "Y", "M"]
        assert all(p.target_rgb is None for p in chromatic)
        # And they are excluded from the patches a response is fitted through.
        assert all(p.role == cards.PATCH_NEUTRAL for p in card.neutral_patches)

    def test_unknown_card_raises(self):
        with pytest.raises(InvalidCalibrationError, match="colorchecker"):
            cards.get_card("colorchecker")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class TestTwoPatchStrategy:
    def _normalize(self, **params):
        return registry.get_kind("response").normalize({"strategy": "two_patch", **params})

    def test_black_and_white_map_onto_the_full_range(self):
        params = self._normalize(black_level=40, white_level=200)
        assert _apply(params, [40, 40, 40])[0] == pytest.approx(0.0, abs=0.5)
        assert _apply(params, [200, 200, 200])[0] == pytest.approx(255.0, abs=0.5)
        assert _apply(params, [120, 120, 120])[0] == pytest.approx(
            255.0 * (120 - 40) / (200 - 40), abs=0.5)

    def test_a_neutral_patch_balances_without_changing_brightness(self):
        """Cast removal is this strategy's job; exposure is the black/white points'."""
        params = self._normalize(black_level=10, white_level=200,
                                 neutral_rgb=[180, 160, 120])
        corrected = _apply(params, [180, 160, 120])
        assert max(corrected) - min(corrected) < 1.0

    def test_no_neutral_patch_means_tone_only(self):
        """A legitimate calibration: not every frame has a neutral reference."""
        params = self._normalize(black_level=10, white_level=200)
        assert params["gains"] == pytest.approx([1.0, 1.0, 1.0])

    def test_white_must_exceed_black(self):
        with pytest.raises(InvalidCalibrationError, match="at least 1 level"):
            self._normalize(black_level=200, white_level=100)

    def test_levels_outside_the_byte_range_are_rejected(self):
        with pytest.raises(InvalidCalibrationError, match="0-255"):
            self._normalize(black_level=-5, white_level=200)

    def test_a_neutral_patch_at_the_black_point_is_rejected(self):
        with pytest.raises(InvalidCalibrationError, match="under-exposed"):
            self._normalize(black_level=100, white_level=200, neutral_rgb=[100, 150, 150])

    def test_absurd_gains_are_rejected(self):
        with pytest.raises(InvalidCalibrationError, match="20x"):
            self._normalize(black_level=0, white_level=255, gains=[1.0, 1.0, 50.0])

    def test_missing_references_are_reported_not_guessed(self):
        with pytest.raises(InvalidCalibrationError, match="black_level"):
            self._normalize(gains=[1.0, 1.0, 1.0])


class TestGrayWedgeStrategy:
    def _normalize(self, **params):
        return registry.get_kind("response").normalize({
            "strategy": "gray_wedge", "samples": _wedge_samples(), **params,
        })

    def test_a_linear_fit_maps_each_sample_onto_its_card_target(self):
        params = self._normalize(card="kodak_q13_legacy")
        # Patch A read 200 and should be 255; the middle patch read 137 -> 165.
        assert _apply(params, [200, 200, 200])[0] == pytest.approx(255, abs=0.5)
        assert _apply(params, [137, 137, 137])[0] == pytest.approx(165, abs=0.5)
        assert _apply(params, [29, 29, 29])[0] == pytest.approx(9, abs=0.5)

    def test_the_card_choice_changes_the_result(self):
        """Same measurements, different target: this is the MATLAB correction gap."""
        legacy = self._normalize(card="kodak_q13_legacy")
        physical = self._normalize(card="kodak_q13")
        assert _apply(legacy, [137, 137, 137])[0] == pytest.approx(165, abs=0.5)
        assert _apply(physical, [137, 137, 137])[0] == pytest.approx(117, abs=0.5)

    def test_targets_are_never_regressed(self):
        """The original fitted a line to its reference too.

        That was harmless only because its reference was already a straight line.
        Doing it to the physically correct target would discard exactly the accuracy
        that target exists to provide, so the card's values are used as stated.
        """
        params = self._normalize(card="kodak_q13")
        targets = [patch.target_rgb[0]
                   for patch in cards.get_card("kodak_q13").neutral_patches]
        anchored = {round(target, 1) for _, target in params["anchors"]["r"]}
        assert {round(t, 1) for t in targets}.issubset(anchored)

    def test_a_linear_fit_smooths_a_mis_clicked_patch(self):
        """One bad sample should barely move a twenty-point least-squares line."""
        clean = self._normalize()
        noisy_observed = list(WEDGE_OBSERVED)
        noisy_observed[10] += 40                       # a glint on one patch
        noisy = registry.get_kind("response").normalize({
            "strategy": "gray_wedge", "samples": _wedge_samples(noisy_observed),
        })
        drift = abs(_apply(clean, [137] * 3)[0] - _apply(noisy, [137] * 3)[0])
        assert drift < 6

    def test_the_measured_fit_follows_the_samples_instead(self):
        params = self._normalize(fit_model="measured")
        # With no smoothing, every sample is an anchor at its own value.
        observed = [point[0] for point in params["anchors"]["r"]]
        assert all(value in observed for value in WEDGE_OBSERVED)

    def test_every_patch_must_be_sampled(self):
        with pytest.raises(InvalidCalibrationError, match="20 usable patches"):
            registry.get_kind("response").normalize({
                "strategy": "gray_wedge", "samples": _wedge_samples()[:5],
            })

    @pytest.mark.parametrize("fit_model", ["linear", "measured"])
    def test_patches_sampled_in_the_wrong_order_are_rejected(self, fit_model):
        """Reversed samples would otherwise produce an inverted transfer curve.

        Both fit models must say so in terms of what the user did, not in terms of
        the repaired data — "start from the lightest patch" is actionable where
        "these patches are all the same brightness" is not.
        """
        with pytest.raises(InvalidCalibrationError, match="order|darker"):
            registry.get_kind("response").normalize({
                "strategy": "gray_wedge",
                "samples": _wedge_samples(list(reversed(WEDGE_OBSERVED))),
                "fit_model": fit_model,
            })

    def test_sampling_the_same_patch_repeatedly_is_rejected(self):
        with pytest.raises(InvalidCalibrationError, match="no brightness gradient"):
            registry.get_kind("response").normalize({
                "strategy": "gray_wedge", "samples": _wedge_samples([128.0] * 20),
            })

    def test_unknown_fit_model_is_rejected(self):
        with pytest.raises(InvalidCalibrationError, match="fit model"):
            self._normalize(fit_model="spline")


class TestResponseTransform:
    def test_values_beyond_the_references_clamp_rather_than_extrapolate(self):
        """The original returned NaN there, which silently became black."""
        params = registry.get_kind("response").normalize({
            "strategy": "two_patch", "black_level": 40, "white_level": 200,
        })
        assert _apply(params, [10, 10, 10])[0] == pytest.approx(0.0, abs=0.5)
        assert _apply(params, [255, 255, 255])[0] == pytest.approx(255.0, abs=0.5)

    def test_gamma_applies_after_the_anchors(self):
        params = registry.get_kind("response").normalize({
            "strategy": "two_patch", "black_level": 0, "white_level": 255, "gamma": 2.0,
        })
        # Identity anchors, so the output is purely the gamma: (128/255)^2 * 255.
        assert _apply(params, [128, 128, 128])[0] == pytest.approx(
            255.0 * (128 / 255) ** 2, abs=0.5)

    def test_gamma_out_of_range_is_rejected(self):
        with pytest.raises(InvalidCalibrationError, match="gamma"):
            registry.get_kind("response").normalize({
                "strategy": "two_patch", "black_level": 0, "white_level": 255, "gamma": 99,
            })

    def test_precomputed_anchors_are_accepted_without_the_measurements(self):
        """What a dataset-wide apply propagates when the samples do not travel."""
        params = registry.get_kind("response").normalize({
            "strategy": "two_patch",
            "anchors": {channel: [[0.0, 0.0], [255.0, 255.0]] for channel in "rgb"},
        })
        assert _apply(params, [128, 128, 128])[0] == pytest.approx(128, abs=0.5)

    def test_non_monotonic_anchors_are_rejected(self):
        with pytest.raises(InvalidCalibrationError, match="strictly increasing"):
            registry.get_kind("response").normalize({
                "strategy": "two_patch",
                "anchors": {c: [[200.0, 0.0], [10.0, 255.0]] for c in "rgb"},
            })


class TestRegistry:
    def test_tone_and_colour_are_one_kind(self):
        """They were two; two estimates of one transform could be stacked."""
        assert [kind.key for kind in registry.all_kinds()] == ["scale", "response"]
        for legacy in ("intensity", "color"):
            with pytest.raises(UnknownCalibrationKindError):
                registry.get_kind(legacy)

    def test_scale_does_not_transform_pixels(self):
        assert registry.get_kind("scale").apply is None
        assert [kind.key for kind in registry.pixel_stages()] == ["response"]

    def test_the_card_based_strategy_is_the_default(self):
        kind = registry.get_kind("response")
        assert kind.strategy_keys[0] == "gray_wedge"
        assert kind.as_dict()["default_strategy"] == "gray_wedge"

    def test_the_legacy_card_is_the_default_target(self):
        """So adopting this system does not silently move existing results."""
        assert cards.DEFAULT_CARD == "kodak_q13_legacy"
        assert registry.get_kind("response").as_dict()["default_card"] == "kodak_q13_legacy"

    def test_scale_keeps_the_older_exception_type(self):
        """So the existing /scale routes' 422 mapping is unaffected."""
        with pytest.raises(InvalidScaleError):
            registry.get_kind("scale").normalize({"scale_x": 0, "scale_y": 1, "unit": "mm"})

    def test_unknown_kind_raises(self):
        with pytest.raises(UnknownCalibrationKindError, match="flat_field"):
            registry.get_kind("flat_field")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TestSetAndRead:
    def test_a_fresh_image_reports_every_kind_as_uncalibrated(self, session, image):
        state = service.get_calibration_state(session, image.id)
        assert state["calibrated_count"] == 0
        assert state["total_count"] == len(registry.all_kinds())
        assert all(entry["params"] is None for entry in state["calibrations"])

    def test_setting_a_calibration_makes_it_readable_with_provenance(self, session, image):
        service.set_calibration(session, image.id, "response",
                                {"strategy": "gray_wedge", "samples": _wedge_samples()},
                                source=CalibrationSource.MEASURED, username="u")
        entry = _entry_for(service.get_calibration_state(session, image.id), "response")
        assert entry["calibrated"] is True
        assert entry["source"] == CalibrationSource.MEASURED
        assert entry["created_by"] == "u"
        assert "20 patches" in entry["description"]

    def test_setting_a_calibration_twice_replaces_rather_than_accumulates(self, session, image):
        for white in (200, 220):
            service.set_calibration(session, image.id, "response",
                                    {"strategy": "two_patch", "black_level": 10,
                                     "white_level": white})
        rows = session.query(ImageCalibrations).filter_by(
            image_id=image.id, kind="response").all()
        assert len(rows) == 1
        assert rows[0].params["white_level"] == pytest.approx(220)

    def test_scale_is_mirrored_into_the_legacy_image_columns(self, session, image):
        """Everything predating the calibration table reads scale off the columns."""
        service.set_calibration(session, image.id, "scale",
                                {"scale_x": 0.25, "scale_y": 0.25, "unit": "mm"})
        session.refresh(image)
        assert image.scale_x == pytest.approx(0.25)
        assert image.unit == "mm"

    def test_scale_set_before_this_table_existed_still_reads_as_calibrated(self, session, image):
        image.scale_x = image.scale_y = 0.5
        image.unit = "mm"
        session.commit()
        assert not store.list_rows(session, image.id)

        entry = _entry_for(service.get_calibration_state(session, image.id), "scale")
        assert entry["calibrated"] is True
        assert entry["params"]["unit"] == "mm"
        assert entry["source"] is None  # no row, so no provenance to report

    def test_the_px_default_reads_as_uncalibrated_not_as_1to1(self, session, image):
        entry = _entry_for(service.get_calibration_state(session, image.id), "scale")
        assert entry["calibrated"] is False

    def test_invalid_params_are_rejected_before_anything_is_written(self, session, image):
        with pytest.raises(InvalidCalibrationError):
            service.set_calibration(session, image.id, "response",
                                    {"strategy": "two_patch", "black_level": 200,
                                     "white_level": 100})
        assert not store.list_rows(session, image.id)

    def test_unknown_kind_raises(self, session, image):
        with pytest.raises(UnknownCalibrationKindError):
            service.set_calibration(session, image.id, "flat_field", {})


class TestDatasetDefaults:
    def test_an_unconfigured_dataset_falls_back_to_the_registry_default(self, session, image):
        entry = _entry_for(service.get_calibration_state(session, image.id), "response")
        assert entry["dataset_defaults"] == {
            "strategy": "gray_wedge", "card": "kodak_q13_legacy",
        }

    def test_setting_defaults_changes_what_an_image_starts_from(self, session, image):
        service.set_dataset_defaults(session, image.dataset_id, "response",
                                     {"strategy": "gray_wedge", "card": "kodak_q13"},
                                     username="u")
        entry = _entry_for(service.get_calibration_state(session, image.id), "response")
        assert entry["dataset_defaults"]["card"] == "kodak_q13"

    def test_defaults_are_choosable_before_anything_is_measured(self, session, image):
        """The whole point: the strategy is picked once, not per image."""
        result = service.set_dataset_defaults(session, image.dataset_id, "response",
                                              {"strategy": "two_patch"})
        assert result["defaults"] == {"strategy": "two_patch"}

    def test_changing_defaults_leaves_existing_calibrations_alone(self, session, image):
        service.set_calibration(session, image.id, "response",
                                {"strategy": "gray_wedge", "samples": _wedge_samples()})
        service.set_dataset_defaults(session, image.dataset_id, "response",
                                     {"strategy": "two_patch"})
        entry = _entry_for(service.get_calibration_state(session, image.id), "response")
        assert entry["params"]["strategy"] == "gray_wedge"

    def test_a_strategy_the_kind_does_not_offer_is_rejected(self, session, image):
        with pytest.raises(InvalidCalibrationError, match="does not apply"):
            service.set_dataset_defaults(session, image.dataset_id, "response",
                                         {"strategy": "flat_field"})

    def test_a_kind_with_one_measurement_path_has_nothing_to_configure(self, session, image):
        with pytest.raises(InvalidCalibrationError, match="nothing to configure"):
            service.set_dataset_defaults(session, image.dataset_id, "scale", {})


class TestClear:
    def test_clearing_removes_the_row(self, session, image):
        service.set_calibration(session, image.id, "response",
                                {"strategy": "two_patch", "black_level": 10,
                                 "white_level": 200})
        result = service.clear_calibration(session, image.id, "response")
        assert result["cleared"] is True
        assert not store.list_rows(session, image.id)

    def test_clearing_scale_also_resets_the_mirror_columns(self, session, image):
        """Otherwise the old scale would stay silently in force after a 'clear'."""
        service.set_calibration(session, image.id, "scale",
                                {"scale_x": 0.25, "scale_y": 0.25, "unit": "mm"})
        service.clear_calibration(session, image.id, "scale")
        session.refresh(image)
        assert image.unit == "px"
        assert image.scale_x == pytest.approx(1.0)

    def test_clearing_something_unset_is_a_no_op(self, session, image):
        result = service.clear_calibration(session, image.id, "response")
        assert result["cleared"] is False
        assert result["metrics_invalidated"] == 0


class TestStaleness:
    def test_a_response_change_invalidates_appearance_but_not_geometry(
            self, session, image_with_contour):
        """A calibration marks dependent measurements stale; it never rewrites them."""
        _seed_metrics(session, image_with_contour.id)

        result = service.set_calibration(
            session, image_with_contour.id, "response",
            {"strategy": "two_patch", "black_level": 10, "white_level": 200})
        assert result["metrics_invalidated"] > 0
        assert _stale_keys(session) == set(registry.APPEARANCE_STALE_METRIC_KEYS)

    def test_a_scale_change_invalidates_geometry_but_not_appearance(
            self, session, image_with_contour):
        _seed_metrics(session, image_with_contour.id)
        service.set_calibration(session, image_with_contour.id, "scale",
                                {"scale_x": 0.25, "scale_y": 0.25, "unit": "mm"})

        stale = _stale_keys(session)
        assert "area" in stale
        assert not stale & set(registry.APPEARANCE_STALE_METRIC_KEYS)

    def test_circularity_survives_a_scale_change(self, session, image_with_contour):
        """It is dimensionless — the scale cancels, so recomputing it is waste."""
        _seed_metrics(session, image_with_contour.id)
        service.set_calibration(session, image_with_contour.id, "scale",
                                {"scale_x": 0.25, "scale_y": 0.25, "unit": "mm"})
        assert "circularity" not in _stale_keys(session)


class TestDatasetPropagation:
    def test_applying_to_a_dataset_marks_the_rows_as_propagated(self, session, image, tmp_path):
        second = _write_image(tmp_path / "img2.png", (100, 100, 100))
        session.add(Images(dataset_id=image.dataset_id, file_name="img2.png",
                           file_path=second, thumbnail_file_path=second,
                           width=WIDTH, height=HEIGHT, color_mode="RGB",
                           scale_x=1.0, scale_y=1.0, unit="px"))
        session.commit()

        result = service.apply_to_dataset(
            session, image.dataset_id, "response",
            {"strategy": "gray_wedge", "samples": _wedge_samples()}, username="u")
        assert result["images_updated"] == 2

        rows = session.query(ImageCalibrations).filter_by(kind="response").all()
        assert len(rows) == 2
        # Provenance is the point: a propagated calibration was not measured in
        # the frame it now applies to.
        assert {row.source for row in rows} == {CalibrationSource.DATASET}


# ---------------------------------------------------------------------------
# The pixel pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_an_uncalibrated_image_is_returned_untouched(self, session, image):
        pixels = np.full((4, 4, 3), 120, dtype=np.uint8)
        out = service.apply_calibration_pipeline(session, image.id, pixels)
        assert np.array_equal(out, pixels)

    def test_a_neutral_patch_neutralises_the_cast(self, session, image):
        service.set_calibration(session, image.id, "response",
                                {"strategy": "two_patch", "black_level": 10,
                                 "white_level": 200, "neutral_rgb": [180, 160, 120]})
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        pixels[:, :] = (180, 160, 120)

        out = service.apply_calibration_pipeline(session, image.id, pixels)
        assert out[0, 0].max() - out[0, 0].min() <= 1

    def test_a_malformed_row_skips_its_stage_instead_of_failing_the_image(
            self, session, image):
        """One bad calibration must not abort a whole dataset's appearance batch."""
        store.upsert(session, image.id, "response", {"strategy": "two_patch"}, "manual")
        session.commit()

        pixels = np.full((2, 2, 3), 90, dtype=np.uint8)
        out = service.apply_calibration_pipeline(session, image.id, pixels)
        assert np.array_equal(out, pixels)

    def test_appearance_metrics_are_computed_from_calibrated_pixels(
            self, session, image_with_contour):
        """The whole point: a calibration has to reach the stored measurements."""
        from app.services.quantification import compute_appearance_metrics_for_dataset

        compute_appearance_metrics_for_dataset(
            session, image_with_contour.dataset_id, only_stale=False)
        before = _mean_color(session)

        service.set_calibration(
            session, image_with_contour.id, "response",
            {"strategy": "two_patch", "black_level": 10, "white_level": 200,
             "neutral_rgb": [180, 160, 120]})
        compute_appearance_metrics_for_dataset(
            session, image_with_contour.dataset_id, only_stale=True)
        after = _mean_color(session)

        assert before != pytest.approx(after)
        # The image is a solid (180, 160, 120), which the calibration neutralises.
        assert max(after) - min(after) <= 1.5


class TestSampling:
    def test_sampling_reports_the_median_of_a_disc(self, session, image):
        result = service.sample_patch(session, image.id, x=20, y=15, radius=5)
        assert result["median_rgb"] == pytest.approx([180, 160, 120], abs=0.5)
        assert result["n_pixels"] > 0
        assert result["stages_applied"] == []

    def test_the_median_survives_a_glint_the_mean_does_not(self, session, tmp_path, image):
        """Why the original averaged with a median, and why this does too."""
        array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        array[:, :] = (100, 100, 100)
        array[14:16, 19:21] = 255            # a small specular highlight
        PILImage.fromarray(array).save(image.file_path)

        result = service.sample_patch(session, image.id, x=20, y=15, radius=6)
        assert result["median_rgb"][0] == pytest.approx(100, abs=0.5)
        assert result["mean_rgb"][0] > result["median_rgb"][0]

    def test_resampling_does_not_see_the_calibration_it_will_replace(self, session, image):
        """Otherwise every re-calibration would compound on the previous one.

        A kind's own stage is never replayed for its own sampling — only stages
        ordered strictly before it are — so re-measuring a reference always reads
        the same raw values no matter how many times it has been calibrated.
        """
        raw = service.sample_patch(session, image.id, x=20, y=15, for_kind="response")

        service.set_calibration(session, image.id, "response",
                                {"strategy": "two_patch", "black_level": 10,
                                 "white_level": 200})
        again = service.sample_patch(session, image.id, x=20, y=15, for_kind="response")

        assert again["median_rgb"] == pytest.approx(raw["median_rgb"])
        assert again["stages_applied"] == []

    def test_sampling_without_a_kind_sees_every_stage(self, session, image):
        """The plain read, used to preview what a calibration actually did."""
        service.set_calibration(session, image.id, "response",
                                {"strategy": "two_patch", "black_level": 0,
                                 "white_level": 200})
        corrected = service.sample_patch(session, image.id, x=20, y=15)
        assert corrected["stages_applied"] == ["response"]
        # 180 -> 180/200 * 255 = 229.5
        assert corrected["median_rgb"][0] == pytest.approx(229.5, abs=1.5)

    def test_sampling_outside_the_image_is_rejected(self, session, image):
        with pytest.raises(InvalidCalibrationError, match="outside"):
            service.sample_patch(session, image.id, x=WIDTH + 10, y=5)

    def test_an_absurd_radius_is_rejected(self, session, image):
        with pytest.raises(InvalidCalibrationError, match="radius"):
            service.sample_patch(session, image.id, x=20, y=15, radius=10_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(params: dict, rgb) -> list[float]:
    """Run one pixel through a normalised response calibration."""
    pixel = np.array([[list(rgb)]], dtype=np.float32)
    out = registry.get_kind("response").apply(pixel, params)
    return [float(v) for v in out[0, 0]]


def _entry_for(state: dict, kind: str) -> dict:
    return next(entry for entry in state["calibrations"] if entry["kind"] == kind)


def _seed_metrics(session, image_id: int) -> None:
    """Compute geometry and appearance metrics so there is something to invalidate."""
    from app.database.contours import Contours
    from app.services.quantification import (
        GEOMETRY_METRIC_KEYS,
        compute_and_store_metrics,
        compute_appearance_metrics_for_dataset,
    )

    dataset_id = session.query(Images.dataset_id).filter_by(id=image_id).scalar()
    compute_appearance_metrics_for_dataset(session, dataset_id, only_stale=False)

    image = session.query(Images).filter_by(id=image_id).first()
    contour_rows = (
        session.query(Contours)
        .join(Masks, Masks.id == Contours.mask_id)
        .filter(Masks.image_id == image_id)
        .all()
    )
    compute_and_store_metrics(
        session, GEOMETRY_METRIC_KEYS,
        [Contour.from_db(row) for row in contour_rows], image,
    )
    session.query(ContourMetrics).update({ContourMetrics.stale: False},
                                         synchronize_session=False)
    session.commit()


def _stale_keys(session) -> set[str]:
    return {
        row.metric_key
        for row in session.query(ContourMetrics).filter(ContourMetrics.stale.is_(True)).all()
    }


def _mean_color(session) -> list[float]:
    rows = (
        session.query(ContourMetrics)
        .filter(ContourMetrics.metric_key == "mean_color_rgb")
        .order_by(ContourMetrics.component)
        .all()
    )
    return [row.value for row in rows]
