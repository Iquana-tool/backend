"""Pure active-learning ranking: units, k-center greedy, density and uncertainty."""
import numpy as np
import pytest

from app.services import al_ranking as r


def v(*xs):
    return np.asarray(xs, dtype=float)


def test_build_units_groups_stack_frames_in_frame_order():
    units = r.build_units([(1, None, None), (2, 9, 1), (3, 9, 0), (4, None, None)])
    assert [(u.key, u.image_ids) for u in units] == [(1, (1,)), (3, (3, 2)), (4, (4,))]


def test_binary_entropy_peaks_at_fifty_fifty():
    assert r.binary_entropy(0.5) == pytest.approx(1.0)
    assert r.binary_entropy(0.9) == pytest.approx(r.binary_entropy(0.1))
    assert r.binary_entropy(1.0) == pytest.approx(0.0, abs=1e-6)
    assert r.binary_entropy(0.6) > r.binary_entropy(0.9)


def test_image_uncertainty_is_the_most_borderline_detection():
    assert r.image_uncertainty([0.95, 0.5, 0.9]) == pytest.approx(1.0)
    assert r.image_uncertainty([]) == 0.0


def test_unit_uncertainty_breaks_ties_by_borderline_count():
    unit = r.Unit(key=3, image_ids=(3, 2))
    score, borderline = r.unit_uncertainty(unit, {2: [0.5, 0.95], 3: [0.4]})
    assert score == pytest.approx(1.0)
    assert borderline == 2
    assert r.unit_uncertainty(unit, {}) is None


def test_density_favours_images_with_many_look_alikes():
    vectors = {1: v(1, 0), 2: v(0.99, 0.01), 3: v(0.98, 0.02), 4: v(0, 1)}
    scores = r.density_scores(vectors, neighbours=2)
    assert max(scores, key=scores.get) in {1, 2, 3}
    assert min(scores, key=scores.get) == 4
    assert r.density_scores({1: v(1, 0)}) == {1: 0.0}


def test_unit_vector_is_normalised_mean_and_none_without_vectors():
    unit = r.Unit(key=1, image_ids=(1, 2))
    vec = r.unit_vector(unit, {1: v(1, 0), 2: v(0, 1)})
    assert np.allclose(vec, v(1, 1) / np.sqrt(2))
    assert r.unit_vector(unit, {}) is None


def test_k_center_starts_far_from_seeds_and_spreads_out():
    candidates = {10: v(1, 0), 11: v(0.99, 0.01), 12: v(0, 1)}
    order = r.k_center_order(candidates, seeds=[v(1, 0)])
    assert order[0] == 12
    assert sorted(order) == [10, 11, 12]


def test_k_center_without_seeds_starts_farthest_from_mean():
    candidates = {1: v(1, 0), 2: v(1, 0), 3: v(1, 0), 4: v(0, 1)}
    assert r.k_center_order(candidates, seeds=[])[0] == 4


def test_k_center_handles_empty_input():
    assert r.k_center_order({}, seeds=[]) == []


def test_order_by_score_puts_unscored_last_and_uses_tie_breaks():
    assert r.order_by_score({1: 0.2, 2: None, 3: 0.9, 4: 0.5}) == [3, 4, 1, 2]
    assert r.order_by_score({1: (0.9, 1), 2: (0.9, 3)}) == [2, 1]
