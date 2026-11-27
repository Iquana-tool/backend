"""Saving an unclipped prediction must not quantize its original boundary."""

import numpy as np
import pytest
from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.database.contours import Contour

from app.services.database_access.masks import add_contour_to_hierarchy, fit_contour_at_image_resolution


def box(x0, y0, x1, y1, **kwargs):
    return Contour(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], **kwargs)


@pytest.mark.parametrize("with_parent", [False, True])
@pytest.mark.parametrize("small", [False, True])
def test_unclipped_prediction_keeps_exact_native_coordinates(with_parent, small):
    # Native image coordinates rarely coincide with the hierarchy's 1000px grid.
    width, height = 4096, 3072
    points = np.array([(521, 502), (543, 493), (566, 518), (548, 545), (519, 531)])
    if not small:
        points = points * 3
    prediction = Contour(
        label_id=2,
        parent_id=1 if with_parent else None,
        x=(points[:, 0] / width).tolist(),
        y=(points[:, 1] / height).tolist(),
        confidence=0.93,
        added_by="model",
    )
    hierarchy = ContourHierarchy()
    if with_parent:
        parent = box(0.05, 0.05, 0.9, 0.9, id=1, label_id=1)
        hierarchy = ContourHierarchy(root_contours=[parent], id_to_contour={1: parent})
    before = prediction.model_dump()

    saved = add_contour_to_hierarchy(hierarchy, prediction)

    assert saved is not None
    assert saved.model_dump() == before
    assert hierarchy.label_id_to_contours[2] == [saved]
    if with_parent:
        assert hierarchy.id_to_contour[1].children == [saved]
    else:
        assert hierarchy.root_contours == [saved]


def test_prediction_crossing_parent_is_still_clipped():
    parent = box(0.2, 0.2, 0.8, 0.8, id=1, label_id=1)
    hierarchy = ContourHierarchy(root_contours=[parent], id_to_contour={1: parent})
    prediction = box(0.7, 0.3, 0.9, 0.6, label_id=2, parent_id=1)
    parent_mask = parent.to_binary_mask(1000, 1000)
    expected = prediction.to_binary_mask(1000, 1000) & parent_mask

    saved = add_contour_to_hierarchy(hierarchy, prediction)

    assert saved is not None
    assert np.array_equal(saved.to_binary_mask(1000, 1000), expected)
    assert max(saved.x) <= 0.8
    assert saved.parent_id == 1


def test_overlapping_peer_is_still_clipped():
    peer = box(0.2, 0.2, 0.5, 0.6, id=1, label_id=2)
    hierarchy = ContourHierarchy(root_contours=[peer], id_to_contour={1: peer})
    prediction = box(0.4, 0.3, 0.7, 0.5, label_id=2)

    saved = add_contour_to_hierarchy(hierarchy, prediction)

    assert saved is not None
    assert not np.any(saved.to_binary_mask(1000, 1000) & peer.to_binary_mask(1000, 1000))
    assert min(saved.x) > 0.5


def test_fully_suppressed_prediction_is_not_restored():
    peer = box(0.2, 0.2, 0.8, 0.8, id=1, label_id=2)
    hierarchy = ContourHierarchy(root_contours=[peer], id_to_contour={1: peer})

    assert add_contour_to_hierarchy(hierarchy, box(0.3, 0.3, 0.4, 0.4, label_id=2)) is None
    assert hierarchy.root_contours == [peer]


def test_child_protruding_by_few_pixels_is_clipped_at_image_resolution():
    # 4000 x 3000 image: 1px on 1000x1000 grid corresponds to 4px along width.
    # Parent boundary at x=2000 (0.5).
    # Child extends 3 pixels outside parent (x=2003 / 4000 = 0.50075).
    width, height = 4000, 3000

    def make_hierarchy():
        p = box(0.1, 0.1, 0.5, 0.9, id=1, label_id=1)
        h = ContourHierarchy(root_contours=[p], id_to_contour={1: p})
        h._image_shape = (height, width)
        return h

    # Child extending 3 pixels past parent
    child_3px = box(0.2, 0.2, 2003 / width, 0.8, label_id=2, parent_id=1)
    saved_3px = add_contour_to_hierarchy(make_hierarchy(), child_3px, image_shape=(height, width))

    assert saved_3px is not None
    # Must NOT preserve the protruding original vertices
    assert saved_3px.x != child_3px.x
    assert max(saved_3px.x) <= 0.5

    # Also test resolution read directly from hierarchy._image_shape
    saved_3px_auto = add_contour_to_hierarchy(make_hierarchy(), child_3px)
    assert saved_3px_auto is not None
    assert saved_3px_auto.x != child_3px.x
    assert max(saved_3px_auto.x) <= 0.5

    # Child strictly inside parent must preserve exact native coordinates
    child_inside = box(0.2, 0.2, 1990 / width, 0.8, label_id=2, parent_id=1)
    saved_inside = add_contour_to_hierarchy(make_hierarchy(), child_inside, image_shape=(height, width))

    assert saved_inside is not None
    assert saved_inside.x == child_inside.x


def test_sibling_exclusion_preserved_when_candidate_overlaps_by_few_pixels():
    # 4001 x 3000 image: Sibling at x up to 252/w. Candidate starts at 253/w.
    # On 1000x1000 coarse grid: sibling is at col 62, candidate starts at col 63 (changed == False).
    # At image resolution: both overlap at column 252.
    # This verifies that cropped refit at image resolution is entered and eliminates the overlap.
    width, height = 4001, 3000
    parent = box(0.01, 0.01, 0.9, 0.9, id=1, label_id=1)
    sibling = box(100 / width, 0.2, 252 / width, 0.4, id=2, label_id=2, parent_id=1)
    parent.add_child(sibling)
    hierarchy = ContourHierarchy(root_contours=[parent], id_to_contour={1: parent, 2: sibling})
    hierarchy._image_shape = (height, width)

    candidate = box(253 / width, 0.2, 500 / width, 0.4, id=3, label_id=2, parent_id=1)

    # Coarse grid must not detect the overlap so cropped refit path is exercised
    allowed_1000 = hierarchy.get_parent_mask(1, (1000, 1000)) & ~sibling.to_binary_mask(1000, 1000)
    _, changed_coarse = candidate.fit_to_mask(allowed_1000)
    assert not changed_coarse

    saved = add_contour_to_hierarchy(hierarchy, candidate, image_shape=(height, width))

    assert saved is not None
    assert saved.x != candidate.x

    sib_mask = sibling.to_binary_mask(height, width)
    saved_mask = saved.to_binary_mask(height, width)
    assert np.count_nonzero(sib_mask & saved_mask) == 0


def test_sibling_exclusion_maintained_when_refitting_against_parent():
    # 4001 x 3000 image: Parent starts at pixel 508. Sibling is at [1005, 1500].
    # Candidate extends 2px outside parent (pixel 506) and adjacent to sibling (pixel 1004).
    # On 1000x1000 coarse grid: both 506 and 508 map to col 126, and 1004 is col 250 vs sibling col 251.
    # Coarse grid fit has changed == False.
    # Cropped refit at image resolution clips candidate against parent AND respects sibling exclusion.
    width, height = 4001, 3000
    parent = box(508.1 / width, 0.1, 2000 / width, 0.9, id=1, label_id=1)
    sibling = box(1005 / width, 0.2, 1500 / width, 0.4, id=2, label_id=2, parent_id=1)
    parent.add_child(sibling)
    hierarchy = ContourHierarchy(root_contours=[parent], id_to_contour={1: parent, 2: sibling})
    hierarchy._image_shape = (height, width)

    candidate = box(506 / width, 0.2, 1004 / width, 0.4, id=3, label_id=2, parent_id=1)

    # Coarse grid fit must have changed == False
    allowed_1000 = hierarchy.get_parent_mask(1, (1000, 1000)) & ~sibling.to_binary_mask(1000, 1000)
    _, changed_coarse = candidate.fit_to_mask(allowed_1000)
    assert not changed_coarse

    saved = add_contour_to_hierarchy(hierarchy, candidate, image_shape=(height, width))

    assert saved is not None
    assert saved.x != candidate.x

    sib_mask = sibling.to_binary_mask(height, width)
    parent_mask = parent.to_binary_mask(height, width)
    saved_mask = saved.to_binary_mask(height, width)

    assert np.count_nonzero(sib_mask & saved_mask) == 0
    assert np.count_nonzero(saved_mask & ~parent_mask) == 0


def test_round_trip_containment_at_4001x3000():
    # 4001 x 3000 image: Parent starts at column 508 (pixel 508).
    # Child extends to column 506.
    # Coarse 1000x1000 grid maps both to column 126 (changed == False).
    # Converting the clipped boundary back to normalized coordinates and rasterizing it
    # must not shift the left edge from column 508 to 507 due to floating-point truncation.
    width, height = 4001, 3000
    parent = box(508.1 / width, 0.1, 2000 / width, 0.9, id=1, label_id=1)
    child = box(506 / width, 0.2, 1000 / width, 0.4, id=2, label_id=2, parent_id=1)
    hierarchy = ContourHierarchy(root_contours=[parent], id_to_contour={1: parent})
    hierarchy._image_shape = (height, width)

    allowed_1000 = hierarchy.get_parent_mask(1, (1000, 1000))
    _, changed_coarse = child.fit_to_mask(allowed_1000)
    assert not changed_coarse

    saved = add_contour_to_hierarchy(hierarchy, child, image_shape=(height, width))

    assert saved is not None
    p_mask = parent.to_binary_mask(height, width)
    s_mask = saved.to_binary_mask(height, width)

    assert np.count_nonzero(s_mask & ~p_mask) == 0
    assert np.where(s_mask)[1].min() == 508


def test_large_image_cropped_bounding_box_preserves_coordinates():
    # 10000 x 8000 image: Bounding box crop avoids full-resolution full-image allocation.
    width, height = 10000, 8000
    parent = box(0.1, 0.1, 0.9, 0.9, id=1, label_id=1)
    hierarchy = ContourHierarchy(root_contours=[parent], id_to_contour={1: parent})
    hierarchy._image_shape = (height, width)

    child = box(0.5, 0.5, 0.5 + 200 / width, 0.5 + 200 / height, label_id=2, parent_id=1)
    saved = add_contour_to_hierarchy(hierarchy, child, image_shape=(height, width))

    assert saved is not None
    assert saved.x == child.x
    assert saved.y == child.y


def test_diagonal_edge_crossing_crop_boundary_has_zero_outside_pixels():
    # 4000 x 3000 image: Parent polygon has a diagonal edge crossing the crop window.
    # Child polygon straddles the diagonal edge.
    # When refitting, clipped contour must be validated against full-image raster
    # and have exactly zero pixels outside parent full-image raster.
    width, height = 4000, 3000
    parent_pts = np.array([(500, 200), (3500, 2500), (500, 2500)])
    parent = Contour(x=(parent_pts[:, 0] / width).tolist(), y=(parent_pts[:, 1] / height).tolist(), id=1, label_id=1)

    cx = 2000
    yd = 200 + (cx - 500) * (2300 / 3000)
    pts = np.array([
        (cx - 30, yd - 30),
        (cx + 30, yd + 30),
        (cx - 30, yd + 30),
    ])
    child = Contour(x=(pts[:, 0] / width).tolist(), y=(pts[:, 1] / height).tolist(), id=2, label_id=2, parent_id=1)

    saved, changed = fit_contour_at_image_resolution(child, parent, [], (height, width))
    assert changed is True
    assert saved is not None

    parent_mask = parent.to_binary_mask(height, width)
    saved_mask = saved.to_binary_mask(height, width)
    assert np.count_nonzero(saved_mask & ~parent_mask) == 0


def test_unconstrained_root_contour_skips_rasterization():
    # 10000 x 8000 image: An unconstrained root contour with no competing siblings
    # must skip all rasterization and immediately preserve original vertices.
    import time
    width, height = 10000, 8000
    pts = np.array([(1000, 1000), (9000, 1000), (9000, 7000), (1000, 7000)])
    root = Contour(x=(pts[:, 0] / width).tolist(), y=(pts[:, 1] / height).tolist(), id=1, label_id=1, parent_id=None)
    hierarchy = ContourHierarchy(root_contours=[], id_to_contour={}, label_id_to_contours={})
    hierarchy._image_shape = (height, width)

    t0 = time.perf_counter()
    saved = add_contour_to_hierarchy(hierarchy, root, image_shape=(height, width))
    duration_ms = (time.perf_counter() - t0) * 1000

    assert saved is not None
    assert saved.x == root.x
    assert saved.y == root.y
    assert duration_ms < 20.0


def test_empty_geometry_root_contour_rejected():
    # An empty or incomplete contour (x=[], y=[]) must be rejected before the root fast path
    # and must return None without creating an entry in the hierarchy.
    hierarchy = ContourHierarchy(root_contours=[], id_to_contour={}, label_id_to_contours={})
    empty_root = Contour(x=[], y=[], label_id=1, parent_id=None)
    saved = add_contour_to_hierarchy(hierarchy, empty_root)
    assert saved is None
    assert len(hierarchy.root_contours) == 0

    # Also reject contours with fewer than 3 points
    line_root = Contour(x=[0.1, 0.2], y=[0.1, 0.2], label_id=1, parent_id=None)
    saved_line = add_contour_to_hierarchy(hierarchy, line_root)
    assert saved_line is None
    assert len(hierarchy.root_contours) == 0


def test_preserved_contour_validated_against_full_image_raster():
    # 4000 x 3000 image: A candidate contour that passes the cropped check
    # and has all vertices strictly inside the parent polygon, but has pixels outside
    # the parent in the full-image raster due to OpenCV diagonal scanline discrepancy,
    # must NOT be preserved as-is. It must be refitted against the full-image raster to have 0 outside pixels.
    import cv2

    width, height = 4000, 3000
    parent_pts = np.array([(500, 200), (3500, 2500), (500, 2500)])
    parent = Contour(x=(parent_pts[:, 0] / width).tolist(), y=(parent_pts[:, 1] / height).tolist(), id=1, label_id=1)
    full_parent_mask = parent.to_binary_mask(height, width)

    # Place a child whose vertices are all strictly inside the parent polygon,
    # but which has 5 outside pixels on the full-image parent raster
    pts = np.array([[622.0, 293.75], [675.0, 335.75], [622.0, 335.75]])
    child = Contour(x=(pts[:, 0] / width).tolist(), y=(pts[:, 1] / height).tolist(), id=2, label_id=2, parent_id=1)

    # 1. Verify that all vertices are mathematically inside parent polygon
    p_pts_f = np.column_stack([np.array(parent.x) * width, np.array(parent.y) * height]).astype(np.float32)
    for x, y in zip(child.x, child.y):
        dist = cv2.pointPolygonTest(p_pts_f, (float(x * width), float(y * height)), measureDist=False)
        assert dist > 0, f"Vertex was not strictly inside parent: {dist}"

    # 2. Verify that the raw unclipped child has exactly 5 pixels outside on the full-image raster
    raw_child_mask = child.to_binary_mask(height, width)
    raw_outside = np.count_nonzero(raw_child_mask & ~full_parent_mask)
    assert raw_outside == 5

    # 3. fit_contour_at_image_resolution must detect the outside pixels, reject direct preservation (changed=True),
    # and refit the contour to achieve exactly 0 outside pixels.
    saved, changed = fit_contour_at_image_resolution(child, parent, [], (height, width))
    assert changed is True
    assert saved is not None
    saved_mask = saved.to_binary_mask(height, width)
    outside = np.count_nonzero(saved_mask & ~full_parent_mask)
    assert outside == 0


def test_child_well_inside_parent_skips_full_image_rasterization():
    # On a large 10,000 x 8,000 image, fitting a small child well inside its parent
    # must not allocate or rebuild full-image masks, completing in under 10 ms.
    import time

    width, height = 10000, 8000
    parent_pts = np.array([(500, 200), (9500, 200), (9500, 7500), (500, 7500)])
    parent = Contour(x=(parent_pts[:, 0] / width).tolist(), y=(parent_pts[:, 1] / height).tolist(), id=1, label_id=1)

    child_pts = np.array([(4000, 4000), (4100, 4000), (4100, 4100), (4000, 4100)])
    child = Contour(x=(child_pts[:, 0] / width).tolist(), y=(child_pts[:, 1] / height).tolist(), id=2, label_id=2, parent_id=1)

    t0 = time.perf_counter()
    saved, changed = fit_contour_at_image_resolution(child, parent, [], (height, width))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert changed is False
    assert saved is not None
    assert saved.x == child.x
    assert saved.y == child.y
    assert elapsed_ms < 10.0, f"Expected < 10 ms, took {elapsed_ms:.2f} ms"
