import {
  aggregateAllMetrics,
  aggregateMetric,
  aggregateMetricEntry,
  countMeasuredObjects,
  formatDelta,
  frameCoverage,
  isAdditiveMetric,
  perLabelMetric,
  pickFeaturedMetric,
  relativeToBaseline,
} from './perImageQuantification';

/**
 * The per-image page recombines the summary's per-label aggregates into image-wide ones.
 * The cases worth pinning are the ones where a plausible shortcut gives a different (and
 * wrong) answer: unequal label counts, labels a metric is not scoped to, and an image the
 * dataset has no baseline for.
 */

/** `{unit, components: [...]}` as the summary endpoint returns it. */
const entry = (unit, count, mean, std = 0, min = mean, max = mean) => ({
  unit,
  components: [{ count, mean, std, min, max }],
});

describe('aggregateMetric', () => {
  const metrics = {
    1: { area: entry('mm²', 10, 5) },
    2: { area: entry('mm²', 2, 30) },
  };

  test('sums counts and totals across labels', () => {
    const result = aggregateMetric(metrics, 'area');
    expect(result.count).toBe(12);
    expect(result.total).toBe(10 * 5 + 2 * 30);
    expect(result.unit).toBe('mm²');
  });

  test('weights the mean by count, not by label', () => {
    // The mean of the per-label means would be 17.5, which no object on this image is
    // anywhere near — 10 of the 12 objects measure 5.
    expect(aggregateMetric(metrics, 'area').mean).toBeCloseTo(110 / 12);
  });

  test('can exclude unlabeled objects, which the export omits', () => {
    const withUnlabeled = { ...metrics, null: { area: entry('mm²', 5, 100) } };
    expect(aggregateMetric(withUnlabeled, 'area').count).toBe(17);
    expect(aggregateMetric(withUnlabeled, 'area', { includeUnlabeled: false }).count).toBe(12);
  });

  test('reports no mean rather than zero when nothing measured', () => {
    const result = aggregateMetric({ 1: { perimeter: entry('mm', 3, 9) } }, 'area');
    expect(result.count).toBe(0);
    expect(result.mean).toBeNull();
    expect(result.unit).toBeNull();
  });

  test('pools the spread across labels rather than averaging it', () => {
    // Two labels, no spread WITHIN either, but their means are far apart. Averaging the
    // per-label standard deviations gives 0, which claims the image is uniform when in
    // fact every object is either 10 or 20.
    const pooled = aggregateMetric(
      { 1: { area: entry('mm²', 1, 10) }, 2: { area: entry('mm²', 1, 20) } },
      'area'
    );
    expect(pooled.mean).toBeCloseTo(15);
    expect(pooled.std).toBeCloseTo(5); // population std of {10, 20}
  });

  test('takes the extremes from the widest label', () => {
    const result = aggregateMetric(
      {
        1: { area: entry('mm²', 4, 10, 2, 6, 14) },
        2: { area: entry('mm²', 2, 50, 1, 48, 52) },
      },
      'area'
    );
    expect(result.min).toBe(6);
    expect(result.max).toBe(52);
  });
});

describe('aggregateMetricEntry', () => {
  test('keeps every component, so a colour survives the aggregation', () => {
    const colour = (count, r, g, b) => ({
      unit: null,
      components: [
        { count, mean: r, std: 0, min: r, max: r },
        { count, mean: g, std: 0, min: g, max: g },
        { count, mean: b, std: 0, min: b, max: b },
      ],
    });
    const result = aggregateMetricEntry(
      { 1: { mean_color_rgb: colour(1, 0, 0, 0) }, 2: { mean_color_rgb: colour(3, 100, 200, 40) } },
      'mean_color_rgb'
    );

    expect(result.components).toHaveLength(3);
    // Count-weighted per channel: the label with three objects dominates.
    expect(result.components[0].mean).toBeCloseTo(75);
    expect(result.components[1].mean).toBeCloseTo(150);
    expect(result.components[2].mean).toBeCloseTo(30);
  });

  test('returns null when the metric is absent', () => {
    expect(aggregateMetricEntry({ 1: { area: entry('mm²', 1, 1) } }, 'perimeter')).toBeNull();
  });
});

describe('aggregateAllMetrics', () => {
  const catalog = {
    area: { tier: 'geometry' },
    perimeter: { tier: 'geometry' },
    mean_intensity: { tier: 'appearance' },
    nn_distance: { tier: 'contextual' },
  };

  test('returns every measured metric, ordered by tier then name', () => {
    const metrics = {
      1: {
        nn_distance: entry('mm', 2, 5),
        mean_intensity: entry(null, 2, 120),
        perimeter: entry('mm', 2, 40),
        area: entry('mm²', 2, 100),
      },
    };
    expect(aggregateAllMetrics(metrics, catalog).map((row) => row.metricKey)).toEqual([
      'area',
      'perimeter',
      'mean_intensity',
      'nn_distance',
    ]);
  });

  test('drops metrics with no measurements rather than emitting empty cards', () => {
    const metrics = {
      1: { area: entry('mm²', 2, 100), perimeter: { unit: 'mm', components: [] } },
    };
    expect(aggregateAllMetrics(metrics, catalog).map((row) => row.metricKey)).toEqual(['area']);
  });
});

describe('perLabelMetric', () => {
  test('orders by total contribution, not by mean', () => {
    const result = perLabelMetric(
      {
        1: { area: entry('mm²', 100, 2) },   // total 200
        2: { area: entry('mm²', 1, 50) },    // total 50, but the largest objects
      },
      'area'
    );
    expect(result.map((row) => row.labelId)).toEqual(['1', '2']);
    expect(result[0].total).toBe(200);
  });

  test('omits labels the metric is not scoped to', () => {
    const result = perLabelMetric(
      { 1: { area: entry('mm²', 4, 2) }, 2: { perimeter: entry('mm', 4, 2) } },
      'area'
    );
    expect(result).toHaveLength(1);
    expect(result[0].labelId).toBe('1');
  });
});

describe('pickFeaturedMetric', () => {
  const catalog = {
    area: { unit_kind: 'area', value_dim: 1 },
    perimeter: { unit_kind: 'length', value_dim: 1 },
    mean_color_rgb: { unit_kind: 'color', value_dim: 3 },
  };

  test('prefers area when the profile includes it', () => {
    const metrics = { 1: { perimeter: entry('mm', 1, 1), area: entry('mm²', 1, 1) } };
    expect(pickFeaturedMetric(metrics, catalog)).toBe('area');
  });

  test('prefers additive scalar metrics (like perimeter/length) over non-additive ones (like circularity)', () => {
    const extendedCatalog = {
      ...catalog,
      circularity: { unit_kind: 'ratio', value_dim: 1 },
    };
    const metrics = {
      1: {
        circularity: entry(null, 1, 0.8),
        perimeter: entry('mm', 1, 25),
      },
    };
    expect(pickFeaturedMetric(metrics, extendedCatalog)).toBe('perimeter');
  });

  test('falls back to non-additive scalar metric (like circularity) when no additive metrics exist', () => {
    const extendedCatalog = {
      circularity: { unit_kind: 'ratio', value_dim: 1 },
      mean_color_rgb: { unit_kind: 'color', value_dim: 3 },
    };
    const metrics = {
      1: {
        circularity: entry(null, 1, 0.8),
        mean_color_rgb: entry(null, 1, 1),
      },
    };
    expect(pickFeaturedMetric(metrics, extendedCatalog)).toBe('circularity');
  });

  test('falls back to a scalar metric rather than a colour', () => {
    const metrics = { 1: { mean_color_rgb: entry(null, 1, 1), perimeter: entry('mm', 1, 1) } };
    expect(pickFeaturedMetric(metrics, catalog)).toBe('perimeter');
  });

  test('returns null when nothing is measured', () => {
    expect(pickFeaturedMetric({}, catalog)).toBeNull();
  });
});

describe('isAdditiveMetric', () => {
  test('recognizes additive and non-additive unit kinds', () => {
    expect(isAdditiveMetric({ unit_kind: 'area' })).toBe(true);
    expect(isAdditiveMetric({ unit_kind: 'length' })).toBe(true);
    expect(isAdditiveMetric({ unit_kind: 'count' })).toBe(true);
    expect(isAdditiveMetric({ unit_kind: 'volume' })).toBe(true);
    expect(isAdditiveMetric({ unit_kind: 'ratio' })).toBe(false);
    expect(isAdditiveMetric({ unit_kind: 'intensity' })).toBe(false);
    expect(isAdditiveMetric({ unit_kind: 'color' })).toBe(false);
    expect(isAdditiveMetric(null)).toBe(false);
  });
});

describe('frameCoverage', () => {
  const image = { width: 100, height: 100, scale_x: 0.5, scale_y: 0.5 };

  test('converts the frame into the display unit', () => {
    // 100x100 px at 0.5 mm/px is a 50x50 mm frame = 2500 mm².
    expect(frameCoverage(250, image, { display_physical: true })).toBeCloseTo(0.1);
  });

  test('measures against pixels when the image is uncalibrated', () => {
    expect(frameCoverage(1000, image, { display_physical: false })).toBeCloseTo(0.1);
  });

  test('declines to report coverage above the frame', () => {
    // Overlapping objects can sum past the frame; "140 % of frame" reads as a bug.
    expect(frameCoverage(14000, image, { display_physical: false })).toBeNull();
  });
});

describe('relativeToBaseline / formatDelta', () => {
  test('reports a signed fraction against the dataset', () => {
    expect(relativeToBaseline(12, 10)).toBeCloseTo(0.2);
    expect(formatDelta(relativeToBaseline(12, 10))).toBe('+20.0 %');
    expect(formatDelta(relativeToBaseline(8, 10))).toBe('-20.0 %');
  });

  test('has no opinion without a baseline', () => {
    expect(relativeToBaseline(12, null)).toBeNull();
    expect(relativeToBaseline(12, 0)).toBeNull();
    expect(formatDelta(null)).toBeNull();
  });
});

describe('countMeasuredObjects', () => {
  test('counts objects across multiple labels with different metrics', () => {
    const metrics = {
      1: { area: entry('mm²', 5, 10) },
      2: { circularity: entry(null, 3, 0.8) },
    };
    expect(countMeasuredObjects(metrics)).toBe(8);
  });

  test('does not double count multiple metrics on the same label', () => {
    const metrics = {
      1: {
        area: entry('mm²', 5, 10),
        perimeter: entry('mm', 5, 20),
      },
    };
    expect(countMeasuredObjects(metrics)).toBe(5);
  });

  test('handles empty or missing metrics', () => {
    expect(countMeasuredObjects(null)).toBe(0);
    expect(countMeasuredObjects({})).toBe(0);
    expect(countMeasuredObjects({ 1: {} })).toBe(0);
  });

  test('can exclude unlabeled objects', () => {
    const metrics = {
      1: { area: entry('mm²', 5, 10) },
      null: { area: entry('mm²', 2, 8) },
    };
    expect(countMeasuredObjects(metrics)).toBe(7);
    expect(countMeasuredObjects(metrics, { includeUnlabeled: false })).toBe(5);
  });
});
