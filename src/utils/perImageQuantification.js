/**
 * Derivations for the per-image quantification view.
 *
 * The summary endpoint answers per (label, metric, component) — which is the right shape
 * for the label tree on the dataset page, but not for a page whose subject is one image.
 * There the questions are image-wide ("how many objects, how much area, how does that
 * compare to the dataset?"), so the per-label entries have to be recombined. Doing that
 * here, on pure functions over the response, keeps the arithmetic testable and keeps the
 * page components to layout.
 *
 * Everything below reads the ``metrics`` mapping the endpoint returns:
 * ``{ [labelId]: { [metricKey]: { unit, components: [{count, mean, std, min, max}, ...] } } }``
 */

/** The label id the endpoint uses for objects with no label assigned. */
const UNLABELED_KEY = "null";

/**
 * Recombine one metric's per-label aggregates into ONE image-wide aggregate, in the exact
 * shape the summary endpoint uses per label: `{unit, components: [{count, mean, std, min,
 * max}, ...]}`.
 *
 * Producing the endpoint's own shape rather than a bespoke one is what lets the per-image
 * page render every metric with the same cards as the dataset page — including the colour
 * swatch, which needs all three components rather than a single featured number.
 *
 * The mean is `Σ(count × mean) / Σcount`, a count-weighted mean and NOT the mean of the
 * per-label means: with unequal label counts those differ, and the weighted one is what
 * the server would report for the same set of objects.
 *
 * The standard deviation is likewise pooled rather than averaged, via `E[x²] - E[x]²` —
 * averaging per-label standard deviations would understate the spread by throwing away
 * the variance *between* labels, which on a mixed image is most of it.
 *
 * @param {Object} metricsByLabelId - The summary's `metrics` mapping.
 * @param {string} metricKey
 * @param {Object} [options]
 * @param {boolean} [options.includeUnlabeled=true] - Whether objects with no label count.
 * @returns {{unit: string|null, components: Array<Object>}|null} null when nothing measured.
 */
export const aggregateMetricEntry = (
  metricsByLabelId,
  metricKey,
  { includeUnlabeled = true } = {}
) => {
  let unit = null;
  // component index -> running totals
  const acc = new Map();

  Object.entries(metricsByLabelId || {}).forEach(([labelId, metrics]) => {
    if (!includeUnlabeled && labelId === UNLABELED_KEY) return;
    const entry = metrics?.[metricKey];
    if (!entry?.components) return;
    if (unit == null && entry.unit) unit = entry.unit;

    entry.components.forEach((stats, index) => {
      if (!stats || !stats.count) return;
      const running = acc.get(index) || {
        count: 0, sum: 0, sumSq: 0, min: Infinity, max: -Infinity,
      };
      running.count += stats.count;
      running.sum += stats.count * stats.mean;
      // E[x²] contribution for this label: n(σ² + μ²) is the sum of squares behind it.
      running.sumSq += stats.count * ((stats.std ?? 0) ** 2 + stats.mean ** 2);
      running.min = Math.min(running.min, stats.min);
      running.max = Math.max(running.max, stats.max);
      acc.set(index, running);
    });
  });

  if (acc.size === 0) return null;

  const components = [...acc.keys()]
    .sort((a, b) => a - b)
    .map((index) => {
      const { count, sum, sumSq, min, max } = acc.get(index);
      const mean = sum / count;
      // Clamped: float noise can push a genuinely-zero variance very slightly negative.
      const variance = Math.max(0, sumSq / count - mean * mean);
      return { count, mean, std: Math.sqrt(variance), min, max };
    });

  return { unit, components };
};
/**
 * The scalar view of {@link aggregateMetricEntry}: the first component, plus the sum.
 *
 * Only the first component is totalled, because the metrics worth totalling are scalars
 * (area, perimeter, ...) and a mean colour has no meaningful sum.
 *
 * @returns {{count: number, total: number, mean: number|null, std: number|null,
 *   min: number|null, max: number|null, unit: string|null}}
 */
export const aggregateMetric = (metricsByLabelId, metricKey, options = {}) => {
  const entry = aggregateMetricEntry(metricsByLabelId, metricKey, options);
  const stats = entry?.components?.[0];
  if (!stats) {
    return { count: 0, total: 0, mean: null, std: null, min: null, max: null, unit: null };
  }
  return {
    count: stats.count,
    total: stats.count * stats.mean,
    mean: stats.mean,
    std: stats.std,
    min: stats.min,
    max: stats.max,
    unit: entry.unit,
  };
};

/**
 * Every metric measured on this image, aggregated image-wide and ordered for display.
 *
 * The page featured exactly one metric before this: four cards built around area, with
 * every other metric in the active profile reachable only by reading the per-object table
 * column by column. A profile is a deliberate choice of what to measure, so all of it
 * belongs on the page.
 *
 * @param {Object} metricsByLabelId - The summary's `metrics` mapping.
 * @param {Object} catalogMap - metric_key -> catalog entry, used for the display order.
 * @returns {Array<{metricKey: string, entry: Object}>}
 */
export const aggregateAllMetrics = (metricsByLabelId, catalogMap = {}) => {
  const keys = new Set();
  Object.values(metricsByLabelId || {}).forEach((metrics) => {
    Object.keys(metrics || {}).forEach((key) => keys.add(key));
  });

  return [...keys]
    .map((metricKey) => ({ metricKey, entry: aggregateMetricEntry(metricsByLabelId, metricKey) }))
    .filter((row) => row.entry)
    // Grouped by tier so geometry (what most people came for) leads and the derived tiers
    // follow, then alphabetical inside a tier for a stable order across images.
    .sort((a, b) => {
      const tierOrder = ["geometry", "appearance", "contextual", "relational"];
      const tierOf = (key) => {
        const index = tierOrder.indexOf(catalogMap[key]?.tier);
        return index === -1 ? tierOrder.length : index;
      };
      const byTier = tierOf(a.metricKey) - tierOf(b.metricKey);
      return byTier !== 0 ? byTier : a.metricKey.localeCompare(b.metricKey);
    });
};

/**
 * One metric's aggregate per label, for the "this image vs dataset" comparison.
 *
 * Labels with no measurement are omitted rather than emitted as zero: an absent label and
 * a label that measured zero are different claims, and only one of them is true here.
 *
 * @returns {Array<{labelId: string, count: number, mean: number, total: number, unit: string|null}>}
 *   sorted by descending total, so the biggest contributor to the image is read first.
 */
export const perLabelMetric = (metricsByLabelId, metricKey) =>
  Object.entries(metricsByLabelId || {})
    .map(([labelId, metrics]) => {
      const entry = metrics?.[metricKey];
      const stats = entry?.components?.[0];
      if (!stats || !stats.count) return null;
      return {
        labelId,
        count: stats.count,
        mean: stats.mean,
        total: stats.count * stats.mean,
        unit: entry.unit || null,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.total - a.total);

/**
 * Tests whether a metric's values can be meaningfully summed across objects (e.g. area, perimeter).
 * Non-additive metrics (circularity, mean intensity) should be averaged rather than totalled.
 */
export const isAdditiveMetric = (catalogEntry) => {
  const kind = catalogEntry?.unit_kind;
  return kind === "area" || kind === "length" || kind === "volume" || kind === "count";
};

/**
 * The metric the summary cards and the comparison bars are built around.
 *
 * Area first, because it is what an object's size means to most people and it is in every
 * default profile. Failing that, any other additive metric (perimeter/length, count), then
 * any single-component numeric one — so a profile that deliberately drops area still gets
 * a populated page rather than four empty cards.
 *
 * @param {Object} metricsByLabelId - The summary's `metrics` mapping.
 * @param {Object} catalogMap - metric_key -> catalog entry.
 * @returns {string|null} The metric key, or null when nothing measurable is in scope.
 */
export const pickFeaturedMetric = (metricsByLabelId, catalogMap = {}) => {
  const present = new Set();
  Object.values(metricsByLabelId || {}).forEach((metrics) => {
    Object.keys(metrics || {}).forEach((key) => present.add(key));
  });
  if (present.size === 0) return null;
  if (present.has("area")) return "area";

  const keys = [...present].sort();
  const byUnitKind = (kind) =>
    keys.find((key) => catalogMap[key]?.unit_kind === kind);
  return (
    byUnitKind("area") ||
    byUnitKind("length") ||
    byUnitKind("volume") ||
    byUnitKind("count") ||
    keys.find((key) => (catalogMap[key]?.value_dim ?? 1) === 1 && catalogMap[key]?.unit_kind !== "color") ||
    null
  );
};

/**
 * What fraction of the frame the measured objects cover, or null when it cannot be said.
 *
 * Only meaningful for an area metric, and only when the image's own pixel dimensions are
 * known. The frame area is converted into the same display unit the metric is reported in
 * (`scale_x * scale_y` per pixel), so the ratio is unit-free either way.
 *
 * Returns null rather than a number above 1: overlapping objects can legitimately sum past
 * the frame, and a "140 % of frame" caption reads as a bug rather than as the overlap it
 * actually describes.
 *
 * @param {number} totalArea - Summed area in display units.
 * @param {{width: number, height: number, scale_x: number, scale_y: number}} image
 * @param {{display_physical: boolean}} scaleStatus
 * @returns {number|null} A fraction in [0, 1].
 */
export const frameCoverage = (totalArea, image, scaleStatus) => {
  if (!totalArea || !image?.width || !image?.height) return null;
  const factor = scaleStatus?.display_physical
    ? (image.scale_x || 1) * (image.scale_y || 1)
    : 1;
  const frameArea = image.width * image.height * factor;
  if (!frameArea) return null;
  const fraction = totalArea / frameArea;
  return fraction > 1 ? null : fraction;
};

/**
 * How far this image's value sits from the dataset's, as a signed fraction.
 *
 * The page's whole reason for loading the dataset summary alongside the image's is this
 * one number: an image is interesting when it disagrees with the dataset it belongs to.
 *
 * @returns {number|null} e.g. 0.217 for "21.7 % above the dataset mean", or null when
 *   there is no baseline to compare against.
 */
export const relativeToBaseline = (value, baseline) => {
  if (value == null || baseline == null || !Number.isFinite(baseline) || baseline === 0) {
    return null;
  }
  return (value - baseline) / baseline;
};

/**
 * Format a measurement for a card or a bar label.
 *
 * Significant digits rather than fixed decimals: these numbers span pixel areas in the
 * hundreds of thousands and millimetre areas below one, and a fixed `toFixed(2)` renders
 * one of those as noise and the other as "0.00".
 */
export const formatMeasurement = (value, { digits = 3 } = {}) => {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 10000 || magnitude < 0.001) return value.toExponential(2);
  return Number(value.toPrecision(digits)).toLocaleString(undefined, {
    maximumSignificantDigits: digits,
  });
};

/** A signed percentage for the "vs dataset" captions, e.g. "+21.7 %". */
export const formatDelta = (fraction) => {
  if (fraction == null) return null;
  const percent = fraction * 100;
  const rounded = Math.abs(percent) < 0.1 ? 0 : percent;
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(1)} %`;
};

/**
 * Total number of distinct objects on the image having at least one measurement.
 *
 * Each label's measured count is the maximum count among that label's metrics,
 * and the image total is the sum over all labels. This avoids undercounting
 * when metrics are scoped by label (e.g. area only on Cells, intensity on Nuclei)
 * or when contextual metrics apply to a subset of classes.
 *
 * @param {Object} metricsByLabelId - The summary's `metrics` mapping.
 * @param {Object} [options]
 * @param {boolean} [options.includeUnlabeled=true]
 * @returns {number}
 */
export const countMeasuredObjects = (metricsByLabelId, { includeUnlabeled = true } = {}) => {
  if (!metricsByLabelId || typeof metricsByLabelId !== "object") return 0;
  return Object.entries(metricsByLabelId).reduce((total, [labelId, labelMetrics]) => {
    if (!includeUnlabeled && labelId === UNLABELED_KEY) return total;
    if (!labelMetrics || typeof labelMetrics !== "object") return total;
    let maxForLabel = 0;
    for (const metricData of Object.values(labelMetrics)) {
      if (!metricData?.components) continue;
      for (const comp of metricData.components) {
        if (typeof comp?.count === "number" && comp.count > maxForLabel) {
          maxForLabel = comp.count;
        }
      }
    }
    return total + maxForLabel;
  }, 0);
};
