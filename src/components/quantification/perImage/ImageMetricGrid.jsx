import React from 'react';
import SummaryMetricCard from '../SummaryMetricCard';
import {
  aggregateAllMetrics,
  aggregateMetricEntry,
  formatDelta,
  formatMeasurement,
  relativeToBaseline,
} from '../../../utils/perImageQuantification';

/**
 * The dataset comparison that turns a measurement into a finding.
 *
 * Only rendered for scalar metrics: "this image's mean colour is 12 % above the dataset's"
 * is not a statement anyone can act on, since the components move independently and a
 * percentage over a channel triple means nothing.
 */
const DatasetComparison = ({ mine, theirs, unit, baselineUnit, valueDim }) => {
  if (valueDim > 1 || mine == null || theirs == null) return null;
  const myUnit = unit || '';
  const theirUnit = baselineUnit !== undefined ? (baselineUnit || '') : myUnit;
  // A dataset comparison is only meaningful when both measurements use the same unit.
  // For mixed-scale datasets, the dataset summary falls back to pixels while a calibrated
  // image measures in physical units (e.g. mm² vs px²). Comparing them directly is
  // misleading, so hide the comparison when units differ.
  if (myUnit !== theirUnit) return null;

  const delta = relativeToBaseline(mine, theirs);
  const text = formatDelta(delta);
  if (!text) return null;

  return (
    <p className="text-[11px] text-t3 mt-3 pt-2 border-t border-ln">
      Dataset mean{' '}
      <span className="text-t2 tabular-nums">
        {formatMeasurement(theirs)} {theirUnit}
      </span>{' '}
      ·{' '}
      {/* Neither direction is good or bad — this is a measurement, not a target — so a
          large gap is emphasised but never coloured as a pass or a failure. */}
      <span className={Math.abs(delta) >= 0.1 ? 'text-t2 font-medium' : ''}>{text}</span>
    </p>
  );
};

/**
 * Every metric the active profile measures on this image, each against the dataset.
 *
 * The page used to feature exactly one metric (area) in its cards, leaving the rest of the
 * profile visible only as columns of the per-object table. A profile is a deliberate
 * choice of what to measure, so the summary shows all of it.
 *
 * Cards are `SummaryMetricCard`, the same component the dataset page's Overview uses — the
 * aggregation here produces the endpoint's own per-label shape (see `aggregateMetricEntry`)
 * precisely so that card can be reused unchanged, colour swatches included. The only
 * addition is the comparison in its footer slot.
 *
 * @param {Object} props
 * @param {Object} props.imageMetrics - `metrics` from the image-scoped summary.
 * @param {Object} props.datasetMetrics - `metrics` from the dataset-wide summary.
 * @param {Object} props.catalogMap - metric_key -> catalog entry.
 */
const ImageMetricGrid = ({ imageMetrics, datasetMetrics, catalogMap = {} }) => {
  const metrics = aggregateAllMetrics(imageMetrics, catalogMap);
  if (metrics.length === 0) return null;

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <div className="h-1 w-8 bg-accent rounded-full flex-shrink-0" />
        <h3 className="text-sm font-semibold text-t2 uppercase tracking-wide">
          Measurements on this image
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {metrics.map(({ metricKey, entry }) => {
          const catalog = catalogMap[metricKey];
          const baseline = aggregateMetricEntry(datasetMetrics, metricKey);
          return (
            <SummaryMetricCard
              key={metricKey}
              metricKey={metricKey}
              metric={entry}
              catalog={catalog}
              footer={
                <DatasetComparison
                  mine={entry.components[0]?.mean}
                  theirs={baseline?.components?.[0]?.mean}
                  unit={entry.unit || ''}
                  baselineUnit={baseline?.unit || ''}
                  valueDim={catalog?.value_dim ?? entry.components.length}
                />
              }
            />
          );
        })}
      </div>
    </div>
  );
};

export default ImageMetricGrid;
