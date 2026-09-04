import React from 'react';
import { Boxes, Ruler, Sigma } from 'lucide-react';
import {
  aggregateMetric,
  countMeasuredObjects,
  formatMeasurement,
  frameCoverage,
  isAdditiveMetric,
} from '../../../utils/perImageQuantification';

const Card = ({ icon: Icon, label, value, unit, hint }) => (
  <div className="bg-p1 rounded-lg border border-ln p-4">
    <div className="flex items-center gap-2 mb-2">
      <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-acS text-ac">
        <Icon className="w-4 h-4" />
      </span>
      <h3 className="text-sm font-medium text-t2">{label}</h3>
    </div>
    <p className="text-2xl font-bold text-t1 leading-tight">
      {value}
      {unit && <span className="ml-1 text-base font-normal text-t2">{unit}</span>}
    </p>
    {hint && <p className="text-xs text-t3 mt-1">{hint}</p>}
  </div>
);

/**
 * Headline stats about the current image.
 *
 * Three cards across the top of the details panel:
 *   1. Object census: total on the image, and how many the current toggles exclude.
 *   2. Featured metric: sum of area (or the profile's primary scalar) across the measured
 *      objects, plus what fraction of the frame they cover when the units allow it.
 *   3. Scale: the image's physical calibration, or an explicit "Not calibrated" so the
 *      reader is never left guessing whether a number is in microns or pixels.
 */
const ImageStatCards = ({
  imageMetrics,
  metricKey,
  catalog,
  objectCounts,
  image,
  scaleStatus,
  scaleError = false,
}) => {
  const metricName = catalog?.name || metricKey || 'metric';
  const isArea = catalog?.unit_kind === 'area' || metricKey === 'area';
  const isAdditive = isAdditiveMetric(catalog) || metricKey === 'area';

  const here = metricKey ? aggregateMetric(imageMetrics, metricKey) : null;
  const unit = here?.unit || '';
  const featuredCount = here?.count || 0;

  // Determine calibration: if scaleStatus reports physical display, or if the image has a
  // saved pixel scale with a physical unit (not 'px'). This handles the case where all objects
  // are excluded by filters and scaleStatus falls back to display_physical: false even though
  // the image itself has a calibrated pixel scale.
  const isCalibrated = Boolean(
    !scaleError && (scaleStatus?.display_physical || (image?.unit && image.unit !== 'px'))
  );
  const displayUnit = (scaleStatus?.display_physical ? scaleStatus.display_unit : image?.unit) || '';

  // The census counts every object on the image, including ones the inclusion toggles
  // exclude from the measurements — so the two numbers can legitimately disagree, and the
  // card says by how much rather than letting the reader discover it in the table.
  const census = Object.values(objectCounts || {}).reduce(
    (sum, counts) => sum + (counts?.total || 0),
    0
  );
  // Count measured objects across all labels and metrics in the profile.
  // Using here.count (the featured metric) undercounts whenever a profile scopes
  // metrics by label or has contextual metrics, falsely reporting validly measured
  // objects as excluded.
  const measured = countMeasuredObjects(imageMetrics);
  const excluded = Math.max(0, census - measured);
  const coverage = isArea && !scaleError && (!isCalibrated || image?.scale_x != null)
    ? frameCoverage(here?.total, image, { display_physical: isCalibrated })
    : null;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <Card
        icon={Boxes}
        label="Objects"
        value={census.toLocaleString()}
        hint={
          excluded > 0
            ? `${measured.toLocaleString()} measured · ${excluded.toLocaleString()} excluded`
            : `${measured.toLocaleString()} measured`
        }
      />
      <Card
        icon={Sigma}
        label={isAdditive ? `Total ${metricName.toLowerCase()}` : `Mean ${metricName.toLowerCase()}`}
        value={formatMeasurement(isAdditive ? here?.total : here?.mean)}
        unit={unit}
        hint={
          isAdditive
            ? (coverage != null
                ? `${(coverage * 100).toFixed(1)} % of frame`
                : `summed over ${featuredCount.toLocaleString()} object${featuredCount === 1 ? '' : 's'}`)
            : (featuredCount > 0
                ? `averaged over ${featuredCount.toLocaleString()} object${featuredCount === 1 ? '' : 's'}`
                : '')
        }
      />
      <Card
        icon={Ruler}
        label="Scale"
        value={
          scaleError
            ? 'Scale unavailable'
            : isCalibrated
            ? (image?.scale_x != null ? `1 px = ${formatMeasurement(image?.scale_x)}` : '…')
            : 'Not calibrated'
        }
        unit={!scaleError && isCalibrated && image?.scale_x != null ? displayUnit : ''}
        hint={
          scaleError
            ? 'could not load calibration'
            : isCalibrated
            ? 'measurements are in real-world units'
            : 'measurements are in pixels'
        }
      />
    </div>
  );
};

export default ImageStatCards;
