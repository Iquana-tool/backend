import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ImageStatCards from './ImageStatCards';

describe('ImageStatCards calibration determination', () => {
  const catalog = {
    name: 'Area',
    unit_kind: 'area',
  };

  it('determines calibration from saved pixel scale when scaleStatus falls back to display_physical: false', () => {
    // When all objects are excluded by filters, backend scale_status reports display_physical: false
    // even though the image has a calibrated pixel scale.
    const imageWithSavedScale = {
      width: 1000,
      height: 1000,
      scale_x: 0.05,
      scale_y: 0.05,
      unit: 'mm',
    };
    const scaleStatus = {
      display_physical: false,
      display_unit: 'px',
      consistent: false,
    };

    render(
      <ImageStatCards
        imageMetrics={{}}
        metricKey="area"
        catalog={catalog}
        objectCounts={{ 1: { total: 5 } }}
        image={imageWithSavedScale}
        scaleStatus={scaleStatus}
      />
    );

    // It should NOT display "Not calibrated"
    expect(screen.queryByText('Not calibrated')).not.toBeInTheDocument();
    // It should display the saved scale
    expect(screen.getByText('1 px = 0.05')).toBeInTheDocument();
    expect(screen.getByText('mm')).toBeInTheDocument();
    expect(screen.getByText('measurements are in real-world units')).toBeInTheDocument();
  });

  it('displays "Not calibrated" when the image is truly uncalibrated', () => {
    const uncalibratedImage = {
      width: 1000,
      height: 1000,
      scale_x: 1,
      scale_y: 1,
      unit: 'px',
    };
    const scaleStatus = {
      display_physical: false,
      display_unit: 'px',
      consistent: true,
    };

    render(
      <ImageStatCards
        imageMetrics={{}}
        metricKey="area"
        catalog={catalog}
        objectCounts={{ 1: { total: 5 } }}
        image={uncalibratedImage}
        scaleStatus={scaleStatus}
      />
    );

    expect(screen.getByText('Not calibrated')).toBeInTheDocument();
    expect(screen.getByText('measurements are in pixels')).toBeInTheDocument();
  });

  it('displays calibration when scaleStatus.display_physical is true', () => {
    const image = {
      width: 1000,
      height: 1000,
      scale_x: 0.5,
      scale_y: 0.5,
      unit: 'mm',
    };
    const scaleStatus = {
      display_physical: true,
      display_unit: 'mm',
      consistent: true,
    };

    render(
      <ImageStatCards
        imageMetrics={{}}
        metricKey="area"
        catalog={catalog}
        objectCounts={{ 1: { total: 5 } }}
        image={image}
        scaleStatus={scaleStatus}
      />
    );

    expect(screen.getByText('1 px = 0.5')).toBeInTheDocument();
    expect(screen.getByText('mm')).toBeInTheDocument();
    expect(screen.getByText('measurements are in real-world units')).toBeInTheDocument();
  });

  it('avoids displaying a numeric scale when scaleStatus.display_physical is true but image.scale_x is pending (null)', () => {
    const pendingImage = {
      width: 1000,
      height: 1000,
      scale_x: null,
      scale_y: null,
      unit: null,
    };
    const scaleStatus = {
      display_physical: true,
      display_unit: 'mm',
      consistent: true,
    };

    render(
      <ImageStatCards
        imageMetrics={{}}
        metricKey="area"
        catalog={catalog}
        objectCounts={{ 1: { total: 5 } }}
        image={pendingImage}
        scaleStatus={scaleStatus}
      />
    );

    // It should NOT display fabricated numeric scale like "1 px = 1 mm" or "1 px = 1"
    expect(screen.queryByText(/1 px =/)).not.toBeInTheDocument();
    expect(screen.getByText('…')).toBeInTheDocument();
    expect(screen.getByText('measurements are in real-world units')).toBeInTheDocument();
  });

  it('displays Mean and averages non-additive metrics like circularity', () => {
    const circularityCatalog = {
      name: 'Circularity',
      unit_kind: 'ratio',
      value_dim: 1,
    };
    const imageMetrics = {
      1: {
        circularity: {
          unit: null,
          components: [{ count: 5, mean: 0.85, std: 0.05, min: 0.8, max: 0.9 }],
        },
      },
    };
    const image = { width: 1000, height: 1000, scale_x: 1, scale_y: 1, unit: 'px' };
    const scaleStatus = { display_physical: false, display_unit: 'px', consistent: true };

    render(
      <ImageStatCards
        imageMetrics={imageMetrics}
        metricKey="circularity"
        catalog={circularityCatalog}
        objectCounts={{ 1: { total: 5 } }}
        image={image}
        scaleStatus={scaleStatus}
      />
    );

    // Should display Mean circularity rather than Total circularity
    expect(screen.getByText('Mean circularity')).toBeInTheDocument();
    expect(screen.queryByText('Total circularity')).not.toBeInTheDocument();
    // Value should be the mean (0.85), not count * mean (4.25)
    expect(screen.getByText('0.85')).toBeInTheDocument();
    expect(screen.queryByText('4.25')).not.toBeInTheDocument();
    expect(screen.getByText('averaged over 5 objects')).toBeInTheDocument();
  });

  it('displays Total and sums additive metrics like perimeter', () => {
    const perimeterCatalog = {
      name: 'Perimeter',
      unit_kind: 'length',
      value_dim: 1,
    };
    const imageMetrics = {
      1: {
        perimeter: {
          unit: 'mm',
          components: [{ count: 5, mean: 10, std: 1, min: 9, max: 11 }],
        },
      },
    };
    const image = { width: 1000, height: 1000, scale_x: 0.5, scale_y: 0.5, unit: 'mm' };
    const scaleStatus = { display_physical: true, display_unit: 'mm', consistent: true };

    render(
      <ImageStatCards
        imageMetrics={imageMetrics}
        metricKey="perimeter"
        catalog={perimeterCatalog}
        objectCounts={{ 1: { total: 5 } }}
        image={image}
        scaleStatus={scaleStatus}
      />
    );

    expect(screen.getByText('Total perimeter')).toBeInTheDocument();
    // Value should be count * mean = 50
    expect(screen.getByText('50')).toBeInTheDocument();
    expect(screen.getByText('summed over 5 objects')).toBeInTheDocument();
  });

  it('displays "Scale unavailable" when scaleError is true even if scaleStatus has display_physical: true', () => {
    const image = { width: 1000, height: 1000, scale_x: null, scale_y: null, unit: null };
    const scaleStatus = { display_physical: true, display_unit: 'mm', consistent: true };

    render(
      <ImageStatCards
        imageMetrics={{}}
        metricKey="area"
        catalog={catalog}
        objectCounts={{ 1: { total: 5 } }}
        image={image}
        scaleStatus={scaleStatus}
        scaleError={true}
      />
    );

    expect(screen.getByText('Scale unavailable')).toBeInTheDocument();
    expect(screen.getByText('could not load calibration')).toBeInTheDocument();
    expect(screen.queryByText('Not calibrated')).not.toBeInTheDocument();
    expect(screen.queryByText('…')).not.toBeInTheDocument();
    expect(screen.queryByText('measurements are in real-world units')).not.toBeInTheDocument();
    expect(screen.queryByText('measurements are in pixels')).not.toBeInTheDocument();
  });

  it('displays "Scale unavailable" when scaleError is true and scaleStatus has display_physical: false', () => {
    const image = { width: 1000, height: 1000, scale_x: null, scale_y: null, unit: null };
    const scaleStatus = { display_physical: false, display_unit: 'px', consistent: true };

    render(
      <ImageStatCards
        imageMetrics={{}}
        metricKey="area"
        catalog={catalog}
        objectCounts={{ 1: { total: 5 } }}
        image={image}
        scaleStatus={scaleStatus}
        scaleError={true}
      />
    );

    expect(screen.getByText('Scale unavailable')).toBeInTheDocument();
    expect(screen.getByText('could not load calibration')).toBeInTheDocument();
    expect(screen.queryByText('Not calibrated')).not.toBeInTheDocument();
    expect(screen.queryByText('…')).not.toBeInTheDocument();
  });

  it('does not calculate the measured-object count from only one featured metric in label-scoped profiles', () => {
    const multiMetrics = {
      1: {
        area: { unit: 'mm²', components: [{ count: 5, mean: 10, std: 1, min: 9, max: 11 }] },
      },
      2: {
        circularity: { unit: null, components: [{ count: 3, mean: 0.8, std: 0.05, min: 0.7, max: 0.9 }] },
      },
    };
    const objectCounts = {
      1: { total: 5 },
      2: { total: 3 },
    };
    const image = { scale_x: 0.5, scale_y: 0.5, unit: 'mm' };
    const scaleStatus = { display_physical: true, display_unit: 'mm', consistent: true };

    render(
      <ImageStatCards
        imageMetrics={multiMetrics}
        metricKey="area"
        catalog={catalog}
        objectCounts={objectCounts}
        image={image}
        scaleStatus={scaleStatus}
      />
    );

    // Objects card should display 8 measured (5 Cells + 3 Nuclei), not "5 measured · 3 excluded"
    expect(screen.getByText('8')).toBeInTheDocument();
    expect(screen.getByText('8 measured')).toBeInTheDocument();
    expect(screen.queryByText(/excluded/)).not.toBeInTheDocument();

    // Featured metric (Area) card should correctly indicate it sums over the 5 objects having that metric
    expect(screen.getByText('summed over 5 objects')).toBeInTheDocument();
  });
});
