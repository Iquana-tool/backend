import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ImageMetricGrid from './ImageMetricGrid';

describe('ImageMetricGrid', () => {
  const catalogMap = {
    area: { name: 'Area', unit_kind: 'area', value_dim: 1, tier: 'geometry' },
    circularity: { name: 'Circularity', unit_kind: 'ratio', value_dim: 1, tier: 'geometry' },
    mean_color_rgb: { name: 'Mean Color', unit_kind: 'color', value_dim: 3, tier: 'appearance' },
  };

  it('renders dataset comparison when image and dataset have the same unit', () => {
    const imageMetrics = {
      1: { area: { unit: 'mm²', components: [{ count: 5, mean: 12.0, std: 1, min: 10, max: 14 }] } },
    };
    const datasetMetrics = {
      1: { area: { unit: 'mm²', components: [{ count: 20, mean: 10.0, std: 2, min: 5, max: 15 }] } },
    };

    render(
      <ImageMetricGrid
        imageMetrics={imageMetrics}
        datasetMetrics={datasetMetrics}
        catalogMap={catalogMap}
      />
    );

    expect(screen.getByText(/Measurements on this image/i)).toBeInTheDocument();
    expect(screen.getByText(/Dataset mean/i)).toBeInTheDocument();
    expect(screen.getByText('10 mm²')).toBeInTheDocument();
    expect(screen.getByText('+20.0 %')).toBeInTheDocument();
  });

  it('hides dataset comparison when units differ (e.g. image in mm² but dataset in px²)', () => {
    const imageMetrics = {
      1: { area: { unit: 'mm²', components: [{ count: 5, mean: 12.0, std: 1, min: 10, max: 14 }] } },
    };
    const datasetMetrics = {
      1: { area: { unit: 'px²', components: [{ count: 20, mean: 50000, std: 100, min: 40000, max: 60000 }] } },
    };

    render(
      <ImageMetricGrid
        imageMetrics={imageMetrics}
        datasetMetrics={datasetMetrics}
        catalogMap={catalogMap}
      />
    );

    expect(screen.getByText(/Measurements on this image/i)).toBeInTheDocument();
    expect(screen.queryByText(/Dataset mean/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/50000/)).not.toBeInTheDocument();
  });

  it('renders dataset comparison for dimensionless metrics where both units are empty/null', () => {
    const imageMetrics = {
      1: { circularity: { unit: null, components: [{ count: 5, mean: 0.8, std: 0.05, min: 0.7, max: 0.9 }] } },
    };
    const datasetMetrics = {
      1: { circularity: { unit: null, components: [{ count: 20, mean: 0.7, std: 0.1, min: 0.5, max: 0.95 }] } },
    };

    render(
      <ImageMetricGrid
        imageMetrics={imageMetrics}
        datasetMetrics={datasetMetrics}
        catalogMap={catalogMap}
      />
    );

    expect(screen.getByText(/Dataset mean/i)).toBeInTheDocument();
    expect(screen.getByText('0.7')).toBeInTheDocument();
    expect(screen.getByText('+14.3 %')).toBeInTheDocument();
  });

  it('does not render dataset comparison when baseline is missing', () => {
    const imageMetrics = {
      1: { area: { unit: 'mm²', components: [{ count: 5, mean: 12.0, std: 1, min: 10, max: 14 }] } },
    };

    render(
      <ImageMetricGrid
        imageMetrics={imageMetrics}
        datasetMetrics={{}}
        catalogMap={catalogMap}
      />
    );

    expect(screen.queryByText(/Dataset mean/i)).not.toBeInTheDocument();
  });
});
