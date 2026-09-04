import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import LabelComparisonBars from './LabelComparisonBars';

describe('LabelComparisonBars', () => {
  const catalog = {
    name: 'Area',
    unit_kind: 'area',
  };

  const labelIdToName = {
    1: 'Cell',
    2: 'Nucleus',
  };

  it('renders dataset comparison ticks when units match', () => {
    const imageMetrics = {
      1: { area: { unit: 'mm²', components: [{ count: 5, mean: 12.0, std: 1, min: 10, max: 14 }] } },
      2: { area: { unit: 'mm²', components: [{ count: 3, mean: 6.0, std: 0.5, min: 5, max: 7 }] } },
    };
    const datasetMetrics = {
      1: { area: { unit: 'mm²', components: [{ count: 20, mean: 10.0, std: 2, min: 5, max: 15 }] } },
      2: { area: { unit: 'mm²', components: [{ count: 15, mean: 5.0, std: 1, min: 3, max: 8 }] } },
    };

    const { container } = render(
      <LabelComparisonBars
        imageMetrics={imageMetrics}
        datasetMetrics={datasetMetrics}
        metricKey="area"
        catalog={catalog}
        labelIdToName={labelIdToName}
      />
    );

    expect(screen.getByText(/Mean area by label · this image vs dataset/i)).toBeInTheDocument();
    expect(screen.getByText(/Bar = this image · tick = dataset mean for the same label/i)).toBeInTheDocument();
    expect(container.querySelector('[title="Dataset mean: 10 mm²"]')).toBeInTheDocument();
    expect(container.querySelector('[title="Dataset mean: 5 mm²"]')).toBeInTheDocument();
  });

  it('hides ticks and suppresses dataset comparison when units differ (e.g. mm² vs px²)', () => {
    const imageMetrics = {
      1: { area: { unit: 'mm²', components: [{ count: 5, mean: 12.0, std: 1, min: 10, max: 14 }] } },
    };
    // Mixed-scale dataset fell back to px²
    const datasetMetrics = {
      1: { area: { unit: 'px²', components: [{ count: 20, mean: 50000.0, std: 100, min: 40000, max: 60000 }] } },
    };

    const { container } = render(
      <LabelComparisonBars
        imageMetrics={imageMetrics}
        datasetMetrics={datasetMetrics}
        metricKey="area"
        catalog={catalog}
        labelIdToName={labelIdToName}
      />
    );

    // Title should NOT claim vs dataset
    expect(screen.getByText(/^Mean area by label$/i)).toBeInTheDocument();
    expect(screen.queryByText(/this image vs dataset/i)).not.toBeInTheDocument();
    // Footer should only describe the bar
    expect(screen.getByText('Bar = this image')).toBeInTheDocument();
    // No tick with dataset mean
    expect(container.querySelector('[title*="Dataset mean"]')).toBeNull();

    // The bar width should be 100% because scaleMax is 12 (not distorted by 50000)
    const bar = container.querySelector('.bg-well > div');
    expect(bar).toHaveStyle({ width: '100%' });
  });

  it('renders image bars cleanly when dataset baseline is absent', () => {
    const imageMetrics = {
      1: { area: { unit: 'mm²', components: [{ count: 5, mean: 12.0, std: 1, min: 10, max: 14 }] } },
    };

    const { container } = render(
      <LabelComparisonBars
        imageMetrics={imageMetrics}
        datasetMetrics={{}}
        metricKey="area"
        catalog={catalog}
        labelIdToName={labelIdToName}
      />
    );

    expect(screen.getByText(/^Mean area by label$/i)).toBeInTheDocument();
    expect(screen.getByText('Bar = this image')).toBeInTheDocument();
    expect(container.querySelector('[title*="Dataset mean"]')).toBeNull();
  });
});
