import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import ImageQuantificationPage from './ImageQuantificationPage';
import * as api from '../api';
import * as masksApi from '../api/masks';
import * as scaleApi from '../api/scale';
import * as quantApi from '../api/quantifications';

vi.mock('../components/datasets/gallery/DatasetManagementLayout', () => ({
  default: ({ children }) => <div data-testid="layout">{children}</div>,
}));

vi.mock('../components/ui/ImageFilmstrip', () => ({
  default: ({ images, selectedId, onSelect }) => (
    <div data-testid="filmstrip">
      {images.map((img) => (
        <button
          key={img.id}
          data-testid={`filmstrip-item-${img.id}`}
          onClick={() => onSelect(img)}
        >
          {img.name}
        </button>
      ))}
    </div>
  ),
}));

vi.mock('../components/viewer/AnnotationViewerCanvas', () => ({
  default: ({ contextIds, contours }) => (
    <div
      data-testid="viewer-canvas"
      data-context-ids={Array.from(contextIds || []).sort().join(',')}
      data-contour-count={contours?.length || 0}
    />
  ),
}));

const mockCan = vi.fn(() => true);
vi.mock('../hooks/usePermissions', () => ({
  usePermissions: () => ({ can: mockCan }),
}));

vi.mock('../stores/selectors/annotationSelectors', () => ({
  useWorkspaceTheme: () => 'dark',
}));

vi.mock('../api', () => ({
  fetchImages: vi.fn(),
  getImageById: vi.fn(),
}));

vi.mock('../api/masks', () => ({
  getContoursOfMask: vi.fn(),
}));

vi.mock('../api/scale', () => ({
  getPixelScale: vi.fn(),
}));

vi.mock('../api/quantifications', () => ({
  buildQuantificationDownloadUrl: vi.fn(),
  fetchQuantificationRows: vi.fn(),
  getMetricsCatalog: vi.fn(),
  getQuantificationProfiles: vi.fn(),
  getQuantificationSummary: vi.fn(),
}));

const LocationProbe = () => {
  const location = useLocation();
  return (
    <div data-testid="location-probe">
      <span data-testid="pathname">{location.pathname}</span>
      <span data-testid="search">{location.search}</span>
    </div>
  );
};

describe('ImageQuantificationPage study parameters preservation', () => {
  const images = [
    { id: 101, file_name: 'img_101.png', mask_id: 1 },
    { id: 102, file_name: 'img_102.png', mask_id: 2 },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    mockCan.mockReturnValue(true);
    api.fetchImages.mockResolvedValue({
      image_data: images,
    });
    api.getImageById.mockResolvedValue({
      101: 'fake-base-64-101',
      102: 'fake-base-64-102',
    });
    masksApi.getContoursOfMask.mockResolvedValue({ contours: [] });
    scaleApi.getPixelScale.mockResolvedValue({ scale_x: 1, scale_y: 1, unit: 'px' });
    quantApi.getMetricsCatalog.mockResolvedValue({
      metrics: [
        { key: 'area', name: 'Area', unit_kind: 'area', value_dim: 1, tier: 'geometry' },
      ],
    });
    quantApi.getQuantificationProfiles.mockResolvedValue({
      profiles: [{ id: 1, name: 'Default', is_default: true }],
    });
    quantApi.getQuantificationSummary.mockResolvedValue({
      scale_status: { display_physical: false, display_unit: 'px', consistent: true },
      metrics: {
        1: { area: { unit: 'px²', components: [{ count: 2, mean: 100, std: 5, min: 95, max: 105 }] } },
      },
      labels: { id_to_label_object: { 1: { name: 'Cell' } } },
    });
    quantApi.fetchQuantificationRows.mockResolvedValue({
      rows: [],
      message: null,
    });
  });

  it('preserves query parameters on the initial image redirect', async () => {
    render(
      <MemoryRouter
        initialEntries={['/dataset/42/quantifications/image?segmentations=off&table=off']}
      >
        <LocationProbe />
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image"
            element={<ImageQuantificationPage />}
          />
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByTestId('pathname').textContent).toBe(
        '/dataset/42/quantifications/image/101'
      );
    });

    expect(screen.getByTestId('search').textContent).toBe(
      '?segmentations=off&table=off'
    );
  });

  it('preserves query parameters when navigating using next / previous buttons', async () => {
    render(
      <MemoryRouter
        initialEntries={['/dataset/42/quantifications/image/101?segmentations=off&table=off']}
      >
        <LocationProbe />
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
    });

    const nextButton = screen.getByRole('button', { name: /Next image/i });
    fireEvent.click(nextButton);

    await waitFor(() => {
      expect(screen.getByTestId('pathname').textContent).toBe(
        '/dataset/42/quantifications/image/102'
      );
    });
    expect(screen.getByTestId('search').textContent).toBe('?segmentations=off&table=off');

    const prevButton = screen.getByRole('button', { name: /Previous image/i });
    fireEvent.click(prevButton);

    await waitFor(() => {
      expect(screen.getByTestId('pathname').textContent).toBe(
        '/dataset/42/quantifications/image/101'
      );
    });
    expect(screen.getByTestId('search').textContent).toBe('?segmentations=off&table=off');
  });

  it('preserves query parameters when navigating through the filmstrip', async () => {
    render(
      <MemoryRouter
        initialEntries={['/dataset/42/quantifications/image/101?segmentations=off&table=off']}
      >
        <LocationProbe />
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
    });

    const filmstripItem = screen.getByTestId('filmstrip-item-102');
    fireEvent.click(filmstripItem);

    await waitFor(() => {
      expect(screen.getByTestId('pathname').textContent).toBe(
        '/dataset/42/quantifications/image/102'
      );
    });
    expect(screen.getByTestId('search').textContent).toBe('?segmentations=off&table=off');
  });

  it('clears old measurements and shows loading spinner when navigating to a new image', async () => {
    let resolveImage102Metrics;
    quantApi.getQuantificationSummary.mockImplementation((datasetId, options) => {
      if (options.imageId === 102) {
        return new Promise((resolve) => {
          resolveImage102Metrics = resolve;
        });
      }
      return Promise.resolve({
        scale_status: { display_physical: false, display_unit: 'px', consistent: true },
        metrics: {
          1: { area: { unit: 'px²', components: [{ count: 2, mean: 100, std: 5, min: 95, max: 105 }] } },
        },
        labels: { id_to_label_object: { 1: { name: 'Cell' } } },
      });
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    // Initial image loads
    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
      expect(screen.getByText(/Total area/i)).toBeInTheDocument();
    });

    // Navigate to next image
    const nextButton = screen.getByRole('button', { name: /Next image/i });
    fireEvent.click(nextButton);

    // Header updates to img_102
    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_102.png' })).toBeInTheDocument();
    });

    // Old measurements should NOT be visible; spinner should be visible
    expect(screen.queryByText(/Total area/i)).not.toBeInTheDocument();
    expect(screen.getByText('Loading measurements…')).toBeInTheDocument();

    // Now resolve image 102 measurements
    resolveImage102Metrics({
      scale_status: { display_physical: false, display_unit: 'px', consistent: true },
      metrics: {
        1: { area: { unit: 'px²', components: [{ count: 3, mean: 250, std: 10, min: 240, max: 260 }] } },
      },
      labels: { id_to_label_object: { 1: { name: 'Cell' } } },
    });

    await waitFor(() => {
      expect(screen.getByText(/Total area/i)).toBeInTheDocument();
    });
    expect(screen.queryByText('Loading measurements…')).not.toBeInTheDocument();
  });

  it('clears pixelScale when navigating from calibrated Image A to uncalibrated Image B', async () => {
    let resolveImage102Scale;
    scaleApi.getPixelScale.mockImplementation((imageId) => {
      if (imageId === 101) {
        return Promise.resolve({ scale_x: 0.05, scale_y: 0.05, unit: 'mm' });
      }
      if (imageId === 102) {
        return new Promise((resolve) => {
          resolveImage102Scale = resolve;
        });
      }
      return Promise.resolve({ scale_x: 1, scale_y: 1, unit: 'px' });
    });

    quantApi.getQuantificationSummary.mockImplementation((datasetId, options) => {
      if (options.imageId === 101) {
        return Promise.resolve({
          scale_status: { display_physical: true, display_unit: 'mm', consistent: true },
          metrics: {
            1: { area: { unit: 'mm²', components: [{ count: 2, mean: 10, std: 1, min: 9, max: 11 }] } },
          },
          labels: { id_to_label_object: { 1: { name: 'Cell' } } },
        });
      }
      return Promise.resolve({
        scale_status: { display_physical: false, display_unit: 'px', consistent: true },
        metrics: {
          1: { area: { unit: 'px²', components: [{ count: 2, mean: 100, std: 5, min: 95, max: 105 }] } },
        },
        labels: { id_to_label_object: { 1: { name: 'Cell' } } },
      });
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    // Initial calibrated image loads
    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
      expect(screen.getByText('1 px = 0.05')).toBeInTheDocument();
      expect(screen.getByText('measurements are in real-world units')).toBeInTheDocument();
    });

    // Navigate to Image B (102)
    const nextButton = screen.getByRole('button', { name: /Next image/i });
    fireEvent.click(nextButton);

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_102.png' })).toBeInTheDocument();
    });

    // Wait for Image B's metrics to render while its scale request is still pending.
    // This proves the fix: pixelScale was cleared to null, so Image B renders "Not calibrated"
    // instead of temporarily inheriting Image A's 0.05 mm scale.
    await waitFor(() => {
      expect(screen.getByText(/Total area/i)).toBeInTheDocument();
      expect(screen.getByText('Not calibrated')).toBeInTheDocument();
    });
    expect(screen.getByText('measurements are in pixels')).toBeInTheDocument();
    expect(screen.queryByText('1 px = 0.05')).not.toBeInTheDocument();
    expect(screen.queryByText('measurements are in real-world units')).not.toBeInTheDocument();

    // Now resolve Image 102 scale request as uncalibrated
    resolveImage102Scale({ scale_x: 1, scale_y: 1, unit: 'px' });

    await waitFor(() => {
      expect(screen.getByText(/This image has no scale/i)).toBeInTheDocument();
    });
    expect(screen.getByText('Not calibrated')).toBeInTheDocument();
    expect(screen.getByText('measurements are in pixels')).toBeInTheDocument();
  });

  it('avoids displaying a numeric scale when navigating from calibrated Image A to calibrated Image B while Image B scale request is still pending', async () => {
    let resolveImage102Scale;
    scaleApi.getPixelScale.mockImplementation((imageId) => {
      if (imageId === 101) {
        return Promise.resolve({ scale_x: 0.05, scale_y: 0.05, unit: 'mm' });
      }
      if (imageId === 102) {
        return new Promise((resolve) => {
          resolveImage102Scale = resolve;
        });
      }
      return Promise.resolve({ scale_x: 1, scale_y: 1, unit: 'px' });
    });

    quantApi.getQuantificationSummary.mockImplementation((datasetId, options) => {
      if (options.imageId === 101) {
        return Promise.resolve({
          scale_status: { display_physical: true, display_unit: 'mm', consistent: true },
          metrics: {
            1: { area: { unit: 'mm²', components: [{ count: 2, mean: 10, std: 1, min: 9, max: 11 }] } },
          },
          labels: { id_to_label_object: { 1: { name: 'Cell' } } },
        });
      }
      // Image B (102) summary arrives quickly with calibrated scale_status
      return Promise.resolve({
        scale_status: { display_physical: true, display_unit: 'mm', consistent: true },
        metrics: {
          1: { area: { unit: 'mm²', components: [{ count: 3, mean: 20, std: 2, min: 18, max: 22 }] } },
        },
        labels: { id_to_label_object: { 1: { name: 'Cell' } } },
      });
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    // Initial calibrated image loads
    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
      expect(screen.getByText('1 px = 0.05')).toBeInTheDocument();
    });

    // Navigate to Image B (102)
    const nextButton = screen.getByRole('button', { name: /Next image/i });
    fireEvent.click(nextButton);

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_102.png' })).toBeInTheDocument();
    });

    // Image B's metrics render while Image B's scale request is still pending.
    await waitFor(() => {
      expect(screen.getByText(/Total area/i)).toBeInTheDocument();
      expect(screen.getByText('measurements are in real-world units')).toBeInTheDocument();
    });

    // Must NOT display fabricated numeric scale like "1 px = 1 mm" or Image A's "1 px = 0.05"
    expect(screen.queryByText(/1 px =/)).not.toBeInTheDocument();

    // Now resolve Image 102 scale request with its true physical scale (e.g. 0.02 mm)
    resolveImage102Scale({ scale_x: 0.02, scale_y: 0.02, unit: 'mm' });

    await waitFor(() => {
      expect(screen.getByText('1 px = 0.02')).toBeInTheDocument();
      expect(screen.getByText('mm')).toBeInTheDocument();
    });
    expect(screen.getByText('measurements are in real-world units')).toBeInTheDocument();
  });

  it('marks all contours as context when filters produce zero measured rows', async () => {
    masksApi.getContoursOfMask.mockResolvedValue({
      contours: [
        { id: 10, label_id: 1 },
        { id: 20, label_id: 1 },
      ],
    });
    quantApi.getQuantificationSummary.mockResolvedValue({
      scale_status: { display_physical: false, display_unit: 'px', consistent: true },
      metrics: {},
      labels: { id_to_label_object: { 1: { name: 'Cell' } } },
    });
    quantApi.fetchQuantificationRows.mockResolvedValue({
      rows: [],
      message: 'all unreviewed excluded',
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
      expect(screen.getByText(/Nothing on this image is measured/i)).toBeInTheDocument();
    });

    // Both contours must be in contextIds so they render as context rather than normal subjects
    const canvas = screen.getByTestId('viewer-canvas');
    expect(canvas.getAttribute('data-context-ids')).toBe('10,20');
  });

  it('does not let a successful metrics request clear an image-loading error', async () => {
    api.getImageById.mockRejectedValue(new Error('Image load failed'));
    quantApi.getQuantificationSummary.mockResolvedValue({
      scale_status: { display_physical: true, display_unit: 'mm', consistent: true },
      metrics: {
        1: { area: { unit: 'mm²', components: [{ count: 2, mean: 10, std: 1, min: 9, max: 11 }] } },
      },
      labels: { id_to_label_object: { 1: { name: 'Cell' } } },
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    // Image error must remain displayed in the viewer area even after metrics succeed
    await waitFor(() => {
      expect(screen.getByText('Image load failed')).toBeInTheDocument();
      expect(screen.getByText(/Total area/i)).toBeInTheDocument();
    });

    // Viewer canvas is not rendered, so user is not left with an unexplained blank viewer
    expect(screen.queryByTestId('viewer-canvas')).not.toBeInTheDocument();
  });

  it('does not interpret a failed scale request as missing calibration', async () => {
    scaleApi.getPixelScale.mockRejectedValue(new Error('Network timeout fetching scale'));
    quantApi.getQuantificationSummary.mockResolvedValue({
      scale_status: { display_physical: false, display_unit: 'px', consistent: true },
      metrics: {
        1: { area: { unit: 'px²', components: [{ count: 2, mean: 10, std: 1, min: 9, max: 11 }] } },
      },
      labels: { id_to_label_object: { 1: { name: 'Cell' } } },
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Scale unavailable')).toBeInTheDocument();
    });

    expect(screen.getByText('could not load calibration')).toBeInTheDocument();
    // Must NOT display "Not calibrated" or "measurements are in pixels"
    expect(screen.queryByText('Not calibrated')).not.toBeInTheDocument();
    expect(screen.queryByText('measurements are in pixels')).not.toBeInTheDocument();
    // Must NOT show the missing calibration warning banner
    expect(screen.queryByText(/This image has no scale/i)).not.toBeInTheDocument();
    // Must show the scale unavailable warning banner
    expect(screen.getByText(/Could not load calibration for this image/i)).toBeInTheDocument();
  });

  it('preserves context styling and shows info banner for users without export permission', async () => {
    mockCan.mockReturnValue(false);

    masksApi.getContoursOfMask.mockResolvedValue({
      contours: [
        { id: 10, label_id: 1, reviewed_by: ['alice'] }, // reviewed Cell (measured)
        { id: 20, label_id: 1, reviewed_by: [] },        // unreviewed Cell (excluded by unreviewed filter)
        { id: 30, label_id: 2, reviewed_by: ['alice'] }, // reviewed Nucleus (excluded by profile)
      ],
    });
    quantApi.getQuantificationSummary.mockResolvedValue({
      scale_status: { display_physical: true, display_unit: 'mm', consistent: true },
      metrics: {
        1: { area: { unit: 'mm²', components: [{ count: 1, mean: 10, std: 0, min: 10, max: 10 }] } },
      },
      labels: { id_to_label_object: { 1: { name: 'Cell' }, 2: { name: 'Nucleus' } } },
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
      expect(screen.getByText(/Total area/i)).toBeInTheDocument();
    });

    // Rows must not be requested without export permission
    expect(quantApi.fetchQuantificationRows).not.toHaveBeenCalled();

    // Export permission message is shown
    expect(
      screen.getByText(/The per-object table needs the quantification export permission/i)
    ).toBeInTheDocument();

    // Contours 20 (unreviewed) and 30 (excluded label) must be context, contour 10 is measured
    const canvas = screen.getByTestId('viewer-canvas');
    expect(canvas.getAttribute('data-context-ids')).toBe('20,30');
  });

  it('keeps summary cards visible and shows row-specific error when row request fails', async () => {
    quantApi.getQuantificationSummary.mockResolvedValue({
      scale_status: { display_physical: true, display_unit: 'mm', consistent: true },
      metrics: {
        1: { area: { unit: 'mm²', components: [{ count: 2, mean: 10, std: 1, min: 9, max: 11 }] } },
      },
      labels: { id_to_label_object: { 1: { name: 'Cell' } } },
    });
    quantApi.fetchQuantificationRows.mockRejectedValue(
      new Error('Export service unavailable')
    );

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
      // Valid summary cards remain visible
      expect(screen.getByText(/Total area/i)).toBeInTheDocument();
    });

    // Does not show a page-level metrics error
    expect(screen.queryByText(/Could not load the measurements/i)).not.toBeInTheDocument();

    // Shows row-specific error banner
    expect(screen.getByText(/Export service unavailable/i)).toBeInTheDocument();
  });

  it('does not download table data and falls back to label/review context styling when table is off', async () => {
    masksApi.getContoursOfMask.mockResolvedValue({
      contours: [
        { id: 10, label_id: 1, reviewed_by: ['alice'] }, // reviewed Cell (measured)
        { id: 20, label_id: 1, reviewed_by: [] },        // unreviewed Cell (excluded)
        { id: 30, label_id: 2, reviewed_by: ['alice'] }, // reviewed Nucleus (excluded by profile)
      ],
    });
    quantApi.getQuantificationSummary.mockResolvedValue({
      scale_status: { display_physical: true, display_unit: 'mm', consistent: true },
      metrics: {
        1: { area: { unit: 'mm²', components: [{ count: 1, mean: 10, std: 0, min: 10, max: 10 }] } },
      },
      labels: { id_to_label_object: { 1: { name: 'Cell' }, 2: { name: 'Nucleus' } } },
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101?table=off']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
      expect(screen.getByText(/Total area/i)).toBeInTheDocument();
    });

    // fetchQuantificationRows must NOT be called with ?table=off (respects study condition)
    expect(quantApi.fetchQuantificationRows).not.toHaveBeenCalled();

    // Table is hidden
    expect(screen.queryByText(/Individual objects/i)).not.toBeInTheDocument();

    // Contours 20 (unreviewed) and 30 (unmeasured label) are context, contour 10 is measured
    const canvas = screen.getByTestId('viewer-canvas');
    expect(canvas.getAttribute('data-context-ids')).toBe('20,30');
  });

  it('does not fetch quantification rows when both table and segmentations are hidden', async () => {
    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101?table=off&segmentations=off']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
    });

    expect(quantApi.fetchQuantificationRows).not.toHaveBeenCalled();
    expect(masksApi.getContoursOfMask).not.toHaveBeenCalled();
  });

  it('determines measured contours from non-null metric values rather than row presence', async () => {
    masksApi.getContoursOfMask.mockResolvedValue({
      contours: [
        { id: 10, label_id: 1, reviewed_by: ['alice'] }, // measured
        { id: 30, label_id: 1, reviewed_by: ['alice'] }, // contextual only-child (row present but metric null)
      ],
    });
    quantApi.getQuantificationSummary.mockResolvedValue({
      scale_status: { display_physical: true, display_unit: 'mm', consistent: true },
      metrics: {
        1: {
          nn_distance: {
            unit: 'mm',
            components: [{ count: 1, mean: 15, std: 0, min: 15, max: 15 }],
          },
        },
      },
      labels: { id_to_label_object: { 1: { name: 'Cell' } } },
    });
    // Backend export returns rows for all filtered contours; omitted contextual contours have null metric values
    quantApi.fetchQuantificationRows.mockResolvedValue({
      rows: [
        { contour_id: 10, label_id: 1, label: 'Cell', file_name: 'img_101.png', nn_distance: 15.0 },
        { contour_id: 30, label_id: 1, label: 'Cell', file_name: 'img_101.png', nn_distance: null },
      ],
      message: null,
    });

    render(
      <MemoryRouter initialEntries={['/dataset/42/quantifications/image/101']}>
        <Routes>
          <Route
            path="/dataset/:datasetId/quantifications/image/:imageId"
            element={<ImageQuantificationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'img_101.png' })).toBeInTheDocument();
      expect(screen.getByText(/Individual objects/i)).toBeInTheDocument();
    });

    // Contour 30 has null metric values, so it must be styled as context despite having a row
    // Contour 10 has a non-null metric value, so it is a measured subject
    const canvas = screen.getByTestId('viewer-canvas');
    expect(canvas.getAttribute('data-context-ids')).toBe('30');
  });
});
