import React, { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Download,
  Info,
  LayoutGrid,
  SquarePen,
  Table2,
} from "lucide-react";
import * as api from "../api";
import { getContoursOfMask } from "../api/masks";
import { getPixelScale } from "../api/scale";
import {
  buildQuantificationDownloadUrl,
  fetchQuantificationRows,
  getMetricsCatalog,
  getQuantificationProfiles,
  getQuantificationSummary,
} from "../api/quantifications";
import { getAuthHeaders } from "../api/util";
import DatasetManagementLayout from "../components/datasets/gallery/DatasetManagementLayout";
import AnnotationViewerCanvas from "../components/viewer/AnnotationViewerCanvas";
import ProfileSelector from "../components/quantification/ProfileSelector";
import ImageFilmstrip from "../components/ui/ImageFilmstrip";
import ImageMetricGrid from "../components/quantification/perImage/ImageMetricGrid";
import ImageStatCards from "../components/quantification/perImage/ImageStatCards";
import LabelComparisonBars from "../components/quantification/perImage/LabelComparisonBars";
import ObjectsOnImageTable from "../components/quantification/perImage/ObjectsOnImageTable";
import StudyViewControl from "../components/quantification/perImage/StudyViewControl";
import { usePermissions } from "../hooks/usePermissions";
import { useStudyView } from "../hooks/useStudyView";
import { normalizeImage } from "../hooks/useDatasetGalleryData";
import { Permission } from "../utils/permissions";
import { getLabelColor } from "../utils/labelColors";
import { perImageViewerConfig } from "../utils/perspectiveQuantification";
import { useWorkspaceTheme } from "../stores/selectors/annotationSelectors";
import {
  buildMetricCatalogMap,
  createLabelIdToNameMap,
} from "../utils/quantificationUtils";
import { pickFeaturedMetric } from "../utils/perImageQuantification";

// Same lazy boundary as the dataset page: the Perspective engine is ~4.5 MB of WebAssembly,
// and the pivot here is opt-in, so it must not be in the bundle everyone else downloads.
const QuantificationExplorer = React.lazy(() =>
  import("../components/quantification/QuantificationExplorer")
);

/** Row keys that identify or structure the object rather than measure it. */
const NON_METRIC_ROW_KEYS = new Set([
  "file_name",
  "label",
  "label_id",
  "contour_id",
  "parent_id",
  "parent_label",
]);

const readableError = (err, fallback) =>
  (err?.message || "").replace(/^API Error:\s*/i, "") || fallback;

const Centered = ({ children }) => (
  <div className="h-full flex items-center justify-center">
    <div className="text-center max-w-lg px-6">{children}</div>
  </div>
);

const Spinner = ({ label, hint }) => (
  <Centered>
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-acLn mx-auto mb-4" />
    <p className="text-t2">{label}</p>
    {hint && <p className="text-sm text-t3 mt-1">{hint}</p>}
  </Centered>
);

const ViewButton = ({ icon: Icon, label, active, onClick }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
      active ? "bg-p1 text-t1 shadow-sm" : "text-t2 hover:text-t1"
    }`}
  >
    <Icon className="w-4 h-4" />
    {label}
  </button>
);

/**
 * Quantifications for ONE image (issue #15).
 *
 * A dedicated page rather than a tab in the annotation canvas: the canvas is where objects
 * are made, and its whole layout is built around that. Reading measurements is a different
 * job — it wants the numbers, the table and the image side by side, and it wants to be
 * reachable by someone who is not annotating at all.
 *
 * It is the dataset quantification page scoped down, and deliberately reuses its parts
 * (the profile selector, the metric cards, the Perspective explorer, the same endpoints)
 * so the two can never disagree about what a metric means. Two summaries are loaded on
 * every visit: this image's, and the dataset's as a baseline — an image's numbers are only
 * interesting relative to the population they came from.
 *
 * The page doubles as the instrument for the trust study, via `useStudyView`: the
 * segmentation overlay and the per-object table are each removable, so what a participant
 * can see of the evidence behind an aggregate is a property of the URL they were given.
 */
const ImageQuantificationPage = () => {
  const { datasetId, imageId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { can } = usePermissions(datasetId);
  const theme = useWorkspaceTheme();
  const study = useStudyView();

  // The rows come from the export endpoint, which the server guards with
  // `export.quantification` — the same split the dataset page makes between its Overview
  // and its Objects tab. Without it the page still shows the aggregated cards.
  const canExport = can(Permission.EXPORT_QUANTIFICATION);
  // Two independent reasons the per-object evidence may be absent: the viewer is not
  // entitled to it, or the study condition removes it. They read very differently to the
  // person looking at the page, so they are kept apart and explained separately.
  const showTable = canExport && study.table;
  const shouldFetchRows = canExport && study.table;

  const [images, setImages] = useState([]);
  const [catalog, setCatalog] = useState([]);
  const [profiles, setProfiles] = useState([]);
  const [activeProfileId, setActiveProfileId] = useState(null);
  const [includeInProgress, setIncludeInProgress] = useState(false);
  const [includeUnreviewed, setIncludeUnreviewed] = useState(false);
  const [objectView, setObjectView] = useState("table");

  const [imageSummary, setImageSummary] = useState(null);
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [rows, setRows] = useState([]);
  const [rowsMessage, setRowsMessage] = useState(null);
  const [rowsError, setRowsError] = useState(null);
  const [rowsLoadedEmpty, setRowsLoadedEmpty] = useState(false);

  const [imageSrc, setImageSrc] = useState(null);
  const [contours, setContours] = useState([]);
  const [pixelScale, setPixelScale] = useState(null);
  const [naturalSize, setNaturalSize] = useState(null);
  const [selectedContourId, setSelectedContourId] = useState(null);
  const [zoomTarget, setZoomTarget] = useState(null);

  const [loadingMeta, setLoadingMeta] = useState(true);
  const [loadingImage, setLoadingImage] = useState(false);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [metaError, setMetaError] = useState(null);
  const [imageError, setImageError] = useState(null);
  const [metricsError, setMetricsError] = useState(null);
  const [scaleError, setScaleError] = useState(null);

  const numericImageId = imageId ? Number(imageId) : null;
  const currentIndex = images.findIndex((item) => item.id === numericImageId);
  const currentImage = currentIndex >= 0 ? images[currentIndex] : null;

  // -- Dataset-level data, loaded once ---------------------------------------

  useEffect(() => {
    if (!datasetId) return undefined;
    let cancelled = false;

    const load = async () => {
      setLoadingMeta(true);
      setMetaError(null);
      try {
        const [imageResponse, catalogResponse, profileResponse] = await Promise.all([
          api.fetchImages(datasetId),
          getMetricsCatalog(),
          getQuantificationProfiles(Number(datasetId)),
        ]);
        if (cancelled) return;

        // Normalized to the gallery/workspace image shape, which is what the shared
        // filmstrip reads its per-phase status marks from.
        setImages((imageResponse.image_data || []).map(normalizeImage));
        setCatalog(catalogResponse.metrics || []);
        const loadedProfiles = profileResponse.profiles || [];
        setProfiles(loadedProfiles);
        const preferred = loadedProfiles.find((p) => p.is_default) || loadedProfiles[0];
        setActiveProfileId(preferred ? preferred.id : null);
        setMetaError(null);
      } catch (err) {
        if (!cancelled) setMetaError(readableError(err, "Could not load this dataset."));
      } finally {
        if (!cancelled) setLoadingMeta(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const navigateToImage = useCallback(
    (targetImageId, options = {}) => {
      navigate(
        `/dataset/${datasetId}/quantifications/image/${targetImageId}${location.search}`,
        options
      );
    },
    [datasetId, location.search, navigate]
  );

  // The route may carry no image (the entry point from the dataset page does not know one
  // yet). Land on the first image rather than showing an empty shell, and replace rather
  // than push so Back leaves the page instead of bouncing between the two URLs.
  useEffect(() => {
    if (numericImageId || images.length === 0) return;
    navigateToImage(images[0].id, { replace: true });
  }, [numericImageId, images, navigateToImage]);

  // -- The image itself ------------------------------------------------------

  useEffect(() => {
    if (!currentImage) return undefined;
    let cancelled = false;

    const load = async () => {
      setLoadingImage(true);
      setImageError(null);
      setScaleError(null);
      setSelectedContourId(null);
      setZoomTarget(null);
      setNaturalSize(null);
      setContours([]);
      setPixelScale(null);
      try {
        const [imageData, contourResponse, scaleResult] = await Promise.all([
          api.getImageById(currentImage.id, false),
          // Not fetched at all when the study condition hides the overlay: an outline that
          // is merely not drawn is still in the response, and the point of the condition is
          // that the participant has no access to it.
          currentImage.mask_id && study.segmentations
            ? getContoursOfMask(currentImage.mask_id, true)
            : Promise.resolve({ contours: [] }),
          // The image list carries no scale, and every measurement on this page is either
          // in that unit or in pixels because it is missing.
          getPixelScale(currentImage.id)
            .then((scale) => ({ data: scale, error: null }))
            .catch((err) => ({
              data: null,
              error: readableError(err, "Could not load image calibration."),
            })),
        ]);
        if (cancelled) return;

        // The endpoint keys the base64 payload by image id.
        const base64 =
          imageData[currentImage.id] ??
          imageData[String(currentImage.id)] ??
          Object.entries(imageData).find(
            ([key]) => key !== "success" && key !== "message"
          )?.[1];

        setImageSrc(base64 ? `data:image/png;base64,${base64}` : null);
        setContours(contourResponse.contours || []);
        setPixelScale(scaleResult.data);
        setScaleError(scaleResult.error);
      } catch (err) {
        if (!cancelled) setImageError(readableError(err, "Could not load this image."));
      } finally {
        if (!cancelled) setLoadingImage(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [currentImage, study.segmentations]);

  // -- Measurements ----------------------------------------------------------

  // Both summaries in one effect and one Promise.all: they are rendered together (a card
  // and its dataset comparison are meaningless apart), so committing them separately would
  // show a comparison against a baseline that is still the previous profile's.
  useEffect(() => {
    if (!datasetId || !numericImageId || activeProfileId === null) return undefined;
    let cancelled = false;

    const options = {
      profileId: activeProfileId,
      excludeNotFullyAnnotated: !includeInProgress,
      excludeUnreviewed: !includeUnreviewed,
    };

    const load = async () => {
      setLoadingMetrics(true);
      setMetricsError(null);
      setRowsError(null);
      setImageSummary(null);
      setDatasetSummary(null);
      setRows([]);
      setRowsMessage(null);
      setRowsLoadedEmpty(false);
      try {
        const [scoped, whole, rowResult] = await Promise.all([
          getQuantificationSummary(Number(datasetId), { ...options, imageId: numericImageId }),
          getQuantificationSummary(Number(datasetId), options),
          // A hidden table is an unmade request: avoid downloading hidden table data when
          // study.table is off or user lacks export permission.
          shouldFetchRows
            ? fetchQuantificationRows(Number(datasetId), {
                ...options,
                imageId: numericImageId,
              })
                .then((res) => ({ ...res, error: null }))
                .catch((err) => ({
                  rows: [],
                  message: null,
                  error: readableError(err, "Could not load per-object measurements."),
                }))
            : Promise.resolve({ rows: [], message: null, error: null }),
        ]);
        if (cancelled) return;
        setImageSummary(scoped);
        setDatasetSummary(whole);
        setRows(rowResult.rows || []);
        setRowsMessage(rowResult.message || null);
        setRowsError(rowResult.error || null);
        setRowsLoadedEmpty(
          shouldFetchRows && !rowResult.error
            ? (rowResult.rows || []).length === 0
            : !scoped?.metrics || Object.keys(scoped.metrics).length === 0
        );
        setMetricsError(null);
      } catch (err) {
        if (!cancelled) setMetricsError(readableError(err, "Could not load the measurements."));
      } finally {
        if (!cancelled) setLoadingMetrics(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [
    datasetId,
    numericImageId,
    activeProfileId,
    includeInProgress,
    includeUnreviewed,
    shouldFetchRows,
  ]);

  // -- Derived ---------------------------------------------------------------

  const catalogMap = useMemo(() => buildMetricCatalogMap(catalog), [catalog]);
  const labelIdToName = useMemo(
    () => (imageSummary?.labels ? createLabelIdToNameMap(imageSummary.labels) : {}),
    [imageSummary]
  );
  const featuredMetric = useMemo(
    () => pickFeaturedMetric(imageSummary?.metrics, catalogMap),
    [imageSummary, catalogMap]
  );

  const measuredImage = useMemo(
    () => ({
      width: naturalSize?.width,
      height: naturalSize?.height,
      scale_x: pixelScale?.scale_x ?? null,
      scale_y: pixelScale?.scale_y ?? null,
      unit: pixelScale?.unit || null,
    }),
    [naturalSize, pixelScale]
  );

  // Only objects with at least one non-null profile metric value are drawn as subjects;
  // everything else on the image is context. The backend export returns one row for every
  // filtered contour (including contextual only-children or profile-excluded contours) with
  // null metric cells, so row presence alone does not mean the contour was measured.
  const contextIds = useMemo(() => {
    if (rowsLoadedEmpty) {
      return new Set(contours.map((c) => c.id));
    }
    if (rows.length > 0) {
      const measured = new Set(
        rows
          .filter((row) =>
            Object.entries(row).some(
              ([key, value]) =>
                !NON_METRIC_ROW_KEYS.has(key) &&
                !key.startsWith("meta_") &&
                value !== null &&
                value !== undefined &&
                !Number.isNaN(value)
            )
          )
          .map((row) => row.contour_id)
      );
      return new Set(contours.filter((c) => !measured.has(c.id)).map((c) => c.id));
    }
    // When table rows are intentionally unavailable (e.g. showTable is false due to
    // study.table=off or lack of export permission, or row request failed), determine
    // context contours from the active inclusion filters and profile metrics.
    // Note: Contextual metrics that omit individual contours (such as an only child without
    // nearest neighbour) remain a known limitation until a backend measured-ID endpoint exists.
    if (imageSummary) {
      const measuredLabelIds = new Set(
        imageSummary.metrics ? Object.keys(imageSummary.metrics).map(String) : []
      );
      const isContourMeasured = (c) => {
        if (c.label_id == null) return false;
        if (!measuredLabelIds.has(String(c.label_id))) return false;
        if (!includeUnreviewed) {
          const reviewed = Array.isArray(c.reviewed_by)
            ? c.reviewed_by.length > 0
            : Boolean(c.reviewed);
          if (!reviewed) return false;
        }
        return true;
      };
      return new Set(contours.filter((c) => !isContourMeasured(c)).map((c) => c.id));
    }
    return new Set();
  }, [
    contours,
    rows,
    rowsLoadedEmpty,
    imageSummary,
    includeUnreviewed,
  ]);

  const colorFor = useCallback(
    (contour) => (contour.label_id ? getLabelColor(contour.label_id) : "#94a3b8"),
    []
  );

  // Identifies the row set for the Perspective table, exactly as the dataset page does:
  // everything that changes which contours are in it and which columns they carry.
  const dataKey = `${datasetId}:${numericImageId}:${activeProfileId}:${includeInProgress}:${includeUnreviewed}`;

  const goToIndex = useCallback(
    (index) => {
      const target = images[index];
      if (!target) return;
      navigateToImage(target.id);
    },
    [images, navigateToImage]
  );

  const handleSelectContour = useCallback(
    (contourId) => {
      setSelectedContourId(contourId);
      if (contourId == null) {
        setZoomTarget(null);
        return;
      }
      const contour = contours.find((c) => c.id === contourId);
      // A fresh object each time, so re-selecting the same object re-frames it.
      if (contour) setZoomTarget({ ...contour });
    },
    [contours]
  );

  const handleExport = async () => {
    try {
      const url = buildQuantificationDownloadUrl(Number(datasetId), {
        profileId: activeProfileId,
        fileFormat: "csv",
        excludeNotFullyAnnotated: !includeInProgress,
        excludeUnreviewed: !includeUnreviewed,
        imageId: numericImageId,
      });
      const response = await fetch(url, { headers: getAuthHeaders() });
      if (!response.ok) throw new Error(`Export failed: ${response.status}`);
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `quantifications_${currentImage?.name || numericImageId}.csv`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      console.error("Export error:", err);
    }
  };

  // -- Render ----------------------------------------------------------------

  if (loadingMeta) {
    return (
      <DatasetManagementLayout>
        <Spinner label="Loading images…" />
      </DatasetManagementLayout>
    );
  }

  if (metaError) {
    return (
      <DatasetManagementLayout>
        <Centered>
          <p className="text-err">{metaError}</p>
        </Centered>
      </DatasetManagementLayout>
    );
  }

  if (images.length === 0) {
    return (
      <DatasetManagementLayout>
        <Centered>
          <p className="text-t2 mb-2">This dataset has no images yet.</p>
          <p className="text-sm text-t3">
            Upload some in Data Management, annotate them, and their measurements will show
            up here.
          </p>
        </Centered>
      </DatasetManagementLayout>
    );
  }

  const scaleStatus = imageSummary?.scale_status;
  const isImageCalibrated = Boolean(
    !scaleError && (scaleStatus?.display_physical || (pixelScale?.unit && pixelScale.unit !== "px"))
  );
  const hasMeasurements =
    imageSummary?.metrics && Object.keys(imageSummary.metrics).length > 0;

  /** The per-object evidence: the plain table, or the same rows pivoted by hierarchy. */
  const renderObjectSurface = () => {
    if (rows.length === 0) return null;
    if (objectView === "table") {
      return (
        <ObjectsOnImageTable
          rows={rows}
          catalogMap={catalogMap}
          selectedContourId={selectedContourId}
          onSelectContour={handleSelectContour}
        />
      );
    }
    return (
      <div className="h-[520px] flex flex-col">
        <Suspense
          fallback={
            <div className="flex-1 rounded-lg border border-ln">
              <Spinner label="Loading the pivot…" />
            </div>
          }
        >
          <QuantificationExplorer
            datasetId={datasetId}
            profileId={activeProfileId}
            dataKey={dataKey}
            rows={rows}
            theme={theme}
            // Its own saved analysis and its own opening layout — sharing the dataset
            // explorer's would overwrite that page's saved view with a grouping that only
            // means something here.
            configScope="perImage"
            buildDefault={perImageViewerConfig}
          />
        </Suspense>
      </div>
    );
  };

  return (
    <DatasetManagementLayout>
      <div className="h-full flex flex-col bg-p1">
        {/* Header: where you are, which image, and the ways out of the page. */}
        <div className="border-b border-ln flex-shrink-0 px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div className="min-w-0">
              <nav className="flex items-center gap-2 text-sm text-t3 mb-1">
                <button
                  onClick={() => navigate(`/dataset/${datasetId}/quantifications`)}
                  className="hover:text-ac transition-colors"
                >
                  Dataset Quantifications
                </button>
                <span>/</span>
                <span className="text-t2">Per image</span>
              </nav>
              <h1 className="text-2xl font-bold text-t1 truncate">
                {currentImage?.name || "…"}
              </h1>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1 bg-well rounded-lg px-1 py-1">
                <button
                  onClick={() => goToIndex(currentIndex - 1)}
                  disabled={currentIndex <= 0}
                  aria-label="Previous image"
                  className="p-1.5 rounded-md text-t2 hover:bg-hv disabled:opacity-40 disabled:hover:bg-transparent"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="text-sm text-t2 tabular-nums px-2">
                  {currentIndex + 1} / {images.length}
                </span>
                <button
                  onClick={() => goToIndex(currentIndex + 1)}
                  disabled={currentIndex >= images.length - 1}
                  aria-label="Next image"
                  className="p-1.5 rounded-md text-t2 hover:bg-hv disabled:opacity-40 disabled:hover:bg-transparent"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              <StudyViewControl study={study} />

              <button
                onClick={() =>
                  navigate(`/dataset/${datasetId}/annotate/${numericImageId}`)
                }
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-t2 bg-well rounded-lg hover:bg-hv2 transition-colors"
              >
                <SquarePen className="w-4 h-4" />
                <span>Open in workspace</span>
              </button>

              {canExport && (
                <button
                  onClick={handleExport}
                  className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-accent text-onAccent rounded-lg hover:brightness-110 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  <span>Export this image</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* The same controls as the dataset page, doing the same thing: they decide which
            objects are measured, and both pages have to answer that the same way. */}
        <div className="border-b border-ln flex-shrink-0 px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <ProfileSelector
              datasetId={Number(datasetId)}
              profiles={profiles}
              activeProfileId={activeProfileId}
              catalog={catalog}
              labels={imageSummary?.labels}
              onSelect={setActiveProfileId}
              onProfilesChanged={async (selectId = null) => {
                const response = await getQuantificationProfiles(Number(datasetId));
                const loaded = response.profiles || [];
                setProfiles(loaded);
                if (selectId !== null) setActiveProfileId(selectId);
                else if (!loaded.find((p) => p.id === activeProfileId)) {
                  const preferred = loaded.find((p) => p.is_default) || loaded[0];
                  setActiveProfileId(preferred ? preferred.id : null);
                }
              }}
            />
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-1.5 text-sm text-t2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeInProgress}
                  onChange={(event) => setIncludeInProgress(event.target.checked)}
                  className="rounded border-ln2 text-ac focus:ring-ac"
                />
                <span>Include in-progress masks</span>
              </label>
              <label className="flex items-center gap-1.5 text-sm text-t2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeUnreviewed}
                  onChange={(event) => setIncludeUnreviewed(event.target.checked)}
                  className="rounded border-ln2 text-ac focus:ring-ac"
                />
                <span>Include unreviewed objects</span>
              </label>
            </div>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
          {!hasMeasurements && !loadingMetrics && (
            <div className="flex items-start gap-3 rounded-lg border border-acLn bg-acS px-4 py-3">
              <Info className="mt-0.5 h-5 w-5 flex-shrink-0 text-ac" />
              <p className="text-sm text-ac">
                Nothing on this image is measured under the current settings. Its objects
                may still be in progress or unreviewed — use the toggles above to include
                them{rowsMessage ? ` (${rowsMessage})` : ""}.
              </p>
            </div>
          )}

          {/* Only shown for an uncalibrated image: dataset-wide inconsistency is not this
              page's problem, since one image is always consistent with itself. */}
          {scaleStatus && !isImageCalibrated && !scaleError && (
            <div className="flex items-start gap-3 rounded-lg border border-warnLn bg-warnBg px-4 py-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-warn" />
              <p className="text-sm text-warn">
                This image has no scale, so its measurements are in pixels. Calibrate it to
                compare it against images measured in real-world units.
              </p>
            </div>
          )}

          {scaleError && (
            <div className="flex items-start gap-3 rounded-lg border border-warnLn bg-warnBg px-4 py-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-warn" />
              <p className="text-sm text-warn">
                Could not load calibration for this image. Scale information is unavailable.
              </p>
            </div>
          )}

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {/* Left: the image the numbers are about. */}
            <div className="bg-p1 rounded-lg border border-ln p-4 flex flex-col">
              <div className="flex items-center justify-between gap-3 mb-3">
                <h3 className="text-sm font-semibold text-t1">Image</h3>
                <span className="text-xs text-t3 font-mono">
                  {naturalSize ? `${naturalSize.width} × ${naturalSize.height}` : "…"}
                  {!scaleError && isImageCalibrated && measuredImage.scale_x != null &&
                    ` · 1 px = ${measuredImage.scale_x} ${measuredImage.unit || scaleStatus?.display_unit || ""}`}
                </span>
              </div>

              <div className="h-[420px] rounded-md overflow-hidden bg-well">
                {loadingImage ? (
                  <Spinner label="Loading image…" />
                ) : imageError ? (
                  <div className="h-full flex flex-col items-center justify-center p-4 text-center">
                    <p className="text-err text-sm font-medium">{imageError}</p>
                  </div>
                ) : (
                  <AnnotationViewerCanvas
                    imageSrc={imageSrc}
                    contours={contours}
                    selectedId={selectedContourId}
                    onSelect={handleSelectContour}
                    zoomTarget={zoomTarget}
                    colorFor={colorFor}
                    contextIds={contextIds}
                    onNaturalSize={setNaturalSize}
                  />
                )}
              </div>

              <p className="text-[11px] text-t3 mt-2">
                {study.segmentations
                  ? "Click an object here or a row below — the two stay in sync."
                  : "Segmentation outlines are hidden for this view."}
              </p>

              <div className="mt-3 pt-3 border-t border-ln">
                <ImageFilmstrip
                  images={images}
                  selectedId={numericImageId}
                  onSelect={(image) => navigateToImage(image.id)}
                  size="md"
                />
              </div>
            </div>

            {/* Right: what this image measures, and how that compares. */}
            <div className="space-y-4">
              {loadingMetrics ? (
                <Spinner
                  label="Loading measurements…"
                  hint="Measurements are computed on demand, so this can take a moment the first time."
                />
              ) : metricsError ? (
                <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-center">
                  <p className="text-err text-sm font-medium">{metricsError}</p>
                </div>
              ) : !imageSummary ? (
                <Spinner
                  label="Loading measurements…"
                  hint="Measurements are computed on demand, so this can take a moment the first time."
                />
              ) : (
                <>
                  <ImageStatCards
                    imageMetrics={imageSummary?.metrics}
                    metricKey={featuredMetric}
                    catalog={catalogMap[featuredMetric]}
                    objectCounts={imageSummary?.object_counts_per_label_id}
                    image={measuredImage}
                    scaleStatus={scaleStatus}
                    scaleError={Boolean(scaleError)}
                  />
                  {featuredMetric && (
                    <LabelComparisonBars
                      imageMetrics={imageSummary?.metrics}
                      datasetMetrics={datasetSummary?.metrics}
                      metricKey={featuredMetric}
                      catalog={catalogMap[featuredMetric]}
                      labelIdToName={labelIdToName}
                    />
                  )}
                </>
              )}
            </div>
          </div>

          <ImageMetricGrid
            imageMetrics={imageSummary?.metrics}
            datasetMetrics={datasetSummary?.metrics}
            catalogMap={catalogMap}
          />

          {showTable && rowsError && (
            <div className="flex items-start gap-3 rounded-lg border border-warnLn bg-warnBg px-4 py-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-warn" />
              <p className="text-sm text-warn">{rowsError}</p>
            </div>
          )}

          {showTable && !rowsError && rows.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="h-1 w-8 bg-accent rounded-full flex-shrink-0" />
                  <h3 className="text-sm font-semibold text-t2 uppercase tracking-wide">
                    Individual objects
                  </h3>
                </div>
                {/* The plain table and the pivot are the same rows. The pivot opens
                    grouped by parent, which is the one thing a flat table of one image
                    cannot show: which objects sit inside which. */}
                <div className="flex items-center gap-1 bg-well rounded-lg p-1">
                  <ViewButton
                    icon={Table2}
                    label="Table"
                    active={objectView === "table"}
                    onClick={() => setObjectView("table")}
                  />
                  <ViewButton
                    icon={LayoutGrid}
                    label="Pivot"
                    active={objectView === "pivot"}
                    onClick={() => setObjectView("pivot")}
                  />
                </div>
              </div>
              {renderObjectSurface()}
            </div>
          )}

          {!canExport && (
            <div className="flex items-start gap-3 rounded-lg border border-ln bg-well px-4 py-3">
              <Info className="mt-0.5 h-5 w-5 flex-shrink-0 text-t3" />
              <p className="text-sm text-t2">
                The per-object table needs the quantification export permission, which your
                role on this dataset does not include. The summary above is unaffected.
              </p>
            </div>
          )}
        </div>
      </div>
    </DatasetManagementLayout>
  );
};

export default ImageQuantificationPage;
