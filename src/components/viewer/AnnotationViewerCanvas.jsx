import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Maximize2, Minus, Plus } from 'lucide-react';

/** Shared empty set, so the default prop does not remount the paths each render. */
const EMPTY_IDS = new Set();

const MIN_ZOOM = 0.2;
const MAX_ZOOM = 12;
const ZOOM_STEP = 1.3;
const DRAG_THRESHOLD = 5;

/** Bounding box of a contour, from its normalized (0–1) coordinate arrays. */
const contourBounds = (contour) => {
  const xs = contour?.x || [];
  const ys = contour?.y || [];
  if (xs.length === 0 || ys.length === 0) return null;
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
};

/**
 * SVG path for a contour, built from its normalized (0–1) coordinates scaled to
 * the dimensions of the bitmap actually on screen.
 *
 * The normalized coordinates are the source of truth here, NOT the backend's
 * precomputed `path`: that path bakes in the `Images.width/height` stored in the
 * database, and rows ingested before the thumbnail-mutation fix carry thumbnail
 * dimensions there — which squashed every contour into the top-left corner at
 * 1/20 scale. Scaling by the loaded image's own size cannot disagree with the
 * viewBox, which is derived from the same measurement. The backend path is only
 * a fallback for contours that arrive without coordinate arrays.
 */
const pathFor = (contour, width, height) => {
  const xs = contour?.x || [];
  const ys = contour?.y || [];
  if (xs.length < 3 || ys.length < 3) return contour?.path || null;
  let d = `M ${Math.round(xs[0] * width)} ${Math.round(ys[0] * height)}`;
  for (let i = 1; i < xs.length; i += 1) {
    d += ` L ${Math.round(xs[i] * width)} ${Math.round(ys[i] * height)}`;
  }
  return `${d} Z`;
};

/**
 * Read-only image view with the annotations drawn over it.
 *
 * Deliberately not the annotation canvas: no Konva, no WebSocket, no editing.
 * Contours are plain SVG paths — the backend already computes a `path` for each
 * one in pixel coordinates — inside a CSS-transformed wrapper for pan and zoom.
 *
 * @param {Object} props
 * @param {string} props.imageSrc - Data URL or URL of the image.
 * @param {Array} props.contours - Contours with `path`, `x`, `y`, `id`.
 * @param {number|null} props.selectedId - Contour to highlight.
 * @param {Function} props.onSelect - Called with a contour id (or null) on click.
 * @param {Object|null} props.zoomTarget - Contour to frame; changing it re-frames.
 * @param {Function} props.colorFor - contour -> CSS color.
 * @param {Set<number>} [props.contextIds] - Contours drawn as surroundings rather
 *   than as the subject: dashed hairline, barely any fill. They stay clickable —
 *   the point of showing them is that the viewer can act on one. Empty by default,
 *   which renders every contour as a subject exactly as before.
 * @param {Function} [props.onNaturalSize] - Called with `{width, height}` once the
 *   bitmap has decoded. The size is measured here anyway (the viewBox is derived
 *   from it), so a caller that needs the image's real pixel dimensions — the
 *   per-image quantification page, to say what fraction of the frame the objects
 *   cover — can have them without decoding the image a second time.
 */
const AnnotationViewerCanvas = ({
  imageSrc,
  contours = [],
  selectedId = null,
  onSelect,
  zoomTarget = null,
  colorFor,
  contextIds = EMPTY_IDS,
  onNaturalSize,
}) => {
  const containerRef = useRef(null);
  const [natural, setNatural] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const dragRef = useRef(null);
  const isDraggingRef = useRef(false);

  // Track container dimensions to recompute fit and minZoom after layout/viewport changes.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    setContainerSize({
      width: container.clientWidth,
      height: container.clientHeight,
    });

    if (typeof ResizeObserver === 'undefined') {
      const handleWindowResize = () => {
        if (containerRef.current) {
          setContainerSize({
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight,
          });
        }
      };
      window.addEventListener('resize', handleWindowResize);
      return () => window.removeEventListener('resize', handleWindowResize);
    }

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setContainerSize({ width, height });
      }
    });

    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Centre of the selected contour in natural-pixel coordinates, or null when
  // nothing is selected. This is the anchor zooming pivots around while an
  // instance is in focus, so the object stays put instead of drifting toward or
  // away from the image centre.
  const selectedCenter = useMemo(() => {
    if (selectedId == null || !natural.width) return null;
    const contour = contours.find((c) => c.id === selectedId);
    const bounds = contour && contourBounds(contour);
    if (!bounds) return null;
    return {
      x: ((bounds.minX + bounds.maxX) / 2) * natural.width,
      y: ((bounds.minY + bounds.maxY) / 2) * natural.height,
    };
  }, [selectedId, contours, natural]);

  // Precomputed per contour: SAM-drawn contours run to thousands of points, and
  // the SVG re-renders on every zoom tick — rebuilding the path strings each
  // time would jank the wheel.
  const drawableContours = useMemo(
    () =>
      contours
        .map((contour) => ({
          contour,
          d: pathFor(contour, natural.width, natural.height),
        }))
        .filter((entry) => entry.d),
    [contours, natural]
  );

  /** Scale that fits the whole image in the viewport. */
  const fitScale = useCallback(
    (dims = natural) => {
      const container = containerRef.current;
      if (!container || !dims?.width || !dims?.height) return 1;
      const clientWidth = container.clientWidth || containerSize.width;
      const clientHeight = container.clientHeight || containerSize.height;
      if (!clientWidth || !clientHeight) return 1;
      return Math.min(clientWidth / dims.width, clientHeight / dims.height) || 1;
    },
    [natural, containerSize]
  );

  const resetView = useCallback(() => {
    setZoom(fitScale());
    setOffset({ x: 0, y: 0 });
  }, [fitScale]);

  /**
   * Effective minimum zoom: calculated from the live container size so zooming out
   * never clamps upward after a viewport shrink.
   */
  const getMinZoom = useCallback(() => {
    const fit = fitScale();
    return Math.min(MIN_ZOOM, fit * 0.8);
  }, [fitScale]);

  const minZoom = useMemo(() => getMinZoom(), [getMinZoom]);

  const loadedImageRef = useRef(null);
  const lastFramedRef = useRef({ target: null, src: null });

  /**
   * Fit on load once the actual decoded dimensions of the new image are available,
   * preventing race conditions where Image B is fitted using Image A's dimensions.
   */
  const handleImageLoad = (e) => {
    const size = {
      width: e.target.naturalWidth,
      height: e.target.naturalHeight,
    };
    loadedImageRef.current = imageSrc;
    setNatural(size);
    onNaturalSize?.(size);
    if (!zoomTarget) {
      setZoom(fitScale(size));
      setOffset({ x: 0, y: 0 });
    }
  };

  // Frame the requested contour. Recomputed when a new zoom target arrives (or on
  // repeated clicks creating a fresh object) or when navigating to a new image once loaded,
  // while container resizing preserves the user's current view.
  useEffect(() => {
    const container = containerRef.current;
    if (!zoomTarget || !container || !natural.width) {
      if (!zoomTarget) lastFramedRef.current = { target: null, src: null };
      return;
    }
    // Wait until the incoming image has actually loaded so we never frame with the previous image's dimensions
    if (loadedImageRef.current !== imageSrc) return;

    if (lastFramedRef.current.target === zoomTarget && lastFramedRef.current.src === imageSrc) {
      return;
    }
    lastFramedRef.current = { target: zoomTarget, src: imageSrc };

    const bounds = contourBounds(zoomTarget);
    if (!bounds) {
      setZoom(fitScale());
      setOffset({ x: 0, y: 0 });
      return;
    }

    const { clientWidth, clientHeight } = container;
    const boxW = Math.max((bounds.maxX - bounds.minX) * natural.width, 1);
    const boxH = Math.max((bounds.maxY - bounds.minY) * natural.height, 1);
    // Leave a margin so the object is not flush against the viewport edge.
    const target = Math.min(clientWidth / (boxW * 1.6), clientHeight / (boxH * 1.6));
    const nextZoom = Math.min(Math.max(target, getMinZoom()), MAX_ZOOM);

    const centerX = ((bounds.minX + bounds.maxX) / 2) * natural.width;
    const centerY = ((bounds.minY + bounds.maxY) / 2) * natural.height;

    setZoom(nextZoom);
    setOffset({
      x: (natural.width / 2 - centerX) * nextZoom,
      y: (natural.height / 2 - centerY) * nextZoom,
    });
  }, [zoomTarget, imageSrc, natural, fitScale, getMinZoom]);

  /**
   * Zoom to `nextZoom` while keeping `focal` (a point in natural-pixel
   * coordinates) pinned to its current spot on screen.
   *
   * The wrapper is transformed with `transformOrigin: center center`, so a local
   * point P lands on screen at `containerCentre + offset + zoom * (P - imageCentre)`.
   * Holding that screen position fixed across a zoom change gives
   * `offset' = offset + (zoom - nextZoom) * (P - imageCentre)`.
   *
   * `focal` defaults to the selected instance's centre (so the buttons pivot on
   * it too) and falls back to the image centre when nothing is selected.
   */
  const zoomTo = useCallback(
    (nextZoom, focal) => {
      const currentMinZoom = getMinZoom();
      const clamped = Math.min(Math.max(nextZoom, currentMinZoom), MAX_ZOOM);
      if (clamped === zoom || !natural.width) return;
      const imageCenter = { x: natural.width / 2, y: natural.height / 2 };
      const pivot = focal || selectedCenter || imageCenter;
      setOffset({
        x: offset.x + (zoom - clamped) * (pivot.x - imageCenter.x),
        y: offset.y + (zoom - clamped) * (pivot.y - imageCenter.y),
      });
      setZoom(clamped);
    },
    [zoom, offset, natural, selectedCenter, getMinZoom]
  );

  const handleWheel = (e) => {
    e.preventDefault();
    if (!natural.width || !containerRef.current) return;
    const nextZoom = zoom * (e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);

    // Pivot around whatever the cursor is hovering over: invert the transform
    // to recover the natural-pixel point currently under the pointer.
    const rect = containerRef.current.getBoundingClientRect();
    const cursorFromCenter = {
      x: e.clientX - rect.left - rect.width / 2,
      y: e.clientY - rect.top - rect.height / 2,
    };
    const focal = {
      x: natural.width / 2 + (cursorFromCenter.x - offset.x) / zoom,
      y: natural.height / 2 + (cursorFromCenter.y - offset.y) / zoom,
    };
    zoomTo(nextZoom, focal);
  };

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    dragRef.current = { x: e.clientX, y: e.clientY, offset };
    isDraggingRef.current = false;
  };

  const handleMouseMove = (e) => {
    if (!dragRef.current) return;
    const { x, y, offset: start } = dragRef.current;
    const dx = e.clientX - x;
    const dy = e.clientY - y;
    if (!isDraggingRef.current && (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD)) {
      isDraggingRef.current = true;
    }
    if (isDraggingRef.current) {
      setOffset({ x: start.x + dx, y: start.y + dy });
    }
  };

  const endDrag = () => {
    dragRef.current = null;
    // Delay resetting so trailing click events triggered on mouseup are suppressed
    setTimeout(() => {
      isDraggingRef.current = false;
    }, 0);
  };

  const handleClickCapture = (e) => {
    if (isDraggingRef.current) {
      e.stopPropagation();
    }
  };

  return (
    <div className="relative w-full h-full bg-app overflow-hidden">
      <div
        ref={containerRef}
        className="w-full h-full flex items-center justify-center cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={endDrag}
        onMouseLeave={endDrag}
        onClickCapture={handleClickCapture}
      >
        {imageSrc && (
          <div
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
              width: natural.width || undefined,
              height: natural.height || undefined,
              position: 'relative',
              flexShrink: 0,
            }}
          >
            <img
              src={imageSrc}
              alt="Annotated"
              draggable={false}
              onLoad={handleImageLoad}
              style={{ display: 'block', width: '100%', height: '100%' }}
            />

            {natural.width > 0 && (
              <svg
                viewBox={`0 0 ${natural.width} ${natural.height}`}
                className="absolute inset-0 w-full h-full"
                // Clicking empty space clears the selection.
                onClick={() => {
                  if (!isDraggingRef.current) onSelect?.(null);
                }}
              >
                {drawableContours.map(({ contour, d }) => {
                  const isSelected = contour.id === selectedId;
                  // Three tiers, not two: the selection, the other subjects, and
                  // the surroundings. Context needs to be readable enough to spot
                  // a duplicate against without competing with the subject, so it
                  // gets a dashed hairline and almost no fill.
                  const isContext = !isSelected && contextIds.has(contour.id);
                  const color = colorFor ? colorFor(contour) : '#38bdf8';
                  return (
                    <path
                      key={contour.id}
                      d={d}
                      fill={color}
                      fillOpacity={isSelected ? 0.4 : isContext ? 0.05 : 0.18}
                      stroke={color}
                      strokeOpacity={isContext ? 0.6 : 1}
                      // Keep the outline a constant width on screen however far
                      // the user has zoomed in. Same for the dash pattern, which
                      // would otherwise turn solid at high zoom.
                      strokeWidth={(isSelected ? 3 : isContext ? 1 : 1.5) / zoom}
                      strokeDasharray={isContext ? `${5 / zoom} ${4 / zoom}` : undefined}
                      className="cursor-pointer"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (!isDraggingRef.current) onSelect?.(contour.id);
                      }}
                    />
                  );
                })}
              </svg>
            )}
          </div>
        )}
      </div>

      {/* Zoom controls */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-1 bg-p1 rounded-lg shadow p-1">
        <button
          onClick={() => zoomTo(zoom * ZOOM_STEP)}
          className="p-1.5 hover:bg-hv rounded transition-colors"
          title="Zoom in"
        >
          <Plus className="w-4 h-4 text-t2" />
        </button>
        <button
          onClick={() => zoomTo(zoom / ZOOM_STEP)}
          disabled={zoom <= minZoom}
          className="p-1.5 hover:bg-hv rounded transition-colors disabled:opacity-40 disabled:pointer-events-none"
          title="Zoom out"
        >
          <Minus className="w-4 h-4 text-t2" />
        </button>
        <button
          onClick={resetView}
          className="p-1.5 hover:bg-hv rounded transition-colors"
          title="Fit to window"
        >
          <Maximize2 className="w-4 h-4 text-t2" />
        </button>
      </div>
    </div>
  );
};

export default AnnotationViewerCanvas;
