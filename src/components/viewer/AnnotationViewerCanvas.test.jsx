import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import AnnotationViewerCanvas from "./AnnotationViewerCanvas";

const mockContours = [
  {
    id: 1,
    // Left side of the image (x: 5% - 15%, y: 10% - 20%) -> center at x=400, y=450 in 4000x3000
    x: [0.05, 0.15, 0.15, 0.05],
    y: [0.1, 0.1, 0.2, 0.2],
  },
  {
    id: 2,
    x: [0.7, 0.8, 0.8, 0.7],
    y: [0.7, 0.7, 0.8, 0.8],
  },
];

const parseTransform = (element) => {
  const transform = element.style.transform || "";
  const translateMatch = transform.match(/translate\(([^p]+)px,\s*([^p]+)px\)/);
  const scaleMatch = transform.match(/scale\(([^)]+)\)/);
  return {
    scale: scaleMatch ? parseFloat(scaleMatch[1]) : 1,
    x: translateMatch ? parseFloat(translateMatch[1]) : 0,
    y: translateMatch ? parseFloat(translateMatch[2]) : 0,
  };
};

describe("AnnotationViewerCanvas interactions", () => {
  let mockWidth = 600;
  let mockHeight = 420;

  beforeEach(() => {
    mockWidth = 600;
    mockHeight = 420;

    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
      configurable: true,
      get: () => mockWidth,
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get: () => mockHeight,
    });
    Object.defineProperty(HTMLElement.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => ({
        left: 0,
        top: 0,
        width: mockWidth,
        height: mockHeight,
        right: mockWidth,
        bottom: mockHeight,
      }),
    });
    Object.defineProperty(HTMLImageElement.prototype, "naturalWidth", {
      configurable: true,
      get: () => 4000,
    });
    Object.defineProperty(HTMLImageElement.prototype, "naturalHeight", {
      configurable: true,
      get: () => 3000,
    });
  });

  const renderCanvas = (props = {}) => {
    const onSelect = vi.fn();
    const result = render(
      <AnnotationViewerCanvas
        imageSrc="data:image/png;base64,mock"
        contours={mockContours}
        onSelect={onSelect}
        {...props}
      />
    );

    const img = screen.getByAltText("Annotated");
    fireEvent.load(img);

    return { ...result, onSelect };
  };

  it("selects a contour when clicked without dragging", () => {
    const { onSelect } = renderCanvas();

    const paths = document.querySelectorAll("svg.absolute path");
    expect(paths.length).toBe(2);

    fireEvent.mouseDown(paths[0], { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseUp(paths[0], { clientX: 100, clientY: 100 });
    fireEvent.click(paths[0]);

    expect(onSelect).toHaveBeenCalledWith(1);
  });

  it("suppresses selection and auto-zoom when user drags over a contour", () => {
    const { onSelect } = renderCanvas();

    const paths = document.querySelectorAll("svg.absolute path");

    fireEvent.mouseDown(paths[0], { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseMove(paths[0], { clientX: 120, clientY: 120 });
    fireEvent.mouseUp(paths[0], { clientX: 120, clientY: 120 });
    fireEvent.click(paths[0]);

    expect(onSelect).not.toHaveBeenCalled();
  });

  it("clears selection when clicking empty space without dragging", () => {
    const { onSelect } = renderCanvas({ selectedId: 1 });

    const svg = document.querySelector("svg");
    fireEvent.mouseDown(svg, { clientX: 10, clientY: 10, button: 0 });
    fireEvent.mouseUp(svg, { clientX: 10, clientY: 10 });
    fireEvent.click(svg);

    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("preserves zoom and pan position when container resizes, but recomputes minZoom", () => {
    renderCanvas();
    const transformedDiv = document.querySelector("img").parentElement;

    // 1. Zoom in and pan to inspect a specific area
    const zoomInBtn = screen.getByTitle("Zoom in");
    act(() => {
      fireEvent.click(zoomInBtn);
    });
    act(() => {
      fireEvent.click(zoomInBtn);
    });

    // Pan by dragging across the canvas
    const container = screen.getByAltText("Annotated").closest(".cursor-grab");
    act(() => {
      fireEvent.mouseDown(container, { clientX: 200, clientY: 200, button: 0 });
      fireEvent.mouseMove(container, { clientX: 280, clientY: 250 });
      fireEvent.mouseUp(container, { clientX: 280, clientY: 250 });
    });

    const beforeResize = parseTransform(transformedDiv);
    expect(beforeResize.scale).toBeGreaterThan(0.2);
    expect(beforeResize.x).not.toBe(0);
    expect(beforeResize.y).not.toBe(0);

    // 2. Viewport shrinks (e.g. drawer opens or window is resized)
    mockWidth = 300;
    mockHeight = 200;

    act(() => {
      window.dispatchEvent(new Event("resize"));
    });

    // 3. Current view MUST NOT reset to fit/center; it must preserve zoom and offset
    const afterResize = parseTransform(transformedDiv);
    expect(afterResize.scale).toBe(beforeResize.scale);
    expect(afterResize.x).toBe(beforeResize.x);
    expect(afterResize.y).toBe(beforeResize.y);

    // 4. Zooming out repeatedly from here should clamp against the new smaller viewport (200/3000 = 0.0667)
    // and NEVER jump back up to the old scale (0.14 or 0.2).
    const zoomOutBtn = screen.getByTitle("Zoom out");
    for (let i = 0; i < 10; i += 1) {
      act(() => {
        fireEvent.click(zoomOutBtn);
      });
    }

    const fullyZoomedOut = parseTransform(transformedDiv);
    expect(fullyZoomedOut.scale).toBeLessThanOrEqual(0.067);
    expect(fullyZoomedOut.scale).toBeGreaterThan(0.01);
  });

  it("wheel zoom keeps the exact image content under the cursor stationary, ignoring selected object", () => {
    // Select contour 1 which is positioned on the far LEFT (x around 400)
    renderCanvas({ selectedId: 1 });
    const transformedDiv = document.querySelector("img").parentElement;
    const container = screen.getByAltText("Annotated").closest(".cursor-grab");

    const before = parseTransform(transformedDiv);
    const imageCenterX = 4000 / 2; // 2000
    const imageCenterY = 3000 / 2; // 1500

    // Position the mouse on the far RIGHT of the viewport (clientX: 550, clientY: 100)
    const cursorX = 550;
    const cursorY = 100;
    const cursorFromCenterX = cursorX - 600 / 2; // 250
    const cursorFromCenterY = cursorY - 420 / 2; // -110

    // Calculate the natural-pixel point under the cursor before zooming
    const naturalPointX = imageCenterX + (cursorFromCenterX - before.x) / before.scale;
    const naturalPointY = imageCenterY + (cursorFromCenterY - before.y) / before.scale;

    // Zoom in with mouse wheel at the cursor position
    act(() => {
      fireEvent.wheel(container, {
        clientX: cursorX,
        clientY: cursorY,
        deltaY: -100, // Zoom in
      });
    });

    const after = parseTransform(transformedDiv);
    expect(after.scale).toBeGreaterThan(before.scale);

    // Calculate where that same natural-pixel point lands on screen after zoom
    const screenPositionAfterX = after.x + after.scale * (naturalPointX - imageCenterX);
    const screenPositionAfterY = after.y + after.scale * (naturalPointY - imageCenterY);

    // The point on the image directly under the cursor MUST remain under the cursor on screen
    expect(screenPositionAfterX).toBeCloseTo(cursorFromCenterX, 1);
    expect(screenPositionAfterY).toBeCloseTo(cursorFromCenterY, 1);

    // Verify it did NOT pivot on the selected object center (x ~ 400)
    const selectedCenterX = 0.1 * 4000; // 400
    const buggyOffsetX = before.x + (before.scale - after.scale) * (selectedCenterX - imageCenterX);
    expect(Math.abs(after.x - buggyOffsetX)).toBeGreaterThan(20);
  });

  it("fits newly arrived image using its actual dimensions rather than previous image dimensions", () => {
    let currentNaturalWidth = 4000;
    let currentNaturalHeight = 3000;
    Object.defineProperty(HTMLImageElement.prototype, "naturalWidth", {
      configurable: true,
      get: () => currentNaturalWidth,
    });
    Object.defineProperty(HTMLImageElement.prototype, "naturalHeight", {
      configurable: true,
      get: () => currentNaturalHeight,
    });

    const { rerender } = render(
      <AnnotationViewerCanvas
        imageSrc="data:image/png;base64,imageA"
        contours={mockContours}
      />
    );

    const img = screen.getByAltText("Annotated");
    fireEvent.load(img);

    const transformedDiv = img.parentElement;
    const transformA = parseTransform(transformedDiv);
    // 4000x3000 in 600x420 -> fitScale is 420/3000 = 0.14
    expect(transformA.scale).toBeCloseTo(0.14, 2);

    // Switch to Image B with different dimensions (800x600)
    currentNaturalWidth = 800;
    currentNaturalHeight = 600;

    rerender(
      <AnnotationViewerCanvas
        imageSrc="data:image/png;base64,imageB"
        contours={mockContours}
      />
    );

    // When Image B finishes loading
    fireEvent.load(img);

    const transformB = parseTransform(transformedDiv);
    // 800x600 in 600x420 -> fitScale is min(600/800, 420/600) = 0.7
    expect(transformB.scale).toBeCloseTo(0.7, 2);
    expect(transformB.scale).not.toBeCloseTo(0.14, 2);
    expect(transformB.x).toBe(0);
    expect(transformB.y).toBe(0);
  });

  it("preserves view without reframing when container resizes while zoomTarget is set", () => {
    const target = mockContours[0];
    renderCanvas({ zoomTarget: target, selectedId: target.id });

    const transformedDiv = document.querySelector("img").parentElement;
    const framed = parseTransform(transformedDiv);
    expect(framed.scale).toBeGreaterThan(0.14);

    // User pans to adjust view while inspecting the target
    const container = screen.getByAltText("Annotated").closest(".cursor-grab");
    act(() => {
      fireEvent.mouseDown(container, { clientX: 200, clientY: 200, button: 0 });
      fireEvent.mouseMove(container, { clientX: 260, clientY: 240 });
      fireEvent.mouseUp(container, { clientX: 260, clientY: 240 });
    });

    const beforeResize = parseTransform(transformedDiv);
    expect(beforeResize.x).not.toBe(framed.x);

    // Viewport shrinks (e.g. side drawer opened or window resized)
    mockWidth = 300;
    mockHeight = 200;

    act(() => {
      window.dispatchEvent(new Event("resize"));
    });

    // Must NOT reframe the selected contour back to center; user pan and zoom must be preserved
    const afterResize = parseTransform(transformedDiv);
    expect(afterResize.scale).toBe(beforeResize.scale);
    expect(afterResize.x).toBe(beforeResize.x);
    expect(afterResize.y).toBe(beforeResize.y);

    // Dynamic minimum zoom is still updated: zooming out reaches below new fitScale (200/3000 = 0.0667)
    const zoomOutBtn = screen.getByTitle("Zoom out");
    for (let i = 0; i < 15; i += 1) {
      act(() => {
        fireEvent.click(zoomOutBtn);
      });
    }

    const fullyZoomedOut = parseTransform(transformedDiv);
    expect(fullyZoomedOut.scale).toBeLessThanOrEqual(0.067);
    expect(fullyZoomedOut.scale).toBeGreaterThan(0.01);
  });

  it("frames target correctly when switching images while zoomTarget is set", () => {
    let currentNaturalWidth = 4000;
    let currentNaturalHeight = 3000;
    Object.defineProperty(HTMLImageElement.prototype, "naturalWidth", {
      configurable: true,
      get: () => currentNaturalWidth,
    });
    Object.defineProperty(HTMLImageElement.prototype, "naturalHeight", {
      configurable: true,
      get: () => currentNaturalHeight,
    });

    const targetA = {
      id: 10,
      x: [0.05, 0.15, 0.15, 0.05],
      y: [0.1, 0.1, 0.2, 0.2],
    };

    const { rerender } = render(
      <AnnotationViewerCanvas
        imageSrc="data:image/png;base64,imageA"
        contours={[targetA]}
        zoomTarget={targetA}
      />
    );

    const img = screen.getByAltText("Annotated");
    fireEvent.load(img);

    const transformedDiv = img.parentElement;
    const transformA = parseTransform(transformedDiv);
    // On 4000x3000 image, targetA (400x300px) frames at targetScale = min(600/(400*1.6), 420/(300*1.6)) = 0.875
    expect(transformA.scale).toBeCloseTo(0.875, 2);
    expect(transformA.x).toBeCloseTo(1400, 0);

    // Switch to Image B (800x600) with a targetB centered on Image B
    currentNaturalWidth = 800;
    currentNaturalHeight = 600;
    const targetB = {
      id: 20,
      x: [0.4, 0.6, 0.6, 0.4],
      y: [0.4, 0.4, 0.6, 0.6],
    };

    rerender(
      <AnnotationViewerCanvas
        imageSrc="data:image/png;base64,imageB"
        contours={[targetB]}
        zoomTarget={targetB}
      />
    );

    fireEvent.load(img);

    const transformB = parseTransform(transformedDiv);
    // On 800x600 image, targetB (160x120px) frames at targetScale = min(600/(160*1.6), 420/(120*1.6)) = 2.1875
    expect(transformB.scale).toBeCloseTo(2.1875, 2);
    // targetB is centered at (400, 300) on 800x600, so offset is (400 - 400) * 2.1875 = 0
    expect(transformB.x).toBeCloseTo(0, 1);
    expect(transformB.y).toBeCloseTo(0, 1);

    expect(transformB.scale).not.toBeCloseTo(transformA.scale, 2);
  });
});
