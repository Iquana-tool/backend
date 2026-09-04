import React, { useEffect, useRef, useState } from 'react';
import {
  useContextMenuVisible,
  useContextMenuX,
  useContextMenuY,
  useContextMenuTargetObjectId,
  useHideContextMenu,
  useEnterFocusMode,
  useObjectsList,
  useSelectedObjects,
  useImageObject,
  useUpdateObject,
  useRemoveObject,
  useEnterRefinementMode,
  useCurrentTool,
  useSetCurrentTool,
  useRefinementModeActive,
  useFocusModeActive,
  useFocusModeObjectId,
  useExitFocusMode,
  useEnterEditMode,
  useStartLineEdit,
  useSelectObject,
  useCurrentMaskId,
} from '../../../stores/selectors/annotationSelectors';
import { useRefinementMode } from '../../../hooks/useRefinementMode';
import { useZoomToObject } from '../../../hooks/useZoomToObject';
import { useLabelSelection } from '../../../hooks/useLabelSelection';
import { useLabelsHierarchy } from '../../../hooks/useLabelsHierarchy';
import { getChildLabels, resolveParentLabelId } from '../../../utils/labelHierarchy';
import useSuggestSimilar from '../workspace/useSuggestSimilar';
import { useDataset } from '../../../contexts/DatasetContext';
import { calculateRenderedImageDimensions } from '../../../utils/canvasUtils';
import { deleteObject } from '../../../utils/objectOperations';
import { mergeObjects } from '../../../utils/contourOperations';
import { useToast } from '../../../contexts/ToastContext';
import { hasValidLabel } from '../../../stores/utils/labelValidation';
import annotationSession from '../../../services/annotationSession';
import { getContourId } from '../../../utils/objectUtils';
import ContextMenuItem from './ContextMenuItem';
import HierarchicalLabelList from './HierarchicalLabelList';

const ObjectContextMenu = () => {
  const visible = useContextMenuVisible();
  const x = useContextMenuX();
  const y = useContextMenuY();
  const targetObjectId = useContextMenuTargetObjectId();
  const hideContextMenu = useHideContextMenu();
  const enterFocusMode = useEnterFocusMode();
  const objectsList = useObjectsList();
  const selectedObjects = useSelectedObjects();
  const imageObject = useImageObject();
  const updateObject = useUpdateObject();
  const removeObject = useRemoveObject();
  const enterRefinementMode = useEnterRefinementMode();
  const currentTool = useCurrentTool();
  const setCurrentTool = useSetCurrentTool();
  const refinementModeActive = useRefinementModeActive();
  const focusModeActive = useFocusModeActive();
  const focusModeObjectId = useFocusModeObjectId();
  const exitFocusMode = useExitFocusMode();

  // Use the same zoom hook as refinement mode
  const { zoomToObject } = useZoomToObject({
    marginPct: 0.2,
    maxZoom: 4,
    minZoom: 1,
  });
  const enterEditMode = useEnterEditMode();
  const startLineEdit = useStartLineEdit();
  const selectObject = useSelectObject();
  const maskId = useCurrentMaskId();
  const { addToast } = useToast();
  const { currentDataset } = useDataset();
  const menuRef = useRef(null);

  const [adjustedPosition, setAdjustedPosition] = useState({ x, y });
  const [isMerging, setIsMerging] = useState(false);

  // Get all selected objects (targets for batch operations)
  const targetObjects = React.useMemo(() => {
    return objectsList.filter(obj => selectedObjects.includes(obj.id));
  }, [objectsList, selectedObjects]);

  const isMultiSelect = targetObjects.length > 1;

  // Check if the single target object is unlabelled (focus mode requires a label)
  const targetObjectIsUnlabelled = React.useMemo(() => {
    if (isMultiSelect) return false;
    const target = objectsList.find(obj => obj.id === targetObjectId);
    return target ? !hasValidLabel(target.label) : false;
  }, [isMultiSelect, objectsList, targetObjectId]);

  // Use labels hierarchy hook
  const { labelMap, labelsLoading } = useLabelsHierarchy(visible, currentDataset);

  // Restrict the selectable labels to the current hierarchy level: root labels
  // when annotating at the top level, or the children of the parent contour's
  // label when annotating inside another contour.
  const flatLabels = React.useMemo(() => Array.from(labelMap.values()), [labelMap]);

  const primaryTarget = React.useMemo(
    () => objectsList.find((obj) => obj.id === targetObjectId) || targetObjects[0] || null,
    [objectsList, targetObjectId, targetObjects]
  );

  const parentLabelId = React.useMemo(
    () =>
      resolveParentLabelId(primaryTarget, objectsList, {
        active: focusModeActive,
        objectId: focusModeObjectId,
      }),
    [primaryTarget, objectsList, focusModeActive, focusModeObjectId]
  );

  const parentLabelName = React.useMemo(() => {
    if (parentLabelId === null || parentLabelId === undefined) return null;
    const parent =
      labelMap.get(parentLabelId) ||
      labelMap.get(Number(parentLabelId)) ||
      labelMap.get(String(parentLabelId));
    return parent?.name ?? null;
  }, [labelMap, parentLabelId]);

  // Strip nested children so only the single current level is rendered.
  const currentLevelLabels = React.useMemo(
    () =>
      getChildLabels(flatLabels, parentLabelId).map((label) => ({
        id: label.id,
        name: label.name,
        parent_id: label.parent_id,
      })),
    [flatLabels, parentLabelId]
  );

  const suggestSimilar = useSuggestSimilar();

  // Adjust position to keep menu within container bounds and place it intuitively next to the object
  useEffect(() => {
    if (!visible || !menuRef.current) return;

    const menu = menuRef.current;
    const menuRect = menu.getBoundingClientRect();

    // Get the container bounds (the canvas container)
    const container = menu.parentElement;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const containerWidth = containerRect.width;
    const containerHeight = containerRect.height;

    const offset = 10; // Small offset from cursor
    let adjustedX = x + offset;
    let adjustedY = y + offset;

    // Horizontal positioning: prefer right, flip to left if needed
    if (adjustedX + menuRect.width > containerWidth - 10) {
      adjustedX = x - menuRect.width - offset;
    }

    // Keep within horizontal bounds
    adjustedX = Math.max(10, Math.min(adjustedX, containerWidth - menuRect.width - 10));

    // Vertical positioning: prefer below, flip above if needed
    if (adjustedY + menuRect.height > containerHeight - 10) {
      adjustedY = y - menuRect.height - offset;
    }

    // Keep within vertical bounds
    adjustedY = Math.max(10, Math.min(adjustedY, containerHeight - menuRect.height - 10));

    setAdjustedPosition({ x: adjustedX, y: adjustedY });
  }, [visible, x, y]);

  // Close menu on outside click
  useEffect(() => {
    if (!visible) return;

    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        hideContextMenu();
      }
    };

    // Add small delay to prevent immediate close on the same click that opened it
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handleClickOutside);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible]);

  // Use shared label selection hook
  const handleLabelSelectBase = useLabelSelection(
    updateObject,
    () => {
      // onSuccess: switch tool and hide menu. Manual drawing stays put —
      // labelling the outline you just drew should not end the drawing session.
      if (currentTool !== 'manual_drawing') setCurrentTool('ai_annotation');
      hideContextMenu();
    },
    (error) => {
      // onError: show error and hide menu
      alert(`Failed to apply label: ${error.message || 'Unknown error'}`);
      hideContextMenu();
    }
  );

  const handleLabelSelect = async (label) => {
    if (targetObjects.length === 0) {
      hideContextMenu();
      return;
    }

    // Apply label to all selected objects
    try {
      for (const targetObject of targetObjects) {
        await handleLabelSelectBase(targetObject, label);
      }

      // Success: menu is already hidden by handleLabelSelectBase
    } catch (error) {
      // Error is already handled by handleLabelSelectBase
      hideContextMenu();
    }
  };

  const handleFocusMode = async () => {
    // Disable focus mode when in refinement mode or multiple objects selected
    if (refinementModeActive || isMultiSelect) {
      return;
    }

    if (!targetObjectId || !imageObject) return;

    // Find the target object
    const targetObject = objectsList.find(obj => obj.id === targetObjectId);
    if (!targetObject || !targetObject.x || !targetObject.y || targetObject.x.length === 0) {
      hideContextMenu();
      return;
    }

    // Block focus mode for unlabelled objects
    if (!hasValidLabel(targetObject.label)) {
      hideContextMenu();
      return;
    }

    // Get contour ID for WebSocket message
    const contourId = getContourId(targetObject);

    // Create mask from x,y arrays for focus mode boundary checking
    const points = targetObject.x.map((x, i) => [
      x * imageObject.width,
      targetObject.y[i] * imageObject.height
    ]);
    const objectMask = { points: points };

    try {
      // Send focus message to backend via WebSocket
      await annotationSession.focusImage(contourId);

      // Enter focus mode in the store (without zoom - zoom is handled externally)
      enterFocusMode(targetObjectId, objectMask);
    } catch (error) {
      console.error('Failed to enter focus mode:', error);
      hideContextMenu();
      return;
    }

    // Get the container element (the canvas container)
    const container = menuRef.current?.parentElement;
    if (!container) {
      hideContextMenu();
      return;
    }

    const containerWidth = container.offsetWidth || 800;
    const containerHeight = container.offsetHeight || 600;

    // Calculate image dimensions (actual image size)
    const imageDimensions = {
      width: imageObject.width || 800,
      height: imageObject.height || 600
    };

    // Calculate container dimensions
    const containerDimensions = {
      width: containerWidth,
      height: containerHeight
    };

    // Calculate rendered image dimensions (how the image is displayed in the container)
    const renderedImageDimensions = calculateRenderedImageDimensions(
      imageObject,
      containerWidth,
      containerHeight
    );

    // Use the same zoom logic as refinement mode
    zoomToObject(
      targetObject,
      imageDimensions,
      containerDimensions,
      renderedImageDimensions,
      { animateMs: 300, immediate: false }
    );

    hideContextMenu();
  };

  const handleReject = async () => {
    if (targetObjects.length === 0) {
      hideContextMenu();
      return;
    }

    try {
      // Delete all selected objects
      for (const targetObject of targetObjects) {
        await deleteObject(targetObject, removeObject);
      }

      // Switch to AI assisted annotation tool, unless the user is drawing by
      // hand — deleting an object is no reason to put their tool back.
      if (currentTool !== 'manual_drawing') setCurrentTool('ai_annotation');

      hideContextMenu();
    } catch (error) {
      alert(`Failed to reject object(s): ${error.message || 'Unknown error'}`);
      hideContextMenu();
    }
  };

  // Use shared refinement mode hook
  const enterRefinementModeForObject = useRefinementMode({
    enterRefinementMode,
    setCurrentTool,
    exitFocusMode,
    focusModeActive,
    imageObject,
    containerRef: menuRef,
    zoomOptions: {
      marginPct: 0.25,
      maxZoom: 4,
      minZoom: 1,
    },
  });

  const handleRefine = async () => {
    // Disable refinement mode for multiple objects
    if (isMultiSelect) {
      return;
    }

    if (!targetObjectId) return;

    // Find the target object to get its contour_id
    const targetObject = objectsList.find(obj => obj.id === targetObjectId);
    if (!targetObject) {
      hideContextMenu();
      return;
    }

    try {
      await enterRefinementModeForObject(targetObject);

      // Also enter edit mode so control points are immediately available
      if (targetObject.x && targetObject.y && targetObject.x.length > 0 && targetObject.contour_id != null) {
        enterEditMode(targetObject.id, targetObject.contour_id, targetObject.x, targetObject.y);
      }

      hideContextMenu();
    } catch (error) {
      alert(`Failed to enter refinement mode: ${error.message || 'Unknown error'}`);
      hideContextMenu();
    }
  };

  const handleSuggestSimilar = async () => {
    hideContextMenu();
    await suggestSimilar.run();
  };

  const handleLineEditContour = (lineMode = 'reshape') => {
    if (isMultiSelect) return;
    const targetObject = objectsList.find(obj => obj.id === targetObjectId);
    if (!targetObject || targetObject.contour_id == null ||
        !targetObject.x || targetObject.x.length === 0) {
      hideContextMenu();
      return;
    }

    if (focusModeActive) {
      if (annotationSession.isReady()) {
        annotationSession.unfocusImage().catch(() => {});
      }
      exitFocusMode();
    }

    selectObject(targetObject.id);
    setCurrentTool('selection');
    startLineEdit(targetObject.id, targetObject.contour_id, targetObject.x, targetObject.y, lineMode);

    // Frame the instance so there is room to draw the line.
    if (imageObject && targetObject.x.length > 0) {
      const container = menuRef.current?.parentElement;
      if (container?.offsetWidth && container?.offsetHeight) {
        const rendered = calculateRenderedImageDimensions(imageObject, container.offsetWidth, container.offsetHeight);
        zoomToObject(
          targetObject,
          { width: imageObject.width, height: imageObject.height },
          { width: container.offsetWidth, height: container.offsetHeight },
          rendered,
          { animateMs: 300, immediate: false }
        );
      }
    }

    hideContextMenu();
  };

  /**
   * Merge the selection into one object (#44).
   *
   * Only touching or overlapping outlines can be merged: their union is a single
   * ring, which is what a contour already is. A disjoint selection is refused by
   * `mergeObjects` rather than quietly producing a shape that spans the gap.
   */
  const handleMergeObjects = async () => {
    if (!isMultiSelect || isMerging) return;

    setIsMerging(true);
    hideContextMenu();
    try {
      const result = await mergeObjects({
        objects: targetObjects,
        objectsList,
        imageObject,
        maskId,
        updateObject,
      });
      addToast({ type: result.success ? 'success' : 'error', message: result.message });
    } catch (error) {
      addToast({ type: 'error', message: error.message || 'Could not merge those objects.' });
    } finally {
      setIsMerging(false);
    }
  };

  const handleEditContour = () => {
    // Disable edit mode for multiple objects
    if (isMultiSelect) {
      return;
    }

    if (!targetObjectId) {
      hideContextMenu();
      return;
    }

    // Find the target object
    const targetObject = objectsList.find(obj => obj.id === targetObjectId);
    if (!targetObject) {
      hideContextMenu();
      return;
    }

    // Ensure object has valid coordinates
    if (!targetObject.x || !targetObject.y || targetObject.x.length === 0 || targetObject.y.length === 0) {
      alert('Cannot edit object: missing or invalid coordinates');
      hideContextMenu();
      return;
    }

    // Ensure contour_id exists for backend communication
    if (!targetObject.contour_id && targetObject.contour_id !== 0) {
      alert('Cannot edit object: missing contour_id');
      hideContextMenu();
      return;
    }

    // Exit focus mode if active
    if (focusModeActive) {
      if (annotationSession.isReady()) {
        annotationSession.unfocusImage().catch(err =>
          console.error('Failed to send unfocus message:', err)
        );
      }
      exitFocusMode();
    }

    // Enter edit mode
    enterEditMode(targetObject.id, targetObject.contour_id, targetObject.x, targetObject.y);

    // Zoom into the contour (like refinement mode does)
    if (imageObject && targetObject.x && targetObject.y && targetObject.x.length > 0) {
      const container = menuRef.current?.parentElement;
      if (container) {
        const containerWidth = container.offsetWidth;
        const containerHeight = container.offsetHeight;
        if (containerWidth && containerHeight) {
          const renderedImageDimensions = calculateRenderedImageDimensions(imageObject, containerWidth, containerHeight);
          zoomToObject(
            targetObject,
            { width: imageObject.width, height: imageObject.height },
            { width: containerWidth, height: containerHeight },
            renderedImageDimensions,
            { animateMs: 300, immediate: false }
          );
        }
      }
    }

    hideContextMenu();
  };

  if (!visible) return null;

  return (
    <div
      ref={menuRef}
      className="absolute z-50 w-[216px] p-[5px] rounded-9 bg-p2 border border-ln2 shadow-ctx animate-dcPop"
      style={{
        left: `${adjustedPosition.x}px`,
        top: `${adjustedPosition.y}px`,
      }}
    >
      {/* Header showing selection count */}
      {isMultiSelect && (
        <div className="px-[8px] py-[6px] mb-[3px] rounded-6 bg-acS text-meta font-bold text-ac">
          {targetObjects.length} objects selected
        </div>
      )}

      {/* Reject Object Option */}
      <ContextMenuItem
        onClick={handleReject}
        tone="danger"
        label={isMultiSelect ? `Reject ${targetObjects.length} objects` : "Reject object"}
        icon={
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        }
      />

      {/* Focus Mode Option - Disabled in refinement mode, multi-select, or for unlabelled objects */}
      <ContextMenuItem
        onClick={handleFocusMode}
        disabled={refinementModeActive || isMultiSelect || targetObjectIsUnlabelled}
        title={
          isMultiSelect
            ? 'Focus mode is disabled for multiple selections'
            : refinementModeActive
              ? 'Focus mode is disabled during refinement'
            : targetObjectIsUnlabelled
              ? 'Assign a label to this object before entering Focus Mode'
              : 'Enter focus mode'
        }
        label="Focus Mode"
        icon={
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
          </svg>
        }
      />

      {/* Refine Option - Disabled for multi-select */}
      <ContextMenuItem
        onClick={handleRefine}
        disabled={isMultiSelect}
        title={isMultiSelect ? 'Refinement mode is disabled for multiple selections' : 'Refine object'}
        label="Refine Object"
        icon={
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
        }
      />

      {/* Reshape by Line Option - draw a line that is merged into the boundary */}
      <ContextMenuItem
        onClick={() => handleLineEditContour('reshape')}
        disabled={isMultiSelect}
        title={isMultiSelect ? 'Reshape is disabled for multiple selections' : 'Draw a line across the boundary to cut off or add a region'}
        label="Reshape by Line"
        icon={
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 20l6-6m0 0l4-4 6-6M10 14l4 4m-4-4l-2-2" />
          </svg>
        }
      />

      {/* Split Option - draw a line across the object to cut it in two */}
      <ContextMenuItem
        onClick={() => handleLineEditContour('split')}
        disabled={isMultiSelect}
        title={isMultiSelect ? 'Split works on one object at a time' : 'Draw a line across this object to cut it into two objects'}
        label="Split Object"
        icon={
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v18M6 7l-3 5 3 5m12-10l3 5-3 5" />
          </svg>
        }
      />

      {/* Merge Option - union of a touching or overlapping selection */}
      <ContextMenuItem
        onClick={handleMergeObjects}
        disabled={!isMultiSelect || isMerging}
        title={
          !isMultiSelect
            ? 'Select two or more touching objects to merge them'
            : 'Merge the selected objects into one'
        }
        label={isMerging ? 'Merging…' : (isMultiSelect ? `Merge ${targetObjects.length} Objects` : 'Merge Objects')}
        icon={
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h5a3 3 0 013 3v6a3 3 0 003 3h5m0-12h-5a3 3 0 00-3 3" />
          </svg>
        }
      />

      {/* Edit Contour Option - Disabled for multi-select */}
      <ContextMenuItem
        onClick={handleEditContour}
        disabled={isMultiSelect}
        title={isMultiSelect ? 'Edit contour is disabled for multiple selections' : 'Drag the existing outline’s control points'}
        label="Edit Contour"
        icon={
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
          </svg>
        }
      />

      {/* Suggest Similar Instances Option */}
      <ContextMenuItem
        onClick={handleSuggestSimilar}
        disabled={!suggestSimilar.eligible}
        title={
          suggestSimilar.reason ||
          (isMultiSelect
            ? `Use ${targetObjects.length} objects as seeds for suggestion segmentation`
            : 'Find similar instances using suggestion segmentation')
        }
        label={suggestSimilar.isRunning ? 'Finding similar...' : 'Suggest Similar Instances'}
        icon={
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
        }
      />

      {/* Label section header */}
      <div className="mt-[4px] pt-[6px] px-[8px] border-t border-ln">
        <div className="text-sect font-bold tracking-[.08em] uppercase text-t3">
          {isMultiSelect ? `Label ${targetObjects.length} objects` : 'Label'}
        </div>
        <div className="mb-[4px] text-meta text-t3">
          {parentLabelName ? `Sub-labels of ${parentLabelName}` : 'Root level'}
        </div>
      </div>

      {/* Current-level label list (root labels, or children of the parent contour's label) */}
      <HierarchicalLabelList
        labelHierarchy={currentLevelLabels}
        labelsLoading={labelsLoading}
        onLabelSelect={handleLabelSelect}
        emptyMessage={
          parentLabelName
            ? `No sub-labels under "${parentLabelName}"`
            : 'No labels available'
        }
      />
    </div>
  );
};

export default ObjectContextMenu;
