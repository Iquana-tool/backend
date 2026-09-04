import { getObjectColor } from '../utils/objectColors';
import { hasValidLabel } from '../utils/labelValidation';

/**
 * Objects slice - manages annotation objects, selection, visibility, and colors
 */
export const createObjectsSlice = (set) => ({
  // Dataset labels cache – populated once per dataset, shared by all components
  setDatasetLabels: (labelsArray, labelsMap) => set((state) => {
    state.objects.datasetLabels = labelsArray;
    state.objects.datasetLabelsMap = labelsMap;
    if (state.workspace?.activeLabelId != null) {
      const exists = Array.isArray(labelsArray) && labelsArray.some((l) => Number(l.id) === Number(state.workspace.activeLabelId));
      if (!exists) {
        state.workspace.activeLabelId = null;
      }
    }
  }),

  clearDatasetLabels: () => set((state) => {
    state.objects.datasetLabels = [];
    state.objects.datasetLabelsMap = null;
    if (state.workspace?.activeLabelId != null) {
      state.workspace.activeLabelId = null;
    }
  }),

  addObject: (object) => set((state) => {
    // Use contour_id from backend if available, otherwise generate a timestamp ID
    // Normalize ID format to ensure consistent color calculation (prefer number for numeric IDs)
    const rawId = object.contour_id || object.mask?.id || Date.now();
    const objectId = typeof rawId === 'string' && !isNaN(rawId)
      ? Number(rawId)
      : (typeof rawId === 'number' ? rawId : rawId);

    // Normalize contour_id as well for consistency
    const rawContourId = object.contour_id || object.mask?.id || null;
    const contourId = rawContourId && typeof rawContourId === 'string' && !isNaN(rawContourId)
      ? Number(rawContourId)
      : (rawContourId && typeof rawContourId === 'number' ? rawContourId : rawContourId);

    // Normalize parent_id to number to match the normalized id (Map keys must be same type)
    const rawParentId = object.parent_id ?? null;
    const normalizedParentId = rawParentId != null
      ? (typeof rawParentId === 'string' && !isNaN(rawParentId) ? Number(rawParentId) : rawParentId)
      : null;

    // Check if object with this contour_id already exists (prevent duplicates)
    if (contourId !== null && state.objects.list.some(obj => obj.contour_id === contourId)) {
      return;
    }

    const newObject = {
      ...object,
      id: objectId,
      contour_id: contourId,
      parent_id: normalizedParentId,
      pixelCount: object.pixelCount || 0,
      label: object.label || 'Object',
      path: object.path || object.mask?.path || null,
      mask: object.mask,
      x: object.x || [],
      y: object.y || [],
      color: getObjectColor({ labelId: object.labelId, label: object.label }, objectId),
    };
    state.objects.list.push(newObject);
    state.objects.colors[newObject.id] = newObject.color;
  }),

  // Replace all objects from a backend ContourHierarchy
  // labelsMap: Optional Map mapping label_id to label name (always Map or null in production)
  setObjectsFromHierarchy: (hierarchy, labelsMap = null) => set((state) => {
    const list = [];
    const colors = {};

    // labelsMap is always a Map or null (constructed in AnnotationPageV2)
    const labelIdToNameMap = labelsMap;
    const payload = hierarchy?.contours || hierarchy;

    if (payload && Array.isArray(payload.root_contours)) {
      // Queue entries are { contour, parentId } so we preserve hierarchy when backend
      // returns nested children without parent_id on each node (e.g. after page refresh)
      const queue = payload.root_contours.map(c => ({ contour: c, parentId: null }));
      let orderCounter = 0;

      while (queue.length > 0) {
        const { contour: c, parentId: queueParentId } = queue.shift();
        const rawId = c.id ?? Date.now() + Math.random();

        // Normalize ID format to ensure consistent color calculation (prefer number for numeric IDs)
        const id = typeof rawId === 'string' && !isNaN(rawId)
          ? Number(rawId)
          : (typeof rawId === 'number' ? rawId : rawId);

        // Normalize contour_id as well
        const rawContourId = c.id ?? null;
        const contourId = rawContourId && typeof rawContourId === 'string' && !isNaN(rawContourId)
          ? Number(rawContourId)
          : (rawContourId && typeof rawContourId === 'number' ? rawContourId : rawContourId);

        // Backend returns label_id
        const labelId = c.label_id ?? null;

        // Look up label name from labelsMap using label_id
        let labelName = null;
        if (labelId && labelIdToNameMap) {
          const numericLabelId = Number(labelId);
          labelName = labelIdToNameMap.get(numericLabelId) ?? labelIdToNameMap.get(String(labelId)) ?? null;
        }

        // If object has a valid label from backend, assign it an order
        // We'll use the order they appear in the hierarchy as the assignment order
        let labelAssignmentOrder = undefined;
        if (hasValidLabel(labelName)) {
          orderCounter += 1;
          labelAssignmentOrder = orderCounter;
        }

        // Use backend parent_id if present, else the parent we tracked from nesting (for refresh load)
        // Normalize parent_id to number to match the normalized id (Map keys must be same type)
        const rawParentId = c.parent_id ?? queueParentId ?? null;
        const parentId = rawParentId != null
          ? (typeof rawParentId === 'string' && !isNaN(rawParentId) ? Number(rawParentId) : rawParentId)
          : null;

        const obj = {
          id, // Normalized ID format
          contour_id: contourId, // Normalized contour_id format
          label: labelName, // Use label name if available, otherwise null
          labelId: labelId, // Store label ID for future reference
          labelAssignmentOrder,
          x: c.x || [],
          y: c.y || [],
          path: c.path || null, // SVG path from backend
          added_by: c.added_by || null,
          reviewed_by: c.reviewed_by || [],
          parent_id: parentId,
          quantification: c.quantification || null,
          // Use label-based color if labeled, otherwise use ID-based color for stability
          color: getObjectColor({ labelId: labelId, label: labelName }, id),
        };
        list.push(obj);
        colors[obj.id] = obj.color;
        // Backend guarantees children is always an array (default=[])
        if (c.children?.length > 0) {
          for (const child of c.children) {
            queue.push({ contour: child, parentId: id });
          }
        }
      }
      
      // Update the global counter to continue from where backend objects left off
      state.objects.labelAssignmentCounter = orderCounter;
    } else {
      // Reset counter if no hierarchy (or empty)
      state.objects.labelAssignmentCounter = 0;
    }
    
    state.objects.list = list;
    state.objects.selected = [];
    state.objects.colors = colors;
    // The hierarchy is the answer the spinner was waiting for, whether or not it is empty.
    state.objects.loading = false;
    state.objects.loadError = null;
  }),

  clearObjects: () => set((state) => {
    state.objects.list = [];
    state.objects.selected = [];
    state.objects.colors = {};
    state.objects.labelAssignmentCounter = 0;
  }),

  /**
   * Wipe the canvas for an incoming image and put it into the loading state.
   *
   * One action rather than `clearObjects()` + `setObjectsLoading(true)` so the canvas
   * can never render the gap between them: an emptied list that does not yet know it is
   * waiting looks exactly like an image with no annotations.
   */
  beginObjectsLoad: () => set((state) => {
    state.objects.list = [];
    state.objects.selected = [];
    state.objects.colors = {};
    state.objects.labelAssignmentCounter = 0;
    state.objects.loading = true;
    state.objects.loadError = null;
  }),

  /** Give up on the current load and show `error` in place of the spinner. */
  failObjectsLoad: (error) => set((state) => {
    state.objects.loading = false;
    state.objects.loadError = error || 'Could not load the annotations for this image.';
  }),
  
  removeObject: (id) => set((state) => {
    state.objects.list = state.objects.list.filter(obj => obj.id !== id);
    delete state.objects.colors[id];
    state.objects.selected = state.objects.selected.filter(objId => objId !== id);
    
    // Clear focus mode if the focused object is removed
    if (state.focusMode.objectId === id) {
      state.focusMode.active = false;
      state.focusMode.objectId = null;
      state.focusMode.objectMask = null;
    }
  }),
  
  updateObject: (id, updates) => set((state) => {
    const objectIndex = state.objects.list.findIndex(obj => obj.id === id);
    if (objectIndex !== -1) {
      const existingObject = state.objects.list[objectIndex];
      
      // Check if a label is being assigned (object didn't have a valid label before)
      const existingLabelStr = String(existingObject.label || '').trim();
      const hadValidLabel = existingObject.label && 
                            existingLabelStr !== 'Object' && 
                            existingLabelStr !== '' &&
                            !/^\d+$/.test(existingLabelStr);
      
      const updateLabelStr = String(updates.label || '').trim();
      const hasValidLabelUpdate = updates.label && 
                                  updateLabelStr !== 'Object' && 
                                  updateLabelStr !== '' &&
                                  !/^\d+$/.test(updateLabelStr);
      const isNewLabelAssignment = !hadValidLabel && hasValidLabelUpdate;
      
      // If this is a new label assignment, increment the counter and assign the order
      if (isNewLabelAssignment) {
        state.objects.labelAssignmentCounter += 1;
        updates.labelAssignmentOrder = state.objects.labelAssignmentCounter;
      }
      
      // Merge updates into the existing object
      const updatedObject = {
        ...existingObject,
        ...updates
      };
      
      // If label or labelId is being updated, recalculate color based on label
      if (updates.labelId !== undefined || updates.label !== undefined) {
        updatedObject.color = getObjectColor(
          { 
            labelId: updatedObject.labelId, 
            label: updatedObject.label 
          }, 
          updatedObject.id
        );
        // Also update the colors map
        state.objects.colors[updatedObject.id] = updatedObject.color;
      }
      
      state.objects.list[objectIndex] = updatedObject;
    }
  }),
  
  selectObject: (id) => set((state) => {
    if (!state.objects.selected.includes(id)) {
      state.objects.selected.push(id);
    }
  }),
  
  deselectObject: (id) => set((state) => {
    state.objects.selected = state.objects.selected.filter(objId => objId !== id);
  }),
  
  clearSelection: () => set((state) => {
    state.objects.selected = [];
  }),
  
  toggleVisibility: (labelId) => set((state) => {
    // Handle special visibility modes
    if (labelId === 'showRootLabels') {
      state.objects.visibility.showRootLabels = !state.objects.visibility.showRootLabels;
      return;
    }
    
    // Handle label ID toggling (labelId can be a number or string representation)
    const labelIdKey = typeof labelId === 'number' ? labelId : String(labelId);
    
    // If label visibility doesn't exist, initialize it to true (default visible)
    if (!(labelIdKey in state.objects.visibility.labels)) {
      state.objects.visibility.labels[labelIdKey] = true;
    }
    
    // Toggle the visibility
    state.objects.visibility.labels[labelIdKey] = !state.objects.visibility.labels[labelIdKey];
  }),
  
  // Initialize label visibility from actual labels
  initializeLabelVisibility: (labels) => set((state) => {
    if (!labels || !Array.isArray(labels)) return;
    
    // Build array of root-level label IDs (store both number and string versions)
    const rootLabelIds = [];
    
    // Initialize all labels as visible by default and track root labels
    labels.forEach(label => {
      if (label && label.id !== undefined) {
        const labelIdKey = String(label.id);
        // Only initialize if not already set (preserve user's manual changes)
        if (!(labelIdKey in state.objects.visibility.labels)) {
          state.objects.visibility.labels[labelIdKey] = true;
        }
        
        // Track root-level labels (labels with no parent_id)
        if (!label.parent_id || label.parent_id === null) {
          rootLabelIds.push(label.id);
          rootLabelIds.push(String(label.id)); // Also add string version for lookup
        }
      }
    });
    
    // Update root label IDs array
    state.objects.visibility.rootLabelIds = rootLabelIds;
  }),
  
  setVisibilityMode: (mode) => set((state) => {
    state.objects.visibility.showAll = false;
    state.objects.visibility.rootLevelOnly = false;
    state.objects.visibility.selectedLevelOnly = false;
    state.objects.visibility[mode] = true;
  }),
});

