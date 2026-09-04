import React, { useState, useEffect, useRef, useMemo } from "react";
import { Plus, Trash2, ChevronsDownUp, ChevronsUpDown, CornerUpRight, AlertTriangle } from "lucide-react";
import * as api from "../../../api";
import {
  buildLabelHierarchy,
  extractLabelsFromResponse,
  findLabelInHierarchy,
  getDescendants,
  getLabelPath
} from "../../../utils/labelHierarchy";
import { getLabelColor } from "../../../utils/labelColors";
import { useLabelHierarchy } from "../../../hooks/useLabelHierarchy";
import LabelHierarchyRenderer from "../shared/LabelHierarchyRenderer";

/** Deepest nesting level in the tree, counted in levels (a flat list is 1). */
const hierarchyDepth = (nodes) =>
  (nodes || []).reduce(
    (deepest, node) => Math.max(deepest, 1 + hierarchyDepth(node.children)),
    0
  );

const EditableLabels = ({ dataset, labels, onLabelsUpdated }) => {
  const {
    labelHierarchy,
    expandedLabels,
    setLabelHierarchy,
    toggleExpanded,
    expandLabel,
    expandAll,
    collapseAll,
  } = useLabelHierarchy([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Modal states
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [labelToDelete, setLabelToDelete] = useState(null);
  const [labelToEdit, setLabelToEdit] = useState(null);

  // Add label form
  const [newLabelName, setNewLabelName] = useState('');
  const [targetParentLabel, setTargetParentLabel] = useState(null);

  // Edit label states
  const [editLabelName, setEditLabelName] = useState('');

  // Move label states. `moveBlocked` holds the backend's refusal — the move is legal
  // but would strand annotated objects — until the user accepts detaching them.
  const [showMoveModal, setShowMoveModal] = useState(false);
  const [labelToMove, setLabelToMove] = useState(null);
  const [moveTarget, setMoveTarget] = useState('root');
  const [moveBlocked, setMoveBlocked] = useState(null);

  // Drag-to-reparent. `summary` is fetched once per drag so every row can be priced
  // as the pointer passes over it, instead of asking the server per hover.
  const [drag, setDrag] = useState({ id: null, overId: null, overRoot: false });
  const [dragSummary, setDragSummary] = useState(null);

  // Update local labels when props change
  useEffect(() => {
    const hierarchy = buildLabelHierarchy(labels);
    setLabelHierarchy(hierarchy);
  }, [labels]);

  // Open the tree the first time it loads. A hierarchy that arrives fully
  // collapsed is a hierarchy the curator never sees — the user study found
  // people were unaware their label space was nested at all. After that first
  // expansion the user's own collapse state is left alone.
  const didInitialExpand = useRef(false);
  useEffect(() => {
    if (didInitialExpand.current || labelHierarchy.length === 0) return;
    didInitialExpand.current = true;
    expandAll();
  }, [labelHierarchy, expandAll]);

  const totalLabels = Array.isArray(labels) ? labels.length : 0;
  const depth = useMemo(() => hierarchyDepth(labelHierarchy), [labelHierarchy]);
  const hasNesting = depth > 1;

  // Breadcrumb for the add dialog, so "where am I adding this?" is answerable
  // from the dialog itself rather than from the row that opened it.
  const parentPath = useMemo(
    () =>
      targetParentLabel
        ? getLabelPath(labelHierarchy, targetParentLabel.id).map((l) => l.name)
        : [],
    [labelHierarchy, targetParentLabel]
  );

  // Refresh labels from backend
  const refreshLabels = async () => {
    try {
      const labelsData = await api.fetchLabels(dataset.id);
      const labelsArray = extractLabelsFromResponse(labelsData);

      const hierarchy = buildLabelHierarchy(labelsArray);
      setLabelHierarchy(hierarchy);
      if (onLabelsUpdated) {
        onLabelsUpdated(labelsArray);
      }
    } catch (err) {
      console.error('Error refreshing labels:', err);
      setError('Failed to refresh labels');
    }
  };

  // Handle adding a new label
  const handleAddLabel = async () => {
    if (!newLabelName.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const labelData = {
        name: newLabelName.trim(),
        parent_id: targetParentLabel ? targetParentLabel.id : null
      };

      const result = await api.createLabel(labelData, dataset.id);

      if (result.success) {
        await refreshLabels();
        // A part added to a collapsed parent would otherwise land out of sight.
        if (targetParentLabel) expandLabel(targetParentLabel.id);
        setShowAddModal(false);
        setNewLabelName('');
        setTargetParentLabel(null);
      } else {
        setError('Failed to create label');
      }
    } catch (err) {
      console.error('Error creating label:', err);
      setError('Failed to create label');
    } finally {
      setLoading(false);
    }
  };

  // Handle deleting a label
  const handleDeleteLabel = async () => {
    if (!labelToDelete) return;
    
    setLoading(true);
    setError(null);
    
    try {
      await api.deleteLabel(labelToDelete.id, dataset.id);
      await refreshLabels();
      setShowDeleteModal(false);
      setLabelToDelete(null);
    } catch (err) {
      console.error('Error deleting label:', err);
      setError('Failed to delete label. It may be in use by existing annotations.');
    } finally {
      setLoading(false);
    }
  };

  // Handle editing a label name
  const handleEditLabel = async () => {
    if (!labelToEdit || !editLabelName.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      await api.updateLabel(labelToEdit.id, { name: editLabelName.trim() }, dataset.id);
      await refreshLabels();
      setShowEditModal(false);
      setLabelToEdit(null);
      setEditLabelName('');
    } catch (err) {
      console.error('Error updating label:', err);
      setError('Failed to update label');
    } finally {
      setLoading(false);
    }
  };


  // Where "a label" may be moved: every label except itself and its own parts, since a
  // label cannot become a part of something that is already a part of it.
  const moveCandidates = useMemo(() => {
    if (!labelToMove) return [];
    const excluded = new Set([
      labelToMove.id,
      ...getDescendants(labelToMove).map((descendant) => descendant.id),
    ]);
    const candidates = [];
    const walk = (nodes, path) => {
      (nodes || []).forEach((node) => {
        if (excluded.has(node.id)) return;
        const nodePath = [...path, node.name];
        candidates.push({ id: node.id, name: node.name, path: nodePath, depth: nodePath.length - 1 });
        if (node.children && node.children.length > 0) walk(node.children, nodePath);
      });
    };
    walk(labelHierarchy, []);
    return candidates;
  }, [labelHierarchy, labelToMove]);

  const moveTargetLabel =
    moveTarget === 'root' ? null : moveCandidates.find((c) => String(c.id) === String(moveTarget));
  const moveIsNoOp =
    !!labelToMove &&
    (moveTarget === 'root'
      ? !labelToMove.parent_id
      : String(labelToMove.parent_id) === String(moveTarget));

  // The one move path, shared by the dialog and by dropping a label onto another.
  // A refused move is not an error: it opens the confirmation carrying the server's
  // own count, which is authoritative even when the drag's estimate has gone stale.
  const performMove = async (label, newParentId, detachAffected) => {
    if (!label) return 'error';

    setLoading(true);
    setError(null);

    try {
      const result = await api.moveLabel(label.id, newParentId, { detachAffected });

      if (result?.blocked) {
        setLabelToMove(label);
        setMoveTarget(newParentId == null ? 'root' : String(newParentId));
        setMoveBlocked(result);
        setShowMoveModal(true);
        return 'blocked';
      }

      await refreshLabels();
      if (newParentId) expandLabel(newParentId);
      setShowMoveModal(false);
      setLabelToMove(null);
      setMoveBlocked(null);
      return 'ok';
    } catch (err) {
      console.error('Error moving label:', err);
      setError(err?.message || 'Failed to move label');
      return 'error';
    } finally {
      setLoading(false);
    }
  };

  // Handle moving a label from the dialog. Called twice for a blocked move: once to
  // learn what it would break, and again with `detachAffected` once the user accepts.
  const handleMoveLabel = (detachAffected = false) =>
    performMove(labelToMove, moveTarget === 'root' ? null : Number(moveTarget), detachAffected);

  // --- drag to reparent -----------------------------------------------------
  //
  // Native HTML5 drag, matching the object tree in the annotation workspace. Dropping
  // a label onto another makes it a part of that label; dropping it on the strip below
  // the tree makes it a whole object again.

  const endDrag = () => {
    setDrag({ id: null, overId: null, overRoot: false });
    setDragSummary(null);
  };

  // A label cannot become a part of itself or of one of its own parts.
  const forbiddenTargets = useMemo(() => {
    if (drag.id === null) return new Set();
    const dragged = findLabelInHierarchy(labelHierarchy, drag.id);
    return new Set([
      drag.id,
      ...(dragged ? getDescendants(dragged).map((descendant) => descendant.id) : []),
    ]);
  }, [drag.id, labelHierarchy]);

  // How many annotated objects dropping on this target would strand. Mirrors the
  // server's arithmetic; the move endpoint still re-derives it before writing.
  const strandedBy = (targetId) => {
    if (!dragSummary) return 0;
    const byContainer = dragSummary.by_container_label || {};
    const alreadyInside =
      targetId == null ? 0 : byContainer[String(targetId)] ?? byContainer[targetId] ?? 0;
    return Math.max(0, (dragSummary.nested_total || 0) - alreadyInside);
  };

  const handleDragStart = (label) => async (event) => {
    event.dataTransfer.effectAllowed = 'move';
    // Firefox refuses to start a drag unless data is set.
    event.dataTransfer.setData('text/plain', String(label.id));
    setDrag({ id: label.id, overId: null, overRoot: false });
    setDragSummary(null);

    try {
      const result = await api.fetchLabelNestingSummary(label.id);
      setDragSummary(result?.summary ?? null);
    } catch (err) {
      // Without the summary the drag simply shows no count; the drop still asks the
      // server, which refuses anything that would strand objects.
      console.error('Could not price this move:', err);
    }
  };

  const handleDragOverLabel = (label) => (event) => {
    // Not calling preventDefault leaves the browser showing "no drop" on its own.
    if (drag.id === null || forbiddenTargets.has(label.id)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    if (drag.overId !== label.id || drag.overRoot) {
      setDrag((current) => ({ ...current, overId: label.id, overRoot: false }));
    }
  };

  const handleDropOnLabel = (label) => async (event) => {
    event.preventDefault();
    const dragged = findLabelInHierarchy(labelHierarchy, drag.id);
    const illegal = forbiddenTargets.has(label.id);
    endDrag();

    if (!dragged || illegal) return;
    if (String(dragged.parent_id ?? '') === String(label.id)) return; // already there
    await performMove(dragged, label.id, false);
  };

  const handleDragOverRoot = (event) => {
    if (drag.id === null) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    if (!drag.overRoot) {
      setDrag((current) => ({ ...current, overId: null, overRoot: true }));
    }
  };

  const handleDropOnRoot = async (event) => {
    event.preventDefault();
    const dragged = findLabelInHierarchy(labelHierarchy, drag.id);
    endDrag();

    if (!dragged || !dragged.parent_id) return; // already a whole object
    await performMove(dragged, null, false);
  };

  const draggedLabel = drag.id === null ? null : findLabelInHierarchy(labelHierarchy, drag.id);

  // Per-row drag wiring and the warning a costly drop target carries.
  const rowPropsFor = (label) => {
    const isDragged = drag.id === label.id;
    const isForbidden = drag.id !== null && forbiddenTargets.has(label.id) && !isDragged;
    const isOver = drag.overId === label.id;
    const cost = isOver ? strandedBy(label.id) : 0;

    const classes = ['cursor-grab'];
    if (isDragged) classes.push('opacity-45');
    if (isForbidden) classes.push('opacity-40');
    if (isOver) {
      classes.push(
        cost > 0
          ? 'ring-2 ring-inset ring-warn bg-warnBg'
          : 'ring-2 ring-inset ring-accent'
      );
    }

    return {
      draggable: true,
      onDragStart: handleDragStart(label),
      onDragOver: handleDragOverLabel(label),
      onDrop: handleDropOnLabel(label),
      onDragEnd: endDrag,
      title: `Drag "${label.name}" onto another label to make it a part of it`,
      className: classes.join(' '),
      badge:
        isOver && cost > 0 ? (
          // Pulsed rather than blinking, and only where motion is welcome: the count
          // is the message, and it has to stay readable while the pointer is moving.
          <span className="shrink-0 rounded-full border border-warnLn bg-warnBg px-2 py-0.5 text-[11px] font-medium text-warn motion-safe:animate-pulse">
            would detach {cost} object{cost !== 1 ? 's' : ''}
          </span>
        ) : null,
    };
  };

  // Open move modal
  const openMoveModal = (label) => {
    setLabelToMove(label);
    setMoveTarget(label.parent_id ? String(label.parent_id) : 'root');
    setMoveBlocked(null);
    setShowMoveModal(true);
  };

  const closeMoveModal = () => {
    setShowMoveModal(false);
    setLabelToMove(null);
    setMoveBlocked(null);
  };

  // Open add modal
  const openAddModal = (parentLabel = null) => {
    setTargetParentLabel(parentLabel);
    setNewLabelName('');
    setShowAddModal(true);
  };

  // Open delete modal
  const openDeleteModal = (label) => {
    setLabelToDelete(label);
    setShowDeleteModal(true);
  };

  // Open edit modal
  const openEditModal = (label) => {
    setLabelToEdit(label);
    setEditLabelName(label.name);
    setShowEditModal(true);
  };

  // Label colour comes from the shared palette rather than a local formula, so a
  // label reads the same here as on the canvas, in review and in the label picker.
  const colorForLabel = (label) => getLabelColor(label.id);


  return (
    <div>
      {/* Error display */}
      {error && (
        <div className="mb-3 p-2 bg-errBg border border-errLn rounded text-err text-sm">
          {error}
        </div>
      )}

      {/* Labels list */}
      {labelHierarchy.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-t2 text-sm font-medium">No labels yet</p>
          <p className="text-t3 text-sm mt-1 max-w-md mx-auto">
            Start with a whole object — the thing other things sit inside. You can
            add its parts underneath it afterwards.
          </p>
          <button
            onClick={() => openAddModal()}
            className="mt-4 inline-flex items-center px-3 py-2 text-sm bg-accent text-onAccent rounded-lg hover:brightness-110 transition-colors"
          >
            <Plus size={16} className="mr-1" />
            Create first label
          </button>
        </div>
      ) : (
        <div>
          {/* Tree toolbar: what this label space actually is, and the two
              controls that make a large one navigable. */}
          <div className="flex items-center justify-between gap-3 pb-2 mb-1 border-b border-ln">
            <p className="text-xs text-t3">
              {totalLabels} label{totalLabels !== 1 ? 's' : ''}
              {' · '}
              {labelHierarchy.length} top-level
              {hasNesting && ` · ${depth} levels deep`}
              {totalLabels > 1 && (
                <span className="hidden sm:inline"> · drag a label onto another to nest it</span>
              )}
            </p>

            {hasNesting && (
              <div className="flex items-center gap-1">
                <button
                  onClick={expandAll}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs text-t2 rounded hover:bg-hv hover:text-t1 transition-colors"
                  title="Expand every label"
                >
                  <ChevronsUpDown size={13} />
                  Expand all
                </button>
                <button
                  onClick={collapseAll}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs text-t2 rounded hover:bg-hv hover:text-t1 transition-colors"
                  title="Collapse every label"
                >
                  <ChevronsDownUp size={13} />
                  Collapse all
                </button>
              </div>
            )}
          </div>

          <LabelHierarchyRenderer
            labels={labelHierarchy}
            expandedLabels={expandedLabels}
            onToggleExpanded={toggleExpanded}
            onAddLabel={openAddModal}
            onEditLabel={openEditModal}
            onMoveLabel={openMoveModal}
            onDeleteLabel={openDeleteModal}
            mode="editable"
            getLabelColor={colorForLabel}
            getRowProps={rowPropsFor}
          />

          {/* While dragging, the same strip becomes the way back out of the
              hierarchy — otherwise there is no drop target for "not a part of
              anything". Outside a drag it is the add button it has always been. */}
          {draggedLabel ? (
            <div
              onDragOver={handleDragOverRoot}
              onDrop={handleDropOnRoot}
              className={`mt-3 w-full flex items-center justify-center gap-2 px-3 py-2 text-sm border-2 border-dashed rounded-lg transition-colors ${
                drag.overRoot
                  ? strandedBy(null) > 0
                    ? 'border-warnLn bg-warnBg text-warn'
                    : 'border-acLn bg-acS text-ac'
                  : 'border-ln2 text-t3'
              }`}
            >
              Drop here to make "{draggedLabel.name}" a whole object of its own
              {drag.overRoot && strandedBy(null) > 0 && (
                <span className="rounded-full border border-warnLn px-2 py-0.5 text-[11px] font-medium motion-safe:animate-pulse">
                  would detach {strandedBy(null)} object{strandedBy(null) !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          ) : (
            <button
              onClick={() => openAddModal()}
              className="mt-3 w-full flex items-center justify-center px-3 py-2 text-sm border-2 border-dashed border-ln2 text-t2 rounded-lg hover:border-acLn hover:text-ac transition-colors"
            >
              <Plus size={16} className="mr-1" />
              Add top-level label
            </button>
          )}
        </div>
      )}

      {/* Add Label Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" onClick={() => setShowAddModal(false)}>
              <div className="absolute inset-0 bg-t3 opacity-75"></div>
            </div>

            <div className="inline-block align-bottom bg-p1 rounded-xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-p1 px-6 pt-6 pb-4">
                <h3 className="text-lg font-medium text-t1">
                  {targetParentLabel
                    ? `Add a part of "${targetParentLabel.name}"`
                    : 'Add a top-level label'}
                </h3>

                {/* Where it will land */}
                {parentPath.length > 0 && (
                  <p className="mt-1 text-xs text-t3 truncate">
                    {parentPath.join(' › ')} › <span className="text-t2">new label</span>
                  </p>
                )}

                <div className="mt-4 mb-3">
                  <input
                    type="text"
                    value={newLabelName}
                    onChange={(e) => setNewLabelName(e.target.value)}
                    placeholder="Enter label name..."
                    className="w-full px-3 py-2 border border-ln2 rounded-lg focus:ring-2 focus:ring-ac focus:border-transparent"
                    autoFocus
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') handleAddLabel();
                    }}
                  />
                </div>

                {/* The part-of rule, stated where the decision is actually made */}
                <p className="mb-4 text-xs text-t3 leading-relaxed">
                  {targetParentLabel ? (
                    <>
                      Nesting means <span className="font-medium text-t2">part of</span>:{' '}
                      <span className="font-medium text-t2">
                        {newLabelName.trim() || 'this label'}
                      </span>{' '}
                      is part of{' '}
                      <span className="font-medium text-t2">{targetParentLabel.name}</span>.
                      Objects carrying it can then only be annotated inside a{' '}
                      {targetParentLabel.name}. If it is a <em>kind of</em>{' '}
                      {targetParentLabel.name} rather than a part of one, add it at the
                      top level instead.
                    </>
                  ) : (
                    <>
                      Top-level labels are whole objects — the things other things sit
                      inside. Add their parts underneath them with the{' '}
                      <span className="font-medium text-t2">+</span> on the label's row.
                    </>
                  )}
                </p>

                <div className="flex justify-end space-x-3">
                  <button
                    onClick={() => setShowAddModal(false)}
                    className="px-4 py-2 text-sm text-t2 bg-well rounded-lg hover:bg-hv2 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleAddLabel}
                    disabled={!newLabelName.trim() || loading}
                    className="px-4 py-2 text-sm text-onAccent bg-accent rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? 'Creating...' : 'Create'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Edit Label Modal */}
      {showEditModal && labelToEdit && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" onClick={() => setShowEditModal(false)}>
              <div className="absolute inset-0 bg-t3 opacity-75"></div>
            </div>

            <div className="inline-block align-bottom bg-p1 rounded-xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-p1 px-6 pt-6 pb-4">
                <h3 className="text-lg font-medium text-t1 mb-4">
                  Edit Label
                </h3>

                <div className="mb-4">
                  <input
                    type="text"
                    value={editLabelName}
                    onChange={(e) => setEditLabelName(e.target.value)}
                    placeholder="Enter label name..."
                    className="w-full px-3 py-2 border border-ln2 rounded-lg focus:ring-2 focus:ring-ac focus:border-transparent"
                    autoFocus
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') handleEditLabel();
                    }}
                  />
                </div>

                <div className="flex justify-end space-x-3">
                  <button
                    onClick={() => {
                      setShowEditModal(false);
                      setLabelToEdit(null);
                      setEditLabelName('');
                    }}
                    className="px-4 py-2 text-sm text-t2 bg-well rounded-lg hover:bg-hv2 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleEditLabel}
                    disabled={!editLabelName.trim() || loading}
                    className="px-4 py-2 text-sm text-onAccent bg-accent rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? 'Updating...' : 'Update'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Move Label Modal */}
      {showMoveModal && labelToMove && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" onClick={closeMoveModal}>
              <div className="absolute inset-0 bg-t3 opacity-75"></div>
            </div>

            <div className="inline-block align-bottom bg-p1 rounded-xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-p1 px-6 pt-6 pb-4">
                <h3 className="text-lg font-medium text-t1 flex items-center gap-2">
                  <CornerUpRight size={18} className="text-ac" />
                  Move "{labelToMove.name}"
                </h3>

                {!moveBlocked ? (
                  <>
                    <label htmlFor="move-target" className="block mt-4 text-sm text-t2">
                      Make it a part of
                    </label>
                    <select
                      id="move-target"
                      value={moveTarget}
                      onChange={(e) => setMoveTarget(e.target.value)}
                      className="mt-1 w-full px-3 py-2 border border-ln2 rounded-lg bg-p1 text-t1 focus:ring-2 focus:ring-ac focus:border-transparent"
                    >
                      <option value="root">Top level — a whole object of its own</option>
                      {moveCandidates.map((candidate) => (
                        <option key={candidate.id} value={String(candidate.id)}>
                          {candidate.path.join(' › ')}
                        </option>
                      ))}
                    </select>

                    {/* Say what the move will mean, in the part-of terms the tree uses */}
                    <p className="mt-3 text-xs text-t3 leading-relaxed">
                      {moveTargetLabel ? (
                        <>
                          <span className="font-medium text-t2">{labelToMove.name}</span> becomes a
                          part of <span className="font-medium text-t2">{moveTargetLabel.name}</span>.
                          Objects labelled {labelToMove.name} will then only be annotatable inside a{' '}
                          {moveTargetLabel.name}.
                        </>
                      ) : (
                        <>
                          <span className="font-medium text-t2">{labelToMove.name}</span> becomes a
                          whole object of its own, no longer a part of anything. Objects labelled{' '}
                          {labelToMove.name} will no longer be annotatable inside another object.
                        </>
                      )}
                    </p>
                  </>
                ) : (
                  <div className="mt-4 p-3 rounded-lg bg-warnBg border border-warnLn">
                    <div className="flex gap-2.5">
                      <AlertTriangle className="w-5 h-5 text-warn shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-t1">
                          {moveBlocked.affectedCount} annotated object
                          {moveBlocked.affectedCount !== 1 ? 's' : ''} would be left in a place this
                          move makes illegal
                        </p>
                        <p className="mt-1 text-xs text-t2 leading-relaxed">
                          They are labelled{' '}
                          <span className="font-medium">{labelToMove.name}</span> and sit inside
                          objects that will no longer contain that part. Continuing detaches them:
                          they keep their label, but stop being nested inside anything. Their
                          containment is not restored if you move the label back.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-5 flex justify-end space-x-3">
                  <button
                    onClick={closeMoveModal}
                    className="px-4 py-2 text-sm text-t2 bg-well rounded-lg hover:bg-hv2 transition-colors"
                  >
                    Cancel
                  </button>
                  {!moveBlocked ? (
                    <button
                      onClick={() => handleMoveLabel(false)}
                      disabled={loading || moveIsNoOp}
                      className="px-4 py-2 text-sm text-onAccent bg-accent rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      title={moveIsNoOp ? 'Already where it is' : undefined}
                    >
                      {loading ? 'Moving...' : 'Move'}
                    </button>
                  ) : (
                    <button
                      onClick={() => handleMoveLabel(true)}
                      disabled={loading}
                      className="px-4 py-2 text-sm text-onAccent bg-warn rounded-lg hover:brightness-110 disabled:opacity-50 transition-colors"
                    >
                      {loading
                        ? 'Moving...'
                        : `Detach ${moveBlocked.affectedCount} object${moveBlocked.affectedCount !== 1 ? 's' : ''} and move`}
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteModal && labelToDelete && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" onClick={() => setShowDeleteModal(false)}>
              <div className="absolute inset-0 bg-t3 opacity-75"></div>
            </div>

            <div className="inline-block align-bottom bg-p1 rounded-xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-p1 px-6 pt-6 pb-4">
                <div className="flex items-start">
                  <div className="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-errBg">
                    <Trash2 className="h-6 w-6 text-err" />
                  </div>
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-t1">Delete Label</h3>
                    <div className="mt-2">
                      <p className="text-sm text-t3">
                        Are you sure you want to delete the label "{labelToDelete.name}"? 
                        This action cannot be undone and may affect existing annotations.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="mt-5 sm:mt-4 sm:flex sm:flex-row-reverse">
                  <button
                    onClick={handleDeleteLabel}
                    disabled={loading}
                    className="w-full inline-flex justify-center rounded-lg border border-transparent shadow-sm px-4 py-2 bg-err text-base font-medium text-onAccent hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-err sm:ml-3 sm:w-auto sm:text-sm disabled:opacity-50"
                  >
                    {loading ? 'Deleting...' : 'Delete'}
                  </button>
                  <button
                    onClick={() => setShowDeleteModal(false)}
                    className="mt-3 w-full inline-flex justify-center rounded-lg border border-ln2 shadow-sm px-4 py-2 bg-p1 text-base font-medium text-t2 hover:text-t3 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ac sm:mt-0 sm:w-auto sm:text-sm"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EditableLabels; 