import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  calculateCoverage,
  calculatePolicyDiff,
  formatChangeSummary,
} from "./orchestration/orchestrationViewModel";
import MatrixTopSummary from "./orchestration/matrix/MatrixTopSummary";
import MatrixGrid from "./orchestration/matrix/MatrixGrid";
import MatrixSideDrawer from "./orchestration/matrix/MatrixSideDrawer";
import RoutingSaveBar from "./orchestration/RoutingSaveBar";

const hasActiveLabel = (labelsById, labelId) => {
  if (labelId == null || labelId === "") return true;
  const labels = labelsById || {};
  return (
    Object.prototype.hasOwnProperty.call(labels, labelId) ||
    Object.prototype.hasOwnProperty.call(labels, String(labelId))
  );
};

const getEditableBindings = (bindings, labelsById) => {
  const activeBindings = (Array.isArray(bindings) ? bindings : []).filter(
    (binding) => hasActiveLabel(labelsById, binding?.label_id)
  );
  return JSON.parse(JSON.stringify(activeBindings));
};

/**
 * ModelOrchestrationPanel Coordinator (Design B)
 *
 * Coordinates draft state, matrix grid view, right slide-over drawers,
 * and bottom save bar.
 */
export default function ModelOrchestrationPanel({
  datasetId,
  policy,
  labelsById = {},
  catalog = { models: [], retrieval_strategies: [] },
  onSavePolicy,
  onDeletePolicy,
  isSaving = false,
  isDeleting = false,
  canEdit = true,
}) {
  // Local draft bindings state initialized from loaded policy
  const [draftBindings, setDraftBindings] = useState([]);
  const [statusMessage, setStatusMessage] = useState(null);

  // Active drawer target { task, labelId }
  const [drawerTarget, setDrawerTarget] = useState(null);
  const isBusy = isSaving || isDeleting;
  const previousDatasetIdRef = useRef(datasetId);

  // Sync draft bindings when policy changes (e.g. initial load or dataset switch)
  useEffect(() => {
    const datasetChanged = previousDatasetIdRef.current !== datasetId;
    setDraftBindings(getEditableBindings(policy?.bindings, labelsById));
    if (datasetChanged) {
      setStatusMessage(null);
    }
    previousDatasetIdRef.current = datasetId;
  }, [policy, datasetId]);

  // Pure view-model computations
  const coverage = useMemo(
    () => calculateCoverage(draftBindings, labelsById, catalog),
    [draftBindings, labelsById, catalog]
  );

  const policyDiff = useMemo(
    () => calculatePolicyDiff(policy?.bindings || [], draftBindings),
    [policy?.bindings, draftBindings]
  );

  const hasUnsavedChanges = policyDiff.length > 0;

  const changeSummary = useMemo(
    () => formatChangeSummary(policyDiff, { labelsById, catalog }),
    [policyDiff, labelsById, catalog]
  );

  const handleUpdateBinding = (task, labelId, newBinding) => {
    setDraftBindings((prev) => {
      const filtered = prev.filter(
        (b) =>
          !(
            b.task === task &&
            (labelId == null
              ? b.label_id == null
              : Number(b.label_id) === Number(labelId))
          )
      );
      if (newBinding) {
        filtered.push(newBinding);
      }
      return filtered;
    });
    setStatusMessage(null);
  };

  const handleSave = async () => {
    if (!canEdit || isSaving) return;
    setStatusMessage(null);
    try {
      await onSavePolicy(draftBindings);
      setStatusMessage({ type: "success", text: "Routing policy saved successfully." });
    } catch (err) {
      setStatusMessage({
        type: "error",
        text: err.message || "Failed to save routing policy.",
      });
    }
  };

  const handleReset = () => {
    setDraftBindings(getEditableBindings(policy?.bindings, labelsById));
    setStatusMessage(null);
  };

  const handleClearAll = async () => {
    if (!canEdit || isDeleting || isSaving) return;
    try {
      await onDeletePolicy();
      setDraftBindings([]);
      setStatusMessage({ type: "success", text: "Routing policy cleared." });
    } catch (err) {
      setStatusMessage({
        type: "error",
        text: err.message || "Failed to clear routing policy.",
      });
    }
  };

  const handleSelectCell = (taskKey, targetLabelId) => {
    if (isBusy) return;
    setDrawerTarget({ task: taskKey, labelId: targetLabelId });
  };

  return (
    <div className="space-y-6" data-testid="model-orchestration-desk">
      {/* 1. Top 3 Category Summary Cards */}
      <MatrixTopSummary coverage={coverage} />

      {/* 2. Main Matrix Grid */}
      <MatrixGrid
        labelsById={labelsById}
        catalog={catalog}
        draftBindings={draftBindings}
        onSelectCell={handleSelectCell}
        canOpenDrawer={!isBusy}
      />

      {/* 3. Bottom Routing Save Bar */}
      <RoutingSaveBar
        hasUnsavedChanges={hasUnsavedChanges}
        changeSummary={changeSummary}
        hasSavedPolicy={Boolean(policy)}
        onSave={handleSave}
        onReset={handleReset}
        onClear={handleClearAll}
        isSaving={isSaving}
        isDeleting={isDeleting}
        canEdit={canEdit}
        statusMessage={statusMessage}
      />

      {/* 4. Slide-over Model & Inputs Drawer */}
      <MatrixSideDrawer
        isOpen={Boolean(drawerTarget)}
        onClose={() => setDrawerTarget(null)}
        target={drawerTarget}
        labelsById={labelsById}
        catalog={catalog}
        draftBindings={draftBindings}
        onSaveRoute={(taskKey, targetLabelId, newBinding) => {
          handleUpdateBinding(taskKey, targetLabelId, newBinding);
          setDrawerTarget(null);
        }}
        onUnbindRoute={(taskKey, targetLabelId) => {
          handleUpdateBinding(taskKey, targetLabelId, null);
          setDrawerTarget(null);
        }}
        canEdit={canEdit && !isBusy}
      />
    </div>
  );
}
