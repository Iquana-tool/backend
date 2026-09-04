import React from "react";
import { Star, AlertTriangle, CornerDownRight, Plus } from "lucide-react";
import {
  ORCHESTRATION_CATEGORIES,
  ROUTE_STATUS,
  resolveEffectiveRoute,
} from "../orchestrationViewModel";
import { resolveLabelColor } from "../../../annotationPage/workspace/labelColorUtils";

export { resolveLabelColor as getLabelColor };

/**
 * MatrixGrid Component (Design B)
 *
 * Full matrix view rendering one cell per (task, label).
 * Displays Task Default row, hierarchical label rows, and visual cues for
 * configured vs inherited vs unconfigured cells.
 */
export default function MatrixGrid({
  labelsById = {},
  catalog = { models: [], retrieval_strategies: [] },
  draftBindings = [],
  onSelectCell,
  canOpenDrawer = true,
  className = "",
}) {
  const models = catalog?.models || [];

  // Build hierarchical ordered label list
  const allLabels = Object.values(labelsById);
  const rootLabels = allLabels.filter(
    (l) => l.parent_id == null || !labelsById[l.parent_id]
  );
  const childMap = {};
  allLabels.forEach((l) => {
    if (l.parent_id != null) {
      if (!childMap[l.parent_id]) childMap[l.parent_id] = [];
      childMap[l.parent_id].push(l);
    }
  });

  const orderedRows = [];
  const traverse = (label, depth = 0, isLastChild = false) => {
    const children = childMap[label.id] || [];
    orderedRows.push({
      ...label,
      depth,
      hasChildren: children.length > 0,
      isLastChild,
    });
    children.forEach((child, index) =>
      traverse(child, depth + 1, index === children.length - 1)
    );
  };
  rootLabels.forEach((root) => traverse(root, 0, false));

  // If no hierarchy is defined, fall back to flat list
  const displayLabels = orderedRows.length > 0 ? orderedRows : allLabels.map((l) => ({ ...l, depth: 0 }));

  // Column definitions mapping to canonical task keys with theme-matching white-tinted rounded rectangular borders
  const columns = [
    {
      key: "prompted-segmentation",
      task: "prompted-segmentation",
      label: "Prompted segmentation",
      dotColor: "bg-teal-500 dark:bg-teal-400 shadow-teal-400/50",
      dotText: "text-teal-500 dark:text-teal-400",
      explicitCard: "border-2 border-teal-500 bg-teal-500/15 dark:bg-teal-500/20 hover:border-teal-400 text-teal-950 dark:text-white shadow-xs",
      inheritedCard: "border border-dashed border-teal-500/40 bg-teal-500/[0.05] dark:bg-teal-950/20 text-teal-900 dark:text-teal-100 hover:border-teal-500/60",
      emptyCard: "border border-teal-500/30 bg-teal-500/[0.04] dark:bg-teal-500/[0.03] hover:border-teal-500/60 hover:bg-teal-500/10 text-teal-800 dark:text-teal-200/70 hover:text-teal-950 dark:hover:text-teal-100",
    },
    {
      key: "instance-suggestion",
      task: "instance-suggestion",
      label: "Within-image suggestion",
      dotColor: "bg-cyan-500 dark:bg-cyan-400 shadow-cyan-400/50",
      dotText: "text-cyan-500 dark:text-cyan-400",
      explicitCard: "border-2 border-cyan-500 bg-cyan-500/15 dark:bg-cyan-500/20 hover:border-cyan-400 text-cyan-950 dark:text-white shadow-xs",
      inheritedCard: "border border-dashed border-cyan-500/40 bg-cyan-500/[0.05] dark:bg-cyan-950/20 text-cyan-900 dark:text-cyan-100 hover:border-cyan-500/60",
      emptyCard: "border border-cyan-500/30 bg-cyan-500/[0.04] dark:bg-cyan-500/[0.03] hover:border-cyan-500/60 hover:bg-cyan-500/10 text-cyan-800 dark:text-cyan-200/70 hover:text-cyan-950 dark:hover:text-cyan-100",
    },
    {
      key: "instance-segmentation",
      task: "instance-segmentation",
      label: "Instance segmentation",
      dotColor: "bg-amber-500 dark:bg-amber-400 shadow-amber-400/50",
      dotText: "text-amber-500 dark:text-amber-400",
      explicitCard: "border-2 border-amber-500 bg-amber-500/15 dark:bg-amber-500/20 hover:border-amber-400 text-amber-950 dark:text-white shadow-xs",
      inheritedCard: "border border-dashed border-amber-500/40 bg-amber-500/[0.05] dark:bg-amber-950/20 text-amber-900 dark:text-amber-100 hover:border-amber-500/60",
      emptyCard: "border border-amber-500/30 bg-amber-500/[0.04] dark:bg-amber-500/[0.03] hover:border-amber-500/60 hover:bg-amber-500/10 text-amber-800 dark:text-amber-200/70 hover:text-amber-950 dark:hover:text-amber-100",
    },
    {
      key: "cross-image-suggestion",
      task: "cross-image-suggestion",
      label: "Cross-image suggestion",
      dotColor: "bg-purple-500 dark:bg-purple-400 shadow-purple-400/50",
      dotText: "text-purple-500 dark:text-purple-400",
      explicitCard: "border-2 border-purple-500 bg-purple-500/15 dark:bg-purple-500/20 hover:border-purple-400 text-purple-950 dark:text-white shadow-xs",
      inheritedCard: "border border-dashed border-purple-500/40 bg-purple-500/[0.05] dark:bg-purple-950/20 text-purple-900 dark:text-purple-100 hover:border-purple-500/60",
      emptyCard: "border border-purple-500/30 bg-purple-500/[0.04] dark:bg-purple-500/[0.03] hover:border-purple-500/60 hover:bg-purple-500/10 text-purple-800 dark:text-purple-200/70 hover:text-purple-950 dark:hover:text-purple-100",
    },
  ];

  // Tree guide dimensions
  const TREE_ORIGIN = 18;
  const TREE_INDENT = 28;
  const LABEL_DOT_SIZE = 16;

  const renderHierarchyGuides = (label) => {
    const guides = [];

    for (let level = 0; level < label.depth - 1; level += 1) {
      guides.push(
        <span
          key={`ancestor-rail-${level}`}
          className="absolute top-0 bottom-0 z-0 w-px bg-ln2 dark:bg-slate-600/70"
          style={{ left: `${TREE_ORIGIN + level * TREE_INDENT}px` }}
          data-hierarchy-guide="vertical"
          aria-hidden="true"
        />
      );
    }

    if (label.depth > 0) {
      const parentRailLeft = TREE_ORIGIN + (label.depth - 1) * TREE_INDENT;
      const labelDotLeft =
        TREE_ORIGIN - LABEL_DOT_SIZE / 2 + label.depth * TREE_INDENT;

      guides.push(
        <span
          key="current-rail"
          className="absolute top-0 z-0 w-px bg-ln2 dark:bg-slate-600/70"
          style={{
            left: `${parentRailLeft}px`,
            bottom: label.isLastChild ? "50%" : "0",
          }}
          data-hierarchy-guide="vertical"
          aria-hidden="true"
        />
      );
      guides.push(
        <span
          key="branch-elbow"
          className="absolute z-0 h-px bg-ln2 dark:bg-slate-600/70"
          style={{
            left: `${parentRailLeft}px`,
            top: "50%",
            width: `${labelDotLeft - parentRailLeft + 1}px`,
          }}
          data-hierarchy-guide="horizontal"
          aria-hidden="true"
        />
      );
    }

    if (label.hasChildren) {
      guides.push(
        <span
          key="child-rail"
          className="absolute bottom-0 z-0 w-px bg-ln2 dark:bg-slate-600/70"
          style={{
            left: `${TREE_ORIGIN + label.depth * TREE_INDENT}px`,
            top: "50%",
          }}
          data-hierarchy-guide="vertical"
          aria-hidden="true"
        />
      );
    }

    return guides;
  };

  const renderCellContent = (col, labelId) => {
    const isDefault = labelId == null;
    const effective = resolveEffectiveRoute({
      task: col.task,
      labelId,
      bindings: draftBindings,
      catalog,
    });

    const isExplicit = effective.status === ROUTE_STATUS.EXPLICIT;
    const isInherited = effective.status === ROUTE_STATUS.INHERITED;
    const isIncompatible = effective.status === ROUTE_STATUS.UNBOUND_INCOMPATIBLE;
    const isStale =
      effective.status === ROUTE_STATUS.STALE ||
      effective.status === ROUTE_STATUS.STALE_DEFAULT;
    const isBatchTask = col.task === "instance-segmentation" || col.task === "cross-image-suggestion";
    const model = effective.model;
    const isTrainedOnDataset = Boolean(model?.trained_on_dataset || model?.is_fine_tuned || model?.is_trained_here);

    // Title, Subtitle, and Task-Specific Tooltip
    let title = "";
    let subtitle = "";
    let helpText = "";

    if (isExplicit && model) {
      title = model.name || model.registry_key;
      const badgesList = Array.isArray(model.badges) && model.badges.length > 0
        ? model.badges
        : [model.latency_badge, model.model_size].filter(Boolean);
      const perf = badgesList.length > 0 ? `${badgesList.join(" · ")} · ` : "";
      const classText =
        model.label_ids?.length > 0
          ? `${model.label_ids.length} class${model.label_ids.length > 1 ? "es" : ""}`
          : "class-agnostic";
      subtitle = `${perf}${classText}`;
      helpText = `${title} · Configured route for ${col.label}`;
    } else if (isInherited && model) {
      title = `Inherits ${model.name || model.registry_key}`;
      subtitle = "Task default";
      helpText = `Inherits ${model.name || model.registry_key} from task default`;
    } else if (isIncompatible) {
      title = "Needs compatible model";
      subtitle = "Default incompatible";
      helpText = "Task default model does not support this label. Choose a compatible model override.";
    } else if (isStale) {
      title = model ? (model.name || model.registry_key) : "Missing model";
      subtitle = effective.reason === "incompatible_model" ? "Incompatible with label" : "Model not found in catalog";
      helpText = subtitle;
    } else if (isDefault) {
      title = "No task default";
      subtitle = "";
      helpText = isBatchTask
        ? "No task default configured. Batch operations will omit this task unless a label route is configured."
        : "No task default configured. Interactive tools fall back to personal favorite or first compatible model.";
    } else {
      title = "Not configured";
      subtitle = "";
      helpText = isBatchTask
        ? "Not configured. Uses task default when available; omitted from batch operations if no default exists."
        : "Not configured. Uses task default when available, or falls back to personal favorite/model fallback.";
    }

    const handleClick = () => {
      if (!canOpenDrawer || !onSelectCell) return;
      onSelectCell(col.task, labelId);
    };

    return (
      <button
        type="button"
        onClick={handleClick}
        disabled={!canOpenDrawer}
        title={helpText}
        aria-label={`${col.label}: ${title}${subtitle ? ` (${subtitle})` : ""}. ${helpText}`}
        className={`w-full text-left rounded-xl transition-all cursor-pointer group flex items-center justify-between gap-2 min-h-[48px] p-2.5 ${
          isStale
            ? "border border-amber-500/90 bg-amber-500/10 ring-1 ring-amber-400/30"
            : isIncompatible
            ? "border border-dashed border-amber-500/60 bg-amber-500/10 hover:bg-amber-500/15 hover:border-amber-400 text-amber-700 dark:text-amber-300"
            : isExplicit
            ? col.explicitCard
            : isInherited
            ? col.inheritedCard
            : col.emptyCard
        }`}
        data-testid={`matrix-cell-${col.task}-${labelId ?? "default"}`}
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 min-w-0">
            {isStale || isIncompatible ? (
              <AlertTriangle size={12} className="text-amber-500 dark:text-amber-400 shrink-0" />
            ) : isExplicit ? (
              <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${col.dotColor}`} />
            ) : isInherited ? (
              <CornerDownRight size={11} className="text-teal-600 dark:text-teal-300/80 shrink-0" />
            ) : (
              <Plus size={11} className="shrink-0 opacity-60 group-hover:opacity-100 transition-opacity" />
            )}

            <span
              className={`text-xs truncate ${
                isExplicit
                  ? "font-bold text-teal-950 dark:text-white"
                  : isIncompatible
                  ? "text-amber-700 dark:text-amber-300 font-semibold"
                  : isInherited
                  ? "font-semibold text-teal-950 dark:text-white/90 group-hover:text-t1"
                  : "font-medium"
              }`}
            >
              {title}
            </span>

            {isTrainedOnDataset && (
              <Star
                size={11}
                className="text-teal-500 dark:text-teal-300 fill-teal-500 dark:fill-teal-300 shrink-0 ml-0.5"
                title="Trained on this dataset"
              />
            )}
          </div>

          {subtitle ? (
            <p className="text-[10px] opacity-80 truncate mt-0.5 pl-3.5">
              {subtitle}
            </p>
          ) : null}
        </div>
      </button>
    );
  };

  return (
    <div
      className={`overflow-x-auto rounded-2xl border border-ln bg-p1 p-4 shadow-sm ${className}`}
      data-testid="matrix-grid"
    >
      <div className="min-w-[860px] md:min-w-0">
        {/* Matrix Header Row */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3.5 px-3 py-2 text-xs items-center bg-well/30 rounded-xl mb-2">
          <div className="font-bold text-[10px] uppercase tracking-wider text-t3">
            Label
          </div>
          {columns.map((col) => (
            <div key={col.key} className="space-y-0.5">
              <div className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full shrink-0 ${col.dotColor}`} />
                <span className="font-bold text-t1">{col.label}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Row 1: Task Default Row */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3.5 px-3 py-2.5 rounded-2xl bg-amber-500/5 border border-amber-500/20 dark:bg-amber-500/[0.03] dark:border-amber-500/15 items-center my-2">
          <div>
            <div className="flex items-center gap-2">
              <Star size={13} className="text-amber-500 fill-amber-500 shrink-0" />
              <span className="text-xs font-bold text-t1">Task default</span>
            </div>
            <p className="text-[10px] text-amber-600 dark:text-amber-500/80 mt-0.5 pl-5">
              Default where compatible
            </p>
          </div>

          {columns.map((col) => (
            <div key={col.key}>
              {renderCellContent(col, null)}
            </div>
          ))}
        </div>

        {/* Label Rows */}
        {displayLabels.map((l) => {
          const parent = l.parent_id != null ? labelsById[l.parent_id] : null;
          const levelText = `Level ${l.depth + 1}${
            l.instance_count != null ? ` · ${l.instance_count} instances` : ""
          }${parent ? ` · child of ${parent.name}` : ""}`;
          const dotColor = resolveLabelColor(l);

          return (
            <div
              key={l.id}
              className={`group grid grid-cols-1 md:grid-cols-5 gap-3.5 px-3 py-2.5 rounded-2xl transition-colors hover:bg-well/40 items-center my-0.5 ${
                l.depth === 0 ? "bg-well/20" : "bg-transparent"
              }`}
              data-testid={`matrix-label-row-${l.id}`}
              data-hierarchy-depth={l.depth}
            >
              {/* Label Column with a continuous parent/child tree rail */}
              <div
                className="relative min-h-[60px] min-w-0"
                style={{
                  paddingLeft: `${
                    TREE_ORIGIN - LABEL_DOT_SIZE / 2 + l.depth * TREE_INDENT
                  }px`,
                }}
              >
                {renderHierarchyGuides(l)}
                <div className="relative z-10 flex min-h-[60px] min-w-0 flex-col justify-center">
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className={`shrink-0 rounded-full ring-1 ring-black/10 dark:ring-white/10 ${
                        l.depth === 0 ? "h-4 w-4 shadow-sm" : "h-3.5 w-3.5 shadow-xs"
                      }`}
                      style={{ backgroundColor: dotColor }}
                    />
                    <span className="truncate text-sm font-semibold text-t1">
                      {l.name}
                    </span>
                  </div>
                  <p
                    className="relative z-10 mt-1 truncate pl-5 text-[11px] text-t3"
                  >
                    {levelText}
                  </p>
                </div>
              </div>

              {columns.map((col) => (
                <div key={col.key}>
                  {renderCellContent(col, l.id)}
                </div>
              ))}
            </div>
          );
        })}

        {/* Bottom Legend */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-3 pt-3 mt-2 border-t border-ln text-[11px] text-t3">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-md border-2 border-teal-500 bg-teal-500/20" />
            <span className="font-medium text-t1">configured route</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-md border border-dashed border-teal-500/40 bg-teal-500/10" />
            <span>inherited</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-md border border-amber-500/60 bg-amber-500/10 flex items-center justify-center">
              <AlertTriangle size={8} className="text-amber-500" />
            </span>
            <span>needs attention</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Star size={11} className="text-teal-500 dark:text-teal-400 fill-teal-500 dark:fill-teal-400" />
            <span>trained on this dataset</span>
          </div>
        </div>
      </div>
    </div>
  );
}
