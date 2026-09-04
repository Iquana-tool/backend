import React, { useState } from "react";
import {
  CheckCircle2,
  AlertCircle,
  MousePointerClick,
  Wand2,
  Boxes,
  Scan,
  Star,
  Cpu,
  Timer,
} from "lucide-react";
import { TASK_ORDER } from "../../constants/tasks";
import { formatParams, formatLatency } from "./modelStats";
import TaskFavoriteChooser from "./TaskFavoriteChooser";

// Keep in sync with ModelCard's visual map: the primary task drives the tile
// gradient + icon so a model looks the same in the list and the detail panel.
const TASK_VISUAL = {
  "prompted-segmentation": { Icon: MousePointerClick, tile: "bg-acS text-ac" },
  "instance-suggestion": { Icon: Wand2, tile: "bg-acS text-ac" },
  "instance-segmentation": { Icon: Boxes, tile: "bg-warnBg text-warn" },
};
const DEFAULT_VISUAL = { Icon: Scan, tile: "bg-acS text-ac" };

const orderTasks = (tasks) =>
  [...(tasks || [])].sort((a, b) => TASK_ORDER.indexOf(a) - TASK_ORDER.indexOf(b));

/**
 * Compact, selectable preview of a model for the zoo's left list. Shows just
 * what you scan by — name, primary-task icon, status, favorite, and up to two
 * headline stats — and defers the full spec sheet to the detail panel.
 */
const ModelChip = ({
  model,
  isSelected = false,
  isFavorite = false,
  selectedTask = "all",
  favorites = {},
  onSelect,
  onToggleFavorite,
}) => {
  const [showChooser, setShowChooser] = useState(false);
  const tasks = orderTasks(model.tasks);
  const primaryTask = tasks[0];
  const { Icon, tile } = TASK_VISUAL[primaryTask] || DEFAULT_VISUAL;
  const isReady = model.status !== "not_ready";

  const params = formatParams(model.performance?.num_parameters);
  const latency = formatLatency(model.performance?.latency_ms);

  const handleStarClick = (e) => {
    e.stopPropagation();
    if (selectedTask !== "all" || tasks.length <= 1) {
      onToggleFavorite?.(model, selectedTask !== "all" ? selectedTask : tasks[0]);
    } else {
      setShowChooser((prev) => !prev);
    }
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => onSelect?.(model)}
        aria-pressed={isSelected}
        className={`w-full text-left flex items-start gap-3 p-3 rounded-xl border transition-all ${
          isSelected
            ? "bg-acS border-acLn ring-1 ring-ac"
            : "bg-p1 border-ln hover:border-ln2 hover:bg-hv"
        }`}
      >
        <div
          className={`shrink-0 w-9 h-9 rounded-lg ${tile} flex items-center justify-center shadow-sm`}
        >
          <Icon className="w-4.5 h-4.5" style={{ width: 18, height: 18 }} />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <h3 className="text-sm font-semibold text-t1 truncate flex-1" title={model.name}>
              {model.name}
            </h3>
            <span
              className="shrink-0 p-0.5 rounded-md hover:bg-p2 relative"
              role="button"
              tabIndex={0}
              aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
              aria-pressed={isFavorite}
              title={
                isFavorite
                  ? "Favorite — preselected in the annotation page"
                  : "Set as your personal default model"
              }
              onClick={handleStarClick}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  handleStarClick(e);
                }
              }}
            >
              <Star
                className={`w-4 h-4 ${
                  isFavorite ? "fill-amber-400 text-warn" : "text-t3 hover:text-warn"
                }`}
                style={{ width: 15, height: 15 }}
              />
            </span>
          </div>

          <div className="mt-1 flex items-center flex-wrap gap-x-3 gap-y-1 text-[11px] text-t3">
            <span
              className={`inline-flex items-center gap-1 font-medium ${
                isReady ? "text-ok" : "text-warn"
              }`}
            >
              {isReady ? <CheckCircle2 className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
              {isReady ? "Ready" : "Needs training"}
            </span>
            {params && (
              <span className="inline-flex items-center gap-1" title="Parameters">
                <Cpu className="w-3 h-3 text-t3" />
                {params}
              </span>
            )}
            {latency && (
              <span className="inline-flex items-center gap-1" title="Approx. inference time">
                <Timer className="w-3 h-3 text-t3" />
                {latency}
              </span>
            )}
          </div>
        </div>
      </button>

      {showChooser && (
        <TaskFavoriteChooser
          model={model}
          favorites={favorites}
          onToggleTaskFavorite={(m, t) => {
            onToggleFavorite?.(m, t);
          }}
          onClose={() => setShowChooser(false)}
          anchorAlign="right"
        />
      )}
    </div>
  );
};

export default ModelChip;
