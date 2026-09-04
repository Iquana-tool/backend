import React from "react";
import {
  Wrench,
  GraduationCap,
  Lightbulb,
  Sparkles,
  ExternalLink,
  CheckCircle2,
  AlertCircle,
  MousePointerClick,
  RefreshCw,
  Hash,
  Boxes,
  Wand2,
  Scan,
  Star,
  Cpu,
  Timer,
  Zap,
  Gauge,
  MemoryStick,
  Database,
  Layers,
  Scale,
  Maximize,
} from "lucide-react";
import { getTaskMeta, TASK_ORDER } from "../../constants/tasks";
import {
  formatParams,
  formatGflops,
  formatLatency,
  formatThroughput,
  formatVram,
  formatResolution,
} from "./modelStats";
import { filterDisplayableModelTags } from "./modelTags";

import TaskFavoriteChooser from "./TaskFavoriteChooser";

const TASK_VISUAL = {
  "prompted-segmentation": { Icon: MousePointerClick, tile: "bg-acS text-ac" },
  "instance-suggestion": { Icon: Wand2, tile: "bg-acS text-ac" },
  "instance-segmentation": { Icon: Boxes, tile: "bg-warnBg text-warn" },
};
const DEFAULT_VISUAL = { Icon: Scan, tile: "bg-acS text-ac" };

const orderTasks = (tasks) =>
  [...(tasks || [])].sort((a, b) => TASK_ORDER.indexOf(a) - TASK_ORDER.indexOf(b));

// A single performance figure. Rendered only when the value is present.
const StatTile = ({ Icon, label, value, hint }) => {
  if (!value) return null;
  return (
    <div className="rounded-xl border border-ln bg-well px-3 py-2.5" title={hint}>
      <div className="flex items-center gap-1.5 text-[11px] font-medium text-t3">
        <Icon className="w-3.5 h-3.5 text-t3" />
        {label}
      </div>
      <div className="mt-1 text-lg font-semibold text-t1 tabular-nums leading-none">
        {value}
      </div>
    </div>
  );
};

const SpecRow = ({ Icon, label, children }) =>
  children == null || children === "" ? null : (
    <div className="flex items-start gap-3 py-2 border-b border-ln last:border-0">
      <div className="flex items-center gap-1.5 w-40 shrink-0 text-xs font-medium text-t3">
        <Icon className="w-3.5 h-3.5 text-t3" />
        {label}
      </div>
      <div className="min-w-0 flex-1 text-sm text-t1">{children}</div>
    </div>
  );

const SectionTitle = ({ children }) => (
  <h4 className="text-xs font-semibold uppercase tracking-wide text-t3 mb-2">{children}</h4>
);

/**
 * Full detail view ("viewer") for the model selected in the zoo list. Everything
 * the compact chip omits lives here: full description, performance figures,
 * capabilities, a spec table, and the fine-tune action.
 */
const ModelDetailPanel = ({
  model,
  isFavorite = false,
  selectedTask = "all",
  favorites = {},
  onToggleFavorite,
  onAction,
}) => {
  const [showChooser, setShowChooser] = React.useState(false);

  if (!model) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-10">
        <Scan className="w-14 h-14 text-t3 mb-4" />
        <p className="text-t3">Select a model to see its details.</p>
      </div>
    );
  }

  const tasks = orderTasks(model.tasks);
  const primaryTask = tasks[0];
  const { Icon, tile } = TASK_VISUAL[primaryTask] || DEFAULT_VISUAL;

  const badges = Array.isArray(model.badges) ? model.badges : [];
  const promptTypes = Array.isArray(model.promptTypesSupported) ? model.promptTypesSupported : [];
  const isReady = model.status !== "not_ready";
  const showFinetuning = model.trainable === true;
  const perf = model.performance || {};

  const tags = filterDisplayableModelTags(model.tags).filter((t) => {
    const key = String(t.key || "").toLowerCase();
    return key !== "task" && key !== "tasks" && !key.startsWith("task_")
      && !["publisher", "trained_label_names", "trained_on_dataset_id", "trained_on_dataset_name", "dataset_id", "dataset_name"]
        .includes(key);
  });

  const labelNames = Array.isArray(model.predictedLabelNames) ? model.predictedLabelNames : [];
  const predictedLabels = labelNames.filter(Boolean);
  const trainedOnDataset = model.trainedOnDatasetName || null;

  const capabilities = [
    model.pretrained && { label: "Pretrained", className: "bg-okBg text-ok" },
    model.refinementSupported && {
      label: "Refinement",
      className: "bg-acS text-ac",
      Icon: RefreshCw,
    },
    showFinetuning && {
      label: "Fine-tune on dataset",
      className: "bg-acS text-ac",
      Icon: GraduationCap,
    },
  ].filter(Boolean);

  const perfStats = [
    { Icon: Cpu, label: "Parameters", value: formatParams(perf.num_parameters) },
    { Icon: Zap, label: "GFLOPs", value: formatGflops(perf.gflops) },
    {
      Icon: Timer,
      label: "Inference time",
      value: formatLatency(perf.latency_ms),
      hint: perf.reference_device ? `Measured on ${perf.reference_device}` : undefined,
    },
    { Icon: Gauge, label: "Throughput", value: formatThroughput(perf.throughput_img_s) },
    { Icon: MemoryStick, label: "Peak VRAM", value: formatVram(perf.peak_vram_mb) },
  ];
  const hasPerf = perfStats.some((s) => s.value);
  const perfContext = [perf.reference_device, formatResolution(perf.reference_input_size)]
    .filter(Boolean)
    .join(", ");

  const hasSpecTable =
    model.architecture ||
    model.license ||
    formatResolution(model.inputResolution) ||
    promptTypes.length > 0 ||
    model.refinementSupported ||
    predictedLabels.length > 0 ||
    trainedOnDataset ||
    tags.length > 0;

  const handleStarClick = (e) => {
    e.stopPropagation();
    if (selectedTask !== "all" || tasks.length <= 1) {
      onToggleFavorite?.(model, selectedTask !== "all" ? selectedTask : tasks[0]);
    } else {
      setShowChooser((prev) => !prev);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 max-w-3xl">
        {/* Header */}
        <div className="flex items-start gap-4">
          <div
            className={`shrink-0 w-14 h-14 rounded-2xl ${tile} flex items-center justify-center shadow-sm`}
          >
            <Icon className="w-7 h-7" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <h2 className="text-2xl font-bold text-t1 leading-tight">{model.name}</h2>
                {model.identifier && (
                  <p className="text-xs text-t3 font-mono truncate mt-1" title={model.identifier}>
                    {model.identifier}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0 relative">
                {model.status && (
                  <span
                    className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold ${
                      isReady ? "bg-okBg text-ok" : "bg-warnBg text-warn"
                    }`}
                  >
                    {isReady ? (
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    ) : (
                      <AlertCircle className="w-3.5 h-3.5" />
                    )}
                    {isReady ? "Ready" : "Needs training"}
                  </span>
                )}
                <button
                  type="button"
                  onClick={handleStarClick}
                  aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
                  aria-pressed={isFavorite}
                  title={
                    isFavorite
                      ? "Favorite — preselected in the annotation page"
                      : "Set as your personal default model"
                  }
                  className="p-1.5 rounded-lg hover:bg-hv transition-colors"
                >
                  <Star
                    className={
                      isFavorite ? "fill-amber-400 text-warn" : "text-t3 hover:text-warn"
                    }
                    style={{ width: 20, height: 20 }}
                  />
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
            </div>

            {/* Capability + badge chips with per-task star controls */}
            <div className="mt-3 flex flex-wrap gap-1.5">
              {tasks.map((taskKey) => {
                const meta = getTaskMeta(taskKey);
                const isTaskFav = favorites[taskKey] === model.identifier;
                return (
                  <span
                    key={taskKey}
                    className={`inline-flex items-center gap-1.5 pl-2.5 pr-1.5 py-0.5 rounded-full text-[11px] font-medium ${meta.chip}`}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${meta.dot}`} />
                    <span>{meta.label}</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleFavorite?.(model, taskKey);
                      }}
                      aria-label={`Toggle favorite for ${meta.label}`}
                      className="p-0.5 rounded hover:bg-black/10 transition-colors"
                      title={isTaskFav ? `Favorite for ${meta.label}` : `Set as default for ${meta.label}`}
                    >
                      <Star
                        className={`w-3 h-3 ${
                          isTaskFav ? "fill-amber-400 text-warn" : "text-t3 hover:text-warn"
                        }`}
                      />
                    </button>
                  </span>
                );
              })}
              {badges.map((badge, i) => (
                <span
                  key={`badge-${i}`}
                  className="inline-flex items-center gap-1 px-2 py-0.5 bg-acS text-ac rounded-full text-[11px] font-medium"
                >
                  <Sparkles className="w-3 h-3" />
                  {badge}
                </span>
              ))}
              {capabilities.map((cap, i) => (
                <span
                  key={`cap-${i}`}
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${cap.className}`}
                >
                  {cap.Icon && <cap.Icon className="w-3 h-3" />}
                  {cap.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Description */}
        {model.description && (
          <p className="mt-5 text-sm text-t2 leading-relaxed whitespace-pre-wrap">
            {model.description}
          </p>
        )}

        {/* Usage tip */}
        {model.usageTip && (
          <div className="mt-4 flex items-start gap-2 p-3 bg-warnBg rounded-lg">
            <Lightbulb className="w-4 h-4 text-warn mt-0.5 shrink-0" />
            <p className="text-sm text-warn leading-relaxed">{model.usageTip}</p>
          </div>
        )}

        {/* Performance */}
        {hasPerf && (
          <div className="mt-6">
            <SectionTitle>Performance</SectionTitle>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
              {perfStats.map((s) => (
                <StatTile key={s.label} {...s} />
              ))}
            </div>
            <p className="mt-2 text-[11px] text-t3">
              Approximate figures{perfContext ? `, measured at ${perfContext}` : ""}. Actual speed
              varies with hardware and input size.
            </p>
          </div>
        )}

        {/* Spec table */}
        {hasSpecTable && (
          <div className="mt-6">
            <SectionTitle>Specifications</SectionTitle>
            <div className="rounded-xl border border-ln px-4 py-1">
              <SpecRow Icon={Layers} label="Architecture">
                {model.architecture}
              </SpecRow>
              <SpecRow Icon={Scale} label="License">
                {model.license}
              </SpecRow>
              <SpecRow Icon={Maximize} label="Input resolution">
                {formatResolution(model.inputResolution)}
              </SpecRow>
              <SpecRow Icon={MousePointerClick} label="Prompt types">
                {promptTypes.length > 0 ? (
                  <span className="flex flex-wrap gap-1.5">
                    {promptTypes.map((pt, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center px-2 py-0.5 bg-acS text-ac rounded-full text-[11px] font-medium capitalize"
                      >
                        {String(pt).replace(/_/g, " ")}
                      </span>
                    ))}
                  </span>
                ) : null}
              </SpecRow>
              <SpecRow Icon={RefreshCw} label="Refinement">
                {model.refinementSupported ? "Supported" : null}
              </SpecRow>
              <SpecRow Icon={Hash} label="Predicted labels">
                {predictedLabels.length > 0 ? predictedLabels.join(", ") : null}
              </SpecRow>
              <SpecRow Icon={Database} label="Trained on dataset">
                {trainedOnDataset}
              </SpecRow>
              {tags.length > 0 && (
                <SpecRow Icon={Sparkles} label="Tags">
                  <span className="flex flex-wrap gap-1.5">
                    {tags.map((tag, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center px-2 py-0.5 bg-well text-t2 rounded-full text-[11px]"
                      >
                        {tag.key && <span className="text-t3 mr-1">{tag.key}</span>}
                        <span className="font-medium">{tag.value}</span>
                      </span>
                    ))}
                  </span>
                </SpecRow>
              )}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="mt-6 flex items-center gap-3">
          {model.infoUrl && (
            <a
              href={model.infoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-sm font-medium text-t3 hover:text-ac transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              Learn more
            </a>
          )}
          {showFinetuning && (
            <button
              onClick={() => onAction?.(model, "finetuning")}
              className="ml-auto inline-flex items-center gap-1.5 py-2 px-4 rounded-lg text-sm font-medium bg-accent text-onAccent hover:brightness-110 transition-colors"
              title="Fine-tune this model on a dataset"
            >
              <Wrench className="w-4 h-4" />
              Fine-tune
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelDetailPanel;
